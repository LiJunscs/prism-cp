"""Joint placement and microbatch search; never calls or conditions on N.

Exact mode enumerates all sample subsets, dyadic rank partitions and assignments
with ascending sample order within a pack. Layer and MLP placements are jointly
enumerated. Bounded mode searches several complete pool packings and beam paths;
its answer is an incumbent, never an optimality certificate.
"""

from dataclasses import dataclass
from itertools import product
from functools import lru_cache
import math

from .ir import Group, Layout, Microbatch, Plan, validate_layout, validate_plan
from .model import Evaluator, resource_metrics
from .response import H100_SCHEMA, response_coverage
from .memory_selection import prefer_lower_memory


@dataclass(frozen=True)
class SearchOptions:
    mode: str = "bounded"
    beam: int = 8
    layouts: int = 8
    max_states: int = 1_000_000
    node_affinity: bool = False
    homogeneous_packing: bool = False
    balanced_packing: bool = False
    heavy_tail_packing: bool = False
    max_attention_cp: int | None = None
    memory_time_tolerance: float = 0.001

    def __post_init__(self):
        if self.mode not in ("exact", "bounded") or min(self.beam, self.layouts, self.max_states) < 1:
            raise ValueError("Invalid search options")
        if self.max_attention_cp is not None and (self.max_attention_cp < 1 or self.max_attention_cp & (self.max_attention_cp-1)):
            raise ValueError("Attention CP ceiling must be a positive power of two")
        if not math.isfinite(self.memory_time_tolerance) or self.memory_time_tolerance < 0:
            raise ValueError("Memory time tolerance must be finite and nonnegative")


class SearchLimit(RuntimeError):
    """No optimality claim is valid after this exception."""


@lru_cache(None)
def rank_partitions(start, width):
    """All aligned dyadic partitions, each covering the world exactly once."""
    whole = ((tuple(range(start, start + width)),),)
    if width == 1:
        return whole
    half = width // 2
    return whole + tuple(a + b for a in rank_partitions(start, half)
                         for b in rank_partitions(start + half, half))


def layout_candidates(ids, lengths, world, exact, limit, ranks_per_node=None, balanced_packing=False):
    ids = tuple(sorted(ids))
    result, seen = [], set()

    def add(groups):
        layout = Layout(tuple(Group(tuple(ranks), tuple(sorted(samples))) for ranks, samples in groups), "zigzag")
        try:
            validate_layout(layout, lengths, world)
        except ValueError:
            return
        if layout not in seen:
            seen.add(layout)
            result.append(layout)

    if exact:
        for partition in rank_partitions(0, world):
            if len(partition) > len(ids):
                continue
            for assignment in product(range(len(partition)), repeat=len(ids)):
                packs = [[] for _ in partition]
                for sample, group in zip(ids, assignment):
                    packs[group].append(sample)
                add(zip(partition, packs))
        return result

    if ranks_per_node is not None and (ranks_per_node < 1 or ranks_per_node & (ranks_per_node - 1)):
        raise ValueError("Node-affine candidate groups require a power-of-two node size")

    def assign(order, partition):
        packs, loads = [[] for _ in partition], [0.0] * len(partition)
        for sample in order:
            legal = [j for j, ranks in enumerate(partition)
                     if len(ranks) == 1 or lengths[sample] % (2 * len(ranks)) == 0]
            if not legal:
                return None
            j = min(legal, key=lambda j: (bool(packs[j]), loads[j], j))
            packs[j].append(sample)
            loads[j] += lengths[sample] / len(partition[j])
        return packs

    # Heavy sample CPp plus CP1 tails was missing from the old partition pool.
    # Use the same family for both dimensions, independent of operator scores.
    if balanced_packing:
        width = world // 2
        while width > 1:
            for start in (0, world-width):
                big = tuple(range(start, start+width))
                partition = (big,) + tuple((r,) for r in range(world) if r not in big)
                if len(partition) <= len(ids):
                    packs = assign(sorted(ids, key=lambda s: (-lengths[s], s)), partition)
                    if packs is not None:
                        add(sorted(zip(partition, packs)))
            width //= 2

    # Enumerate cheap partitions without constructing the exponential dyadic family.
    partitions = []
    width = world
    while width:
        partition = tuple(tuple(range(i, i + width)) for i in range(0, world, width))
        partitions.append(partition)
        if width > 1 and len(partition) < len(ids):
            for selected in (0, len(partition) - 1):
                ranks = partition[selected]
                half = width // 2
                partitions.append(partition[:selected] + (ranks[:half], ranks[half:]) + partition[selected + 1:])
        width //= 2
    for partition in partitions:
        if len(partition) > len(ids):
            continue
        for order in (sorted(ids, key=lambda s: (-lengths[s], s)), list(ids), list(reversed(ids))):
            if (ranks_per_node is not None and world > ranks_per_node
                    and all(r[0] // ranks_per_node == r[-1] // ranks_per_node for r in partition)):
                # Freeze only node residency for this candidate family, not CP or
                # per-layer rank placement. Reuse it across CP widths so changing
                # node-local CP does not accidentally shuffle half the pool over IB.
                nodes = tuple(tuple(range(start, min(start + ranks_per_node, world)))
                              for start in range(0, world, ranks_per_node))
                buckets = assign(order, nodes)
                packs = []
                if buckets is not None:
                    for node, samples in zip(nodes, buckets):
                        subgroups = tuple(r for r in partition if r[0] in node)
                        placed = assign(samples, subgroups)
                        if placed is None:
                            packs = None
                            break
                        packs.extend(placed)
                else:
                    packs = None
            else:
                packs = assign(order, partition)
            if packs is not None:
                add(zip(partition, packs))
    if len(result) <= limit:
        return result
    # Retain uniform CP choices first; then heterogeneous placements. No GQA costs.
    uniform = [x for x in result if len({len(g.ranks) for g in x.groups}) == 1]
    varied = [x for x in result if x not in uniform]
    selected = []
    widths = set()
    for layout in uniform:
        width = len(layout.groups[0].ranks)
        if width not in widths:
            selected.append(layout)
            widths.add(width)
    # Interleave placement alternatives with heterogeneous CP.
    remaining = varied + [x for x in uniform if x not in selected]
    return (selected + remaining)[:limit]


@dataclass(frozen=True)
class _State:
    cost: float
    live: tuple[int, ...]
    peak: tuple[int, ...]
    ingress: Layout
    previous: Layout
    layers: tuple


class _Search:
    def __init__(self, lengths, world, pattern, model, options):
        self.lengths, self.world, self.pattern = lengths, world, pattern
        self.model, self.options = model, options
        self.states = 0
        self.truncated = False
        self.cache = {}
        self.incumbent = None
        self.packings_examined = 0

    def batch(self, ids):
        # Shape memoization keeps repeated 16Mi templates inexpensive. Stable sample
        # identities are restored after solving; no source-content hashes are used.
        ids = tuple(sorted(ids))
        shape = tuple(self.lengths[i] for i in ids)
        key = shape
        if key in self.cache:
            answer = self.cache[key]
        else:
            local = dict(enumerate(shape))
            answer = self._batch(local)
            self.cache[key] = answer
        if answer is None:
            return None
        cost, batch = answer
        def remap(layout):
            return Layout(tuple(Group(g.ranks, tuple(ids[s] for s in g.samples)) for g in layout.groups), layout.mode)
        return cost, Microbatch(remap(batch.ingress), tuple((remap(a), remap(m)) for a, m in batch.layers))

    def _batch(self, lengths):
        opt = self.options
        if sum(lengths.values()) > self.model.token_capacity_upper_bound(self.pattern, self.world):
            return None
        candidates = layout_candidates(lengths, lengths, self.world, opt.mode == "exact", opt.layouts,
                                       self.model.ranks_per_node if opt.node_affinity else None, opt.balanced_packing)
        ev = Evaluator(self.model, lengths, self.world, self.pattern)
        # The MLP can stay in attention ownership or choose any candidate placement.
        mlps = tuple(dict.fromkeys(x.with_mode(mode) for x in candidates for mode in ("contiguous", "zigzag")))
        attention_candidates = [x for x in candidates if opt.max_attention_cp is None
                                or all(len(g.ranks) <= opt.max_attention_cp for g in x.groups)]
        if self.model.cp_response and self.model.cp_response.get("schema") == H100_SCHEMA:
            # Attention CP is limited by this hardware profile's actual anchors.
            # This does not restrict MLP placement or independent packing search.
            max_measured_cp = max(map(int, self.model.cp_response["anchors"]["GQA"]["milliseconds"]))
            attention_candidates = [x for x in attention_candidates
                                    if all(len(g.ranks) <= max_measured_cp for g in x.groups)]
        states = [_State(0, ev.initial(x), ev.initial(x), x, x, ()) for x in candidates
                  if max(ev.initial(x)) <= self.model.budget]
        for symbol in self.pattern:
            next_states = []
            for state in states:
                for template in attention_candidates:
                    attention = template.with_mode("contiguous" if symbol == "G" else "zigzag")
                    for mlp in mlps:
                        self.states += 1
                        if self.states > opt.max_states:
                            raise SearchLimit("Search state budget exhausted; no global optimum is certified")
                        cost, live, here = ev.layer(state.live, state.previous, attention, mlp, symbol)
                        peak = tuple(max(a, b) for a, b in zip(state.peak, here))
                        if max(peak) <= self.model.budget:
                            next_states.append(_State(state.cost + cost, live, peak, state.ingress,
                                                      attention, state.layers + ((attention, mlp),)))
            if opt.mode == "bounded" and len(next_states) > opt.beam:
                self.truncated = True
                # Preserve low-memory alternatives alongside the fastest prefixes.
                fastest = sorted(next_states, key=lambda x: (x.cost, max(x.peak)))
                safest = sorted(next_states, key=lambda x: (max(x.live), x.cost))
                keep = {}
                for a, b in zip(fastest, safest):
                    for state in (a, b):
                        signature = (state.ingress, state.previous, state.live)
                        old = keep.get(signature)
                        if old is None or state.cost < old.cost:
                            keep[signature] = state
                        if len(keep) >= opt.beam:
                            break
                    if len(keep) >= opt.beam:
                        break
                next_states = list(keep.values())
            states = next_states
        best = None
        for state in states:
            cost, head = ev.finish(state.live, state.previous, state.ingress)
            if max(head) <= self.model.budget:
                row = (state.cost + cost, Microbatch(state.ingress, state.layers))
                if best is None or row[0] < best[0]:
                    best = row
        return best

    def exact(self):
        ids = tuple(sorted(self.lengths))
        @lru_cache(None)
        def visit(mask):
            if not mask:
                return 0.0, ()
            first = mask & -mask
            subset, best = mask, None
            while subset:
                if subset & first:
                    chosen = tuple(ids[i] for i in range(len(ids)) if subset & (1 << i))
                    batch = self.batch(chosen)
                    if batch is not None:
                        rest = visit(mask ^ subset)
                        if rest is not None:
                            row = batch[0] + rest[0], (batch[1],) + rest[1]
                            if best is None or row[0] < best[0]:
                                best = row
                subset = (subset - 1) & mask
            return best
        return visit((1 << len(ids)) - 1)

    def bounded(self):
        m = self.model
        # Pool packing proposals are derived from saved-activation headroom. This
        # is a search seed, never a capacity/admission constraint on any stage.
        ev = Evaluator(m, self.lengths, self.world, self.pattern)
        h, f, d = m.hidden, m.ffn, m.gdn_heads * m.gdn_dim
        per_token = 4 * h + len(self.pattern) * 2 * (12 * h + 3 * f + max(5 * d, h + 2 * m.kv_heads * m.head_dim))
        nominal = max(1, int((m.budget - ev.base) / per_token)) * self.world
        targets = {max(max(self.lengths.values()), int(nominal * scale)) for scale in (0.5, 0.8, 1.0, 1.3)}
        # Include balanced pack boundaries instead of letting arbitrary seed
        # rounding force an extra sample onto one rank (especially uniform pools).
        quantum = self.world * math.gcd(*self.lengths.values())
        targets.update(max(max(self.lengths.values()), (target // quantum) * quantum) for target in tuple(targets))
        targets = sorted(targets)
        orders = [tuple(sorted(self.lengths)), tuple(sorted(self.lengths, key=lambda s: (-self.lengths[s], s)))]
        proposals = []
        for order in orders:
            for target in targets:
                packs, pack, tokens = [], [], 0
                for sample in order:
                    if pack and tokens + self.lengths[sample] > target:
                        packs.append(tuple(sorted(pack)))
                        pack, tokens = [], 0
                    pack.append(sample)
                    tokens += self.lengths[sample]
                if pack:
                    packs.append(tuple(sorted(pack)))
                proposals.append(tuple(packs))
        if self.options.homogeneous_packing:
            buckets = {}
            for sample in sorted(self.lengths):
                buckets.setdefault(self.lengths[sample], []).append(sample)
            for target in targets:
                packs = []
                for length, samples in sorted(buckets.items(), reverse=True):
                    count = max(1, target // length)
                    packs.extend(tuple(samples[i:i + count]) for i in range(0, len(samples), count))
                proposals.append(tuple(packs))
        if self.options.balanced_packing:
            # Equal linear work: a CPp heavy sample paired with approximately
            # (world-p)/p times its tokens on CP1 ranks. The rest remains freely
            # repacked using every existing memory-derived target.
            order = sorted(self.lengths, key=lambda s: (-self.lengths[s], s))
            anchor, tail = order[0], sorted(order[1:], key=lambda s: (self.lengths[s], s))
            width = self.world // 2
            balanced = []
            while width:
                if width == 1 or self.lengths[anchor] % (2*width) == 0:
                    goal = self.lengths[anchor]*(self.world-width)/width
                    selected, tokens = [], 0
                    for sample in tail:
                        if tokens + self.lengths[sample] <= goal:
                            selected.append(sample)
                            tokens += self.lengths[sample]
                    if len(selected) >= self.world-width:
                        paired = frozenset((anchor, *selected))
                        rest = tuple(s for s in order if s not in paired)
                        for target in targets:
                            packs, pack, total = [tuple(sorted(paired))], [], 0
                            for sample in rest:
                                if pack and total+self.lengths[sample] > target:
                                    packs.append(tuple(sorted(pack)))
                                    pack, total = [], 0
                                pack.append(sample)
                                total += self.lengths[sample]
                            if pack:
                                packs.append(tuple(sorted(pack)))
                            balanced.append(tuple(packs))
                width //= 2
            # Try the known coverage correction early enough to survive budgets.
            proposals = balanced + proposals
        if self.options.heavy_tail_packing:
            # Offer multiple numbers of large samples per batch and several
            # allocations of the common short pool. No recipe boundary is an
            # input constraint, and homogeneous/ordinary packings stay eligible.
            ceiling = m.token_capacity_upper_bound(self.pattern, self.world)
            largest, smallest = max(self.lengths.values()), min(self.lengths.values())
            heavy = sorted((s for s in self.lengths if self.lengths[s] >= largest/2),
                           key=lambda s: (-self.lengths[s], s))
            heavy_set = set(heavy)
            tail = sorted((s for s in self.lengths if s not in heavy_set), key=lambda s: (self.lengths[s], s))
            new = []
            if tail and largest >= 4*smallest:
                for count in range(1, min(len(heavy), self.world//m.ranks_per_node, ceiling//largest)+1):
                    blocks = [heavy[i:i+count] for i in range(0, len(heavy), count)]
                    allocations = {0, 1, 2, 4, 8, 16, 32, len(tail)//len(blocks)}
                    for extra in sorted(allocations):
                        for target in targets:
                            packs, cursor = [], 0
                            for block in blocks:
                                pack, total = list(block), sum(self.lengths[s] for s in block)
                                for _ in range(extra):
                                    if cursor == len(tail) or total+self.lengths[tail[cursor]] > ceiling:
                                        break
                                    pack.append(tail[cursor]); total += self.lengths[tail[cursor]]; cursor += 1
                                packs.append(tuple(sorted(pack)))
                            pack, total = [], 0
                            for sample in tail[cursor:]:
                                if pack and total+self.lengths[sample] > target:
                                    packs.append(tuple(sorted(pack))); pack, total = [], 0
                                pack.append(sample); total += self.lengths[sample]
                            if pack:
                                packs.append(tuple(sorted(pack)))
                            new.append(tuple(packs))
            proposals = new + proposals
        seen, best = set(), None
        for signature in proposals:
            if signature in seen:
                continue
            seen.add(signature)
            self.packings_examined += 1
            rows = []
            for ids in signature:
                row = self.batch(ids)
                if row is None:
                    break
                rows.append(row)
            else:
                answer = sum(row[0] for row in rows), tuple(row[1] for row in rows)
                if best is None or answer[0] < best[0]:
                    best = answer
                    self.incumbent = best
        return best


def solve(lengths, world, pattern, model, options=SearchOptions()):
    if (not lengths or world <= 0 or world & (world - 1) or not pattern
            or set(pattern) - set("G*") or any(type(s) is not int or type(n) is not int or n <= 0 for s, n in lengths.items())):
        raise ValueError("Invalid lengths, world or pattern")
    if options.mode == "exact" and (len(lengths) > 6 or world > 4):
        raise ValueError("Exact domain is limited to <=6 samples and <=4 ranks; use bounded mode")
    search = _Search(lengths, world, pattern, model, options)
    method = search.exact if options.mode == "exact" else search.bounded
    limit_hit = False
    try:
        answer = method()
    except SearchLimit:
        if options.mode == "exact":
            raise
        limit_hit = True
        answer = search.incumbent
    metadata = dict(search_mode=options.mode, states_examined=search.states,
                    state_budget_exhausted=limit_hit, pool_packings_examined=search.packings_examined,
                    microbatch_shapes_solved=len(search.cache),
                    beam_pruned=search.truncated,
                    domain="aligned dyadic groups; ascending sample order per pack; all ranks nonempty; no padding; fixed G/* kernel modes; independent MLP placement",
                    objective="minimize allocated GPU-ms for fixed useful input work; fixed world makes this equivalent to sum complete F+B microbatch times plus update/sync; all allocated ranks and idle waits counted",
                    model_provenance=model.provenance, gpu_admitted=False)
    metadata["max_attention_cp"] = options.max_attention_cp
    if answer is None:
        return dict(metadata, status=("search_budget_exhausted" if limit_hit else
                    "infeasible_in_declared_domain" if options.mode == "exact" else "no_feasible_plan_found"), plan=None)
    plan = Plan(world, pattern, answer[1])
    validate_plan(plan, lengths)
    evidence = [Evaluator(model, lengths, world, pattern).evaluate(b) for b in plan.microbatches]
    if not all(row["fits"] for row in evidence):
        raise AssertionError("Solver and full-plan memory accounting disagree")
    recomputed = sum(row["predicted_ms"] for row in evidence) + model.update_ms
    if not math.isclose(recomputed, answer[0] + model.update_ms, rel_tol=1e-10, abs_tol=1e-8):
        raise AssertionError("Solver and full-plan objective disagree")
    if options.mode == "bounded":
        plan, metadata["memory_selection"] = prefer_lower_memory(plan, lengths, model, options.memory_time_tolerance)
        validate_plan(plan, lengths)
        evidence = [Evaluator(model, lengths, world, pattern).evaluate(b) for b in plan.microbatches]
        recomputed = sum(row["predicted_ms"] for row in evidence) + model.update_ms
        metadata["objective"] += "; bounded CP1 memory tie-break within declared time tolerance"
    return dict(metadata, status="optimal_in_declared_domain" if options.mode == "exact" else "best_found_no_optimality_certificate",
                plan=plan.to_dict(), predicted_update_ms=recomputed,
                resource_metrics=resource_metrics(lengths, world, pattern, model, recomputed),
                cp_response_evidence=response_coverage(plan, lengths, model),
                budget_bytes=model.budget, microbatch_evidence=evidence)
