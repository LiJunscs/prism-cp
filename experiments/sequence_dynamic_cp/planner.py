"""Finite one-dimensional CP search with explicit execution constraints.

All groups remain fixed across layers. ProxyCost is a conditional compute
model, NOT a calibrated CP timing model. ExactCosts accepts complete measured
step plans only, so independent module timings cannot be silently composed.
"""

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

from native import replay


@dataclass(frozen=True)
class Group:
    ranks: tuple[int, ...]
    ids: tuple[int, ...]


Plan = tuple[tuple[Group, ...], ...]


@dataclass(frozen=True)
class Constraints:
    world: int
    token_cap: int
    min_cp: int = 1
    max_sequences: int | None = None
    alignment: int = 1

    def __post_init__(self):
        for value in (self.world, self.min_cp):
            if value < 1 or value & (value - 1):
                raise ValueError("First version requires power-of-two world and min_cp")
        if self.min_cp > self.world or self.token_cap < 1 or self.alignment < 1:
            raise ValueError("Invalid CP/token/alignment constraint")
        if self.max_sequences is not None and self.max_sequences < 1:
            raise ValueError("max_sequences must be positive")


def from_rank_ids(batches: list[list[list[int]]]) -> Plan:
    """Decode the executor's replicated sample-ID representation."""
    result = []
    for batch in batches:
        groups = {}
        for rank, ids in enumerate(batch):
            groups.setdefault(tuple(ids), []).append(rank)
        result.append(tuple(Group(tuple(ranks), ids) for ids, ranks in groups.items()))
    return tuple(result)


def to_rank_ids(plan: Plan, world: int) -> list[list[list[int]]]:
    """Encode a plan for unchanged rerouting and packed-batch construction."""
    result = []
    for batch in plan:
        ranks = [[] for _ in range(world)]
        for group in batch:
            for rank in group.ranks:
                ranks[rank] = list(group.ids)
        result.append(ranks)
    return result


def validate(plan: Plan, lengths: dict[int, int], limits: Constraints) -> None:
    """Check group legality, local capacity and exactly-once sample coverage."""
    if any(
        not isinstance(i, int) or not isinstance(n, int) or n <= 0 or n % limits.alignment for i, n in lengths.items()
    ):
        raise ValueError("IDs/lengths must be integers; lengths must be positive and aligned")
    seen = []
    if not plan or not lengths:
        raise ValueError("Empty steps are unsupported")
    for batch in plan:
        ranks = []
        for group in batch:
            cp = len(group.ranks)
            if not cp or cp & (cp - 1) or cp < limits.min_cp or not group.ids:
                raise ValueError("Empty or unsupported CP group")
            start = group.ranks[0]
            if start % cp or group.ranks != tuple(range(start, start + cp)):
                raise ValueError("CP group must match a precreated aligned contiguous group")
            if any(i not in lengths for i in group.ids):
                raise ValueError("Unknown sample ID")
            if any(lengths[i] % (2 * cp) for i in group.ids) and cp > 1:
                raise ValueError("THD zigzag requires sequence padding divisible by 2*CP")
            if sum(lengths[i] for i in group.ids) > limits.token_cap * cp:
                raise ValueError("Per-rank token capacity exceeded")
            if limits.max_sequences is not None and len(group.ids) > limits.max_sequences:
                raise ValueError("Real packed-sequence capacity exceeded")
            ranks.extend(group.ranks)
            seen.extend(group.ids)
        if sorted(ranks) != list(range(limits.world)):
            raise ValueError("All ranks must execute exactly one nonempty group per microbatch")
    if sorted(seen) != sorted(lengths):
        raise ValueError("Samples must appear exactly once at group level across the full step")


@dataclass(frozen=True)
class ProxyCost:
    name: str
    a: float
    b: float
    provenance: str = "conditional_CP1_compute_divided_by_CP"

    def __post_init__(self):
        if not all(math.isfinite(v) and v >= 0 for v in (self.a, self.b)) or self.a + self.b == 0:
            raise ValueError("Cost coefficients must be finite, nonnegative and nonzero")

    @property
    def beta_tokens(self) -> float:
        if self.a <= 0:
            raise ValueError("Weighted native ablation requires positive quadratic coefficient")
        return self.b / self.a * 1024

    def group(self, group: Group, lengths: dict[int, int]) -> dict:
        cp = len(group.ranks)
        q = sum((lengths[i] / 1024) ** 2 for i in group.ids) / cp
        ell = sum(lengths[i] / 1024 for i in group.ids) / cp
        return dict(
            q_ki2=q,
            l_ki=ell,
            quadratic=self.a * q,
            linear=self.b * ell,
            cost=self.a * q + self.b * ell,
        )

    def evaluate(self, plan: Plan, lengths: dict[int, int]) -> float:
        return sum(max(self.group(g, lengths)["cost"] for g in batch) for batch in plan)


def fingerprint(plan: Plan, lengths: dict[int, int]) -> str:
    """Key exact packed shapes, sample order, placement and microbatch order."""
    shape = [[dict(ranks=g.ranks, lengths=[lengths[i] for i in g.ids]) for g in batch] for batch in plan]
    return hashlib.sha256(json.dumps(shape, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class ExactCosts:
    """Complete-plan timings frozen for one model/runtime/topology contract.

    Unknown, failed numerical gates and capacity failures cannot win selection.
    Records may include optimizer/sync; the scope is mandatory and exported.
    """

    def __init__(self, data: dict, contract: dict):
        if data["contract"] != contract:
            raise ValueError("Measured costs do not match model/backend/topology contract")
        self.scope = data["scope"]
        self.records = {}
        seen = set()
        for row in data["records"]:
            key = row["fingerprint"]
            if key in seen:
                raise ValueError("Duplicate exact-plan measurement")
            seen.add(key)
            if row["status"] == "ok" and row["numerical_pass"] is True:
                value = row["milliseconds"]
                if not math.isfinite(value) or value <= 0:
                    raise ValueError("Invalid complete-plan measurement")
                self.records[key] = row

    def evaluate(self, plan: Plan, lengths: dict[int, int]) -> float | None:
        row = self.records.get(fingerprint(plan, lengths))
        return row["milliseconds"] if row else None


def native_plan(lengths: dict[int, int], limits: Constraints, cost: ProxyCost) -> Plan:
    plan = from_rank_ids(
        replay(
            list(lengths.items()),
            limits.world,
            limits.token_cap,
            limits.min_cp,
            cost.beta_tokens,
        )
    )
    validate(plan, lengths, limits)
    return plan


def _fits(ids: tuple[int, ...], cp: int, lengths: dict[int, int], limits: Constraints) -> bool:
    return (
        bool(ids)
        and sum(lengths[i] for i in ids) <= cp * limits.token_cap
        and (limits.max_sequences is None or len(ids) <= limits.max_sequences)
        and (cp == 1 or all(lengths[i] % (2 * cp) == 0 for i in ids))
    )


def pack_layout(
    lengths: dict[int, int],
    limits: Constraints,
    sizes: tuple[int, ...],
    cost: ProxyCost,
) -> Plan:
    """LPT pack a full step into a legal common layout with full-group tail.

    Empty groups are forbidden by the executor; a partial tail is repacked on
    the full group, rather than duplicating work or introducing dummy samples.
    """
    if sum(sizes) != limits.world:
        raise ValueError("Layout must cover the world")
    remaining = sorted(lengths, key=lambda i: (-lengths[i], i))
    result = []
    while remaining:
        groups, start = [], 0
        for cp in sizes:
            groups.append(Group(tuple(range(start, start + cp)), ()))
            start += cp
        leftovers = []
        costs = [0.0] * len(groups)
        for i in remaining:
            options = []
            for j, g in enumerate(groups):
                ids = (*g.ids, i)
                if _fits(ids, len(g.ranks), lengths, limits) and (
                    not hasattr(cost, "feasible_group") or cost.feasible_group(ids, len(g.ranks), lengths)
                ):
                    new = cost.group(Group(g.ranks, ids), lengths)["cost"]
                    objective = max(new, max((v for k, v in enumerate(costs) if k != j), default=0))
                    options.append((objective, costs[j], j, new))
            if not options:
                leftovers.append(i)
                continue
            _, _, j, new = min(options)
            groups[j] = Group(groups[j].ranks, (*groups[j].ids, i))
            costs[j] = new
        if any(not g.ids for g in groups):
            selected, leftovers = [], []
            for i in remaining:
                if _fits(tuple([*selected, i]), limits.world, lengths, limits):
                    selected.append(i)
                else:
                    leftovers.append(i)
            groups = [Group(tuple(range(limits.world)), tuple(selected))]
        if len(leftovers) == len(remaining):
            raise ValueError("No feasible group for the next sequence")
        result.append(tuple(groups))
        remaining = leftovers
    plan = tuple(result)
    validate(plan, lengths, limits)
    return plan


def layouts(limits: Constraints, seeds: list[Plan]) -> list[tuple[int, ...]]:
    """Homogeneous controls plus bounded dyadic neighbors of native layouts."""
    found = set()
    cp = limits.min_cp
    while cp <= limits.world:
        found.add((cp,) * (limits.world // cp))
        cp *= 2
    for plan in seeds:
        for batch in plan:
            sizes = tuple(len(g.ranks) for g in batch)
            found.add(sizes)
            for j, size in enumerate(sizes):
                if size > limits.min_cp:
                    found.add((*sizes[:j], size // 2, size // 2, *sizes[j + 1 :]))
                if j + 1 < len(sizes) and size == sizes[j + 1] and sum(sizes[:j]) % (2 * size) == 0:
                    found.add((*sizes[:j], 2 * size, *sizes[j + 2 :]))
    return sorted(found)


def _neighbors(plan: Plan, lengths: dict[int, int], limits: Constraints):
    locations = [(m, j) for m, batch in enumerate(plan) for j in range(len(batch))]

    def replace(changes):
        return tuple(tuple(changes.get((m, j), g) for j, g in enumerate(batch)) for m, batch in enumerate(plan))

    # Moves within/across microbatches; retain a nonempty source group.
    for source in locations:
        src = plan[source[0]][source[1]]
        if len(src.ids) < 2:
            continue
        for i in src.ids:
            for target in locations:
                if source == target:
                    continue
                dst = plan[target[0]][target[1]]
                ids = (*dst.ids, i)
                if _fits(ids, len(dst.ranks), lengths, limits):
                    yield replace(
                        {
                            source: Group(src.ranks, tuple(k for k in src.ids if k != i)),
                            target: Group(dst.ranks, ids),
                        }
                    )
    # Swaps allow improvement at exact token-cap boundaries.
    for x, source in enumerate(locations):
        src = plan[source[0]][source[1]]
        for target in locations[x + 1 :]:
            dst = plan[target[0]][target[1]]
            for i in src.ids:
                for k in dst.ids:
                    if lengths[i] == lengths[k]:
                        continue
                    a = tuple(k if n == i else n for n in src.ids)
                    b = tuple(i if n == k else n for n in dst.ids)
                    if _fits(a, len(src.ranks), lengths, limits) and _fits(b, len(dst.ranks), lengths, limits):
                        yield replace({source: Group(src.ranks, a), target: Group(dst.ranks, b)})


def search(
    lengths: dict[int, int],
    limits: Constraints,
    cost: ProxyCost,
    local_evaluations: int = 128,
    rounds: int = 2,
) -> dict:
    """Generate finite legal plans; preserve native and weighted seeds.

    The deterministic evaluation budget, rather than a wall-clock cutoff,
    ensures all ranks compute identical schedules. Planning time is measured.
    """
    start = time.perf_counter()
    if local_evaluations < 0 or rounds < 0:
        raise ValueError("Search budgets must be nonnegative")
    candidates, failures = {}, []

    def add(name, plan):
        validate(plan, lengths, limits)
        key = fingerprint(plan, lengths)
        if key not in candidates:
            candidates[key] = dict(
                name=name,
                plan=plan,
                proxy_cost=cost.evaluate(plan, lengths),
                fingerprint=key,
            )

    seeds = []
    for name, model in (
        ("native_square", ProxyCost("square", 1, 0)),
        ("native_weighted", getattr(cost, "seed_proxy", cost)),
    ):
        try:
            plan = native_plan(lengths, limits, model)
            seeds.append(plan)
            add(name, plan)
        except (ValueError, AssertionError) as error:
            failures.append(dict(candidate=name, reason=str(error)))
    for sizes in layouts(limits, seeds):
        try:
            add(
                "layout_" + "x".join(map(str, sizes)),
                pack_layout(lengths, limits, sizes, cost),
            )
        except ValueError as error:
            failures.append(dict(candidate=str(sizes), reason=str(error)))
    if not candidates:
        raise ValueError("No legal candidate for this complete step")

    def improves(value, current):
        return value < current - 1e-9 * max(1.0, abs(current))

    # Preserve insertion order (native first) for numerical ties. A 1-ULP
    # reduction must not be reported as an architecture-dependent decision.
    best = next(iter(candidates.values()))
    for candidate in candidates.values():
        if improves(candidate["proxy_cost"], best["proxy_cost"]):
            best = candidate
    evaluations = 0
    for _ in range(rounds):
        previous = best
        for plan in _neighbors(previous["plan"], lengths, limits):
            if evaluations >= local_evaluations:
                break
            evaluations += 1
            try:
                add("local_move_or_swap", plan)
            except ValueError as error:
                failures.append(dict(candidate="local_move_or_swap", reason=str(error)))
                continue
            value = cost.evaluate(plan, lengths)
            if improves(value, best["proxy_cost"]):
                best = candidates[fingerprint(plan, lengths)]
        if best is previous or evaluations >= local_evaluations:
            break
    # A separate pass is essential: sample moves cannot split the winning
    # rank domains, and must not exhaust the budget before we try CP1 packs.
    from short_sequence import refine_short_sequences
    refinement = refine_short_sequences(best['plan'], lengths, limits, cost)
    if refinement['plan'] != best['plan']:
        add('short_sequence_split', refinement['plan'])
        # The refinement gate requires a strict complete-plan improvement.
        best = candidates[fingerprint(refinement['plan'], lengths)]
    return dict(
        selected=best,
        candidates=list(candidates.values()),
        failures=failures,
        local_evaluations=evaluations,
        short_sequence_refinement={k: v for k, v in refinement.items() if k != 'plan'},
        planning_ms=(time.perf_counter() - start) * 1000,
        selection_provenance=cost.provenance,
        parameter_sync_optimizer="unknown",
    )


def select(
    result: dict,
    lengths: dict[int, int],
    measured: ExactCosts | None = None,
    allow_proxy: bool = False,
) -> tuple[dict, str]:
    """Select using matching measurements, or explicit experimental proxy opt-in."""
    native = next((c for c in result["candidates"] if c["name"] == "native_square"), None)
    if measured:
        known = [(measured.evaluate(c["plan"], lengths), j, c) for j, c in enumerate(result["candidates"])]
        known = [(v, j, c) for v, j, c in known if v is not None]
        # An unknown native cannot be claimed slower by using an unrelated table.
        if known and native and measured.evaluate(native["plan"], lengths) is not None:
            return (
                min(known, key=lambda row: (row[0], row[1]))[2],
                "measured_complete_plan",
            )
    if allow_proxy:
        return result["selected"], "experimental_compute_proxy"
    if native is None:
        raise ValueError("Native plan is illegal; no calibrated fallback exists")
    return native, "native_fallback_unknown_distributed_cost"


def export_plan(plan: Plan, lengths: dict[int, int], cost: ProxyCost, node_size: int) -> dict:
    """Export complete ownership, cost components and per-rank Q/L geometry."""
    batches = []
    for batch in plan:
        groups = []
        rank_q, rank_l = [], []
        for group in batch:
            parts = cost.group(group, lengths)
            rank_q.extend([parts["q_ki2"]] * len(group.ranks))
            rank_l.extend([parts["l_ki"]] * len(group.ranks))
            groups.append(
                dict(
                    ranks=group.ranks,
                    sample_ids=group.ids,
                    padded_lengths=[lengths[i] for i in group.ids],
                    cp=len(group.ranks),
                    placement=("intra_node" if len({r // node_size for r in group.ranks}) == 1 else "cross_node"),
                    local_tokens=sum(lengths[i] for i in group.ids) / len(group.ranks),
                    **parts
                )
            )
        batches.append(
            dict(
                groups=groups,
                cost=max(g["cost"] for g in groups),
                r_q=max(rank_q) / (sum(rank_q) / len(rank_q)),
                r_l=max(rank_l) / (sum(rank_l) / len(rank_l)),
            )
        )
    return dict(
        microbatches=batches,
        compute_proxy_cost=sum(b["cost"] for b in batches),
        fingerprint=fingerprint(plan, lengths),
        provenance=cost.provenance,
        memory="token_cap_only; measured_peak_required",
        parameter_sync_optimizer="unknown",
    )


def models(path: Path) -> dict[str, ProxyCost]:
    """Read the existing frozen four-block GQA/GDN+MLP composition."""
    rows = json.loads(path.read_text())
    selected = {}
    for name, counts in (("gpt", (4, 0)), ("hybrid", (1, 3))):
        row = next(r for r in rows if (r["gqa_layers"], r["gdn_layers"]) == counts)
        selected[name] = ProxyCost(name, row["A_ms_per_ki2"], row["B_ms_per_ki"])
    return {
        "square": ProxyCost("square", 1.0, 0.0, "quadratic_work_proxy_Ki2"),
        **selected,
    }
