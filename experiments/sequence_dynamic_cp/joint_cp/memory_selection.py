"""Bounded memory tie-break for homogeneous, rank-resident CP1 work.

Keep distributed batches intact. Repack only balanced CP1 batches whose
attention and MLP ownership never changes. The time tolerance is relative to
the original complete update, not compounded across length families.
"""

import math

from .ir import Group, Layout, Microbatch, Plan, validate_plan
from .model import Evaluator


def prefer_lower_memory(plan, lengths, model, relative_time_tolerance=0.001):
    if not math.isfinite(relative_time_tolerance) or relative_time_tolerance < 0:
        raise ValueError("Memory selection requires a finite nonnegative time tolerance")
    validate_plan(plan, lengths)
    evaluator = Evaluator(model, lengths, plan.world, plan.pattern)
    cache = {}

    def evaluate(candidate):
        rows = []
        for batch in candidate.microbatches:
            if batch not in cache:
                cache[batch] = evaluator.evaluate(batch)
            rows.append(cache[batch])
        return (max(max(r["peak_bytes"]) for r in rows),
                sum(r["predicted_ms"] for r in rows) + model.update_ms,
                all(r["fits"] for r in rows))

    original_peak, original_ms, _ = evaluate(plan)
    deadline = original_ms * (1 + relative_time_tolerance)
    selected = plan
    best = evaluate(plan)
    families = {}
    for index, batch in enumerate(plan.microbatches):
        owners = batch.ingress.rank_ids(plan.world)
        layouts = [batch.ingress, *(x for pair in batch.layers for x in pair)]
        if (any(len(g.ranks) != 1 for x in layouts for g in x.groups)
                or any(x.rank_ids(plan.world) != owners for x in layouts)
                or len({len(row) for row in owners}) != 1):
            continue
        sizes = {lengths[s] for row in owners for s in row}
        if len(sizes) == 1:
            families.setdefault(next(iter(sizes)), []).append(index)

    candidates_examined = 0
    changes = []
    # Identify families by the original immutable batches so earlier replacements
    # cannot invalidate later indices.
    for length, indices in sorted(families.items()):
        old_batches = {plan.microbatches[i] for i in indices}
        queues = [[] for _ in range(plan.world)]
        for i in indices:
            for rank, ids in enumerate(plan.microbatches[i].ingress.rank_ids(plan.world)):
                queues[rank].extend(ids)
        maximum = max(len(g.samples) for b in old_batches for g in b.ingress.groups)
        # At most 17 cheap candidates per length family; no new layout search.
        capacities = sorted({1, *((maximum * i + 15) // 16 for i in range(1, 17))})
        family_best, chosen_capacity = selected, None
        for count in capacities:
            replacements = []
            for start in range(0, len(queues[0]), count):
                groups = tuple(Group((r,), tuple(ids[start:start + count])) for r, ids in enumerate(queues))
                ingress = Layout(groups, "zigzag")
                contiguous = ingress.with_mode("contiguous")
                replacements.append(Microbatch(ingress, tuple(
                    (contiguous if c == "G" else ingress, contiguous) for c in plan.pattern)))
            batches, inserted = [], False
            for batch in selected.microbatches:
                if batch in old_batches:
                    if not inserted:
                        batches.extend(replacements)
                        inserted = True
                else:
                    batches.append(batch)
            candidate = Plan(plan.world, plan.pattern, tuple(batches))
            peak, ms, fits = evaluate(candidate)
            candidates_examined += 1
            if fits and ms <= deadline and (peak, ms) < best[:2]:
                family_best, chosen_capacity = candidate, count
                best = peak, ms, fits
        selected = family_best
        if chosen_capacity is not None:
            changes.append(dict(sample_length=length, samples_per_rank=chosen_capacity,
                                tokens_per_rank=length * chosen_capacity))
    validate_plan(selected, lengths)
    return selected, dict(
        policy="min_peak_then_time_within_relative_update_tolerance",
        relative_time_tolerance=relative_time_tolerance,
        original_predicted_update_ms=original_ms, selected_predicted_update_ms=best[1],
        original_peak_bytes=original_peak, selected_peak_bytes=best[0],
        original_microbatches=len(plan.microbatches), selected_microbatches=len(selected.microbatches),
        candidates_examined=candidates_examined, changed=selected != plan, repacked_families=changes,
        scope="bounded homogeneous rank-resident CP1 repacking; distributed batches unchanged",
        global_optimality_certificate=False)
