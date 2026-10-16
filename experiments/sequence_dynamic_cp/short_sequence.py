"""Bounded deparallelization of S groups; native N is never modified.

Keep rounds and parent rank domains and redistribute whole samples. Prefer
CP1 within feasible candidates, but adopt a change only when the complete-plan
objective strictly improves. Preserve the original plan on a score tie.
"""

import math
import time

from mixed_plan import coalesce_rounds
from planner import Group, _fits, validate


def refine_short_sequences(plan, lengths, limits, cost):
    started = time.perf_counter()
    validate(plan, lengths, limits)
    windows = coalesce_rounds(plan, lengths, limits.world, limits.token_cap,
                              limits.max_sequences)
    window_of = {i: w for w, rows in enumerate(windows) for i in rows}
    tokens = [[0] * limits.world for _ in windows]
    counts = [[0] * limits.world for _ in windows]
    for i, row in enumerate(plan):
        for g in row:
            for rank in g.ranks:
                tokens[window_of[i]][rank] += sum(lengths[k] for k in g.ids) // len(g.ranks)
                counts[window_of[i]][rank] += len(g.ids)

    times, weights, memo = {}, {}, {}
    attempted = 0
    proxy = getattr(cost, 'seed_proxy', cost)

    def elapsed(g):
        if g not in times:
            value = float(cost.group(g, lengths)['cost'])
            if not math.isfinite(value) or value < 0:
                raise ValueError('Invalid short-sequence group cost')
            times[g] = value
        return times[g]

    def preference(groups):
        # All changes only split rank domains. First retain as many eligible
        # tokens as possible at CP1, then reduce their remaining replication.
        short = [(sum(lengths[i] for i in g.ids if lengths[i] <= limits.token_cap),
                  len(g.ranks)) for g in groups]
        return (sum(n for n, cp in short if cp > 1),
                sum(n * (cp - 1) for n, cp in short),
                max(elapsed(g) for g in groups),
                sum(sum(lengths[i] for i in g.ids) * (len(g.ranks) - 1) for g in groups))

    def partition(parent, cp, placement):
        """Two deterministic whole-sample packings, with no model call per bin."""
        ranks = [parent.ranks[j:j + cp] for j in range(0, len(parent.ranks), cp)]
        if len(parent.ids) < len(ranks) or max(lengths[i] for i in parent.ids) > cp * limits.token_cap:
            return None
        bins, loads, work = [[] for _ in ranks], [0] * len(ranks), [0.] * len(ranks)
        for i in sorted(parent.ids, key=lambda k: (-lengths[k], k)):
            if i not in weights:
                weights[i] = proxy.group(Group((parent.ranks[0],), (i,)), lengths)['cost']
            options = [j for j in range(len(ranks))
                       if loads[j] + lengths[i] <= cp * limits.token_cap
                       and (limits.max_sequences is None or len(bins[j]) < limits.max_sequences)]
            if not options:
                return None
            # Reserve a sample for every child; idle ranks are not legal.
            empty = [j for j in options if not bins[j]]
            if empty:
                options = empty
            if placement == 'balanced':
                j = min(options, key=lambda j: (work[j], loads[j], j))
            else:
                j = min(options, key=lambda j: (-loads[j], work[j], j))
            bins[j].append(i)
            loads[j] += lengths[i]
            work[j] += weights[i]
        # Keep each child's original packed order, including layout semantics.
        order = {i: j for j, i in enumerate(parent.ids)}
        groups = tuple(Group(r, tuple(sorted(ids, key=order.__getitem__))) for r, ids in zip(ranks, bins))
        if any(not _fits(g.ids, cp, lengths, limits) or
               (hasattr(cost, 'feasible_group') and not cost.feasible_group(g.ids, cp, lengths))
               for g in groups):
            return None
        return groups

    def refine(parent):
        nonlocal attempted
        if parent in memo:
            return memo[parent]
        best = (parent,)
        before = elapsed(parent)
        cp = limits.min_cp
        while cp < len(parent.ranks):
            seen = set()
            for placement in ('balanced', 'best_fit'):
                children = partition(parent, cp, placement)
                if children is None or children in seen:
                    continue
                seen.add(children)
                attempted += 1
                # Recursive splitting also permits mixed CP1/CP2/CP4 layouts.
                candidate = tuple(g for child in children for g in refine(child))
                after = max(elapsed(g) for g in candidate)
                if after <= before + 1e-9 * max(1., abs(before)) and preference(candidate) < preference(best):
                    best = candidate
            cp *= 2
        memo[parent] = best
        return best

    result, split_groups, rejected_windows = [], 0, 0
    for i, row in enumerate(plan):
        new_row = []
        w = window_of[i]
        for parent in row:
            children = refine(parent)
            old_tokens = sum(lengths[k] for k in parent.ids) // len(parent.ranks)
            changes = [(r, sum(lengths[k] for k in g.ids) // len(g.ranks) - old_tokens,
                        len(g.ids) - len(parent.ids)) for g in children for r in g.ranks]
            fits = all(tokens[w][r] + dt <= limits.token_cap and
                       (limits.max_sequences is None or counts[w][r] + dc <= limits.max_sequences)
                       for r, dt, dc in changes)
            if not fits:
                rejected_windows += 1
                children = (parent,)
            else:
                for r, dt, dc in changes:
                    tokens[w][r] += dt
                    counts[w][r] += dc
            split_groups += children != (parent,)
            new_row.extend(children)
        result.append(tuple(new_row))
    result = tuple(result)
    validate(result, lengths, limits)
    before, after = cost.evaluate(plan, lengths), cost.evaluate(result, lengths)
    tolerance = 1e-9 * max(1., abs(before))
    if after > before + tolerance:
        raise ValueError('Short-sequence refinement increased the complete-plan objective')
    execution = coalesce_rounds(result, lengths, limits.world, limits.token_cap, limits.max_sequences)
    if len(execution) > len(windows):
        raise ValueError('Short-sequence refinement increased execution microbatches')
    candidate_score, candidate_splits = after, split_groups
    rejected_tie = result != plan and not after < before - tolerance
    if rejected_tie:
        # Local savings can be hidden behind another group's critical path.
        # A higher CP1 share alone does not justify changing execution shapes.
        result, after, split_groups, execution = plan, before, 0, windows
    return dict(plan=result, score_before=before, score_after=after,
                split_groups=split_groups, partition_candidates=attempted,
                candidate_score_after=candidate_score, candidate_split_groups=candidate_splits,
                rejected_objective_tie=rejected_tie,
                rejected_execution_windows=rejected_windows,
                execution_microbatches_before=len(windows), execution_microbatches_after=len(execution),
                planning_ms=(time.perf_counter() - started) * 1000,
                scope='strict complete source-objective improvement; ties preserve input plan; preserve mixed-window feasibility; no GPU timing or memory model')
