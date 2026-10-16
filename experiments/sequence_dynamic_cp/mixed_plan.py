"""Compile legacy scheduling rounds into capacity-bounded mixed-CP microbatches.

Preserves every sequence's CP ownership and collective order. This is an
execution transformation, not a new search or a relabeling of historical plans.
"""


def coalesce_rounds(plan, lengths, world, token_cap, max_sequences=None):
    batches, current, tokens, counts = [], [], [0] * world, [0] * world
    for ordinal, groups in enumerate(plan):
        added, seqs = [0] * world, [0] * world
        for group in groups:
            total = sum(lengths[i] for i in group.ids)
            if total % len(group.ranks):
                raise ValueError("CP token count is not integral")
            for rank in group.ranks:
                added[rank] += total // len(group.ranks)
                seqs[rank] += len(group.ids)
        if any(n > token_cap for n in added):
            raise ValueError("Original round exceeds token capacity")
        exceeds = any(a + b > token_cap for a, b in zip(tokens, added)) or (
            max_sequences is not None and any(a + b > max_sequences for a, b in zip(counts, seqs))
        )
        if current and exceeds:
            batches.append(tuple(current))
            current, tokens, counts = [], [0] * world, [0] * world
        current.append(ordinal)
        tokens = [a + b for a, b in zip(tokens, added)]
        counts = [a + b for a, b in zip(counts, seqs)]
    if current:
        batches.append(tuple(current))
    return tuple(batches)


def local_execution_groups(state, ordinal, rank):
    rounds = state.get("execution_rounds")
    indices = rounds[ordinal] if rounds is not None else (ordinal,)
    return tuple(next(g for g in state["plan"][i] if rank in g.ranks) for i in indices)
