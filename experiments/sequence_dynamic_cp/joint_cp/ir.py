"""Immutable ownership IR. A token is identified by (sample, position).

The executor domain is aligned dyadic rank groups, no padding, TP=PP=1.
GQA uses per-sequence zigzag; GDN uses contiguous slices of the whole pack.
An MLP can use either placement, independent of the attention group.
"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True, order=True)
class Group:
    ranks: tuple[int, ...]
    samples: tuple[int, ...]


@dataclass(frozen=True)
class Layout:
    groups: tuple[Group, ...]
    mode: str

    def with_mode(self, mode):
        return Layout(self.groups, mode)

    def sample_cp(self):
        return tuple(sorted((sample, len(g.ranks)) for g in self.groups for sample in g.samples))

    def tokens(self, lengths, world):
        result = [0] * world
        for group in self.groups:
            n = sum(lengths[s] for s in group.samples) // len(group.ranks)
            for rank in group.ranks:
                result[rank] = n
        return tuple(result)

    def rank_ids(self, world):
        result = [[] for _ in range(world)]
        for group in self.groups:
            for rank in group.ranks:
                result[rank] = list(group.samples)
        return result


@dataclass(frozen=True)
class Microbatch:
    ingress: Layout
    # One pair (attention, MLP) for every transformer layer.
    layers: tuple[tuple[Layout, Layout], ...]


@dataclass(frozen=True)
class Plan:
    world: int
    pattern: str
    microbatches: tuple[Microbatch, ...]

    def to_dict(self):
        return dict(schema="joint_cp_v1", **asdict(self))

    @classmethod
    def from_dict(cls, data):
        if data.get("schema") != "joint_cp_v1":
            raise ValueError("Unknown joint plan schema")
        def layout(row):
            return Layout(tuple(Group(tuple(g["ranks"]), tuple(g["samples"]))
                                for g in row["groups"]), row["mode"])
        return cls(data["world"], data["pattern"], tuple(
            Microbatch(layout(b["ingress"]), tuple((layout(a), layout(m)) for a, m in b["layers"]))
            for b in data["microbatches"]))


def validate_layout(layout, lengths, world):
    if layout.mode not in ("zigzag", "contiguous"):
        raise ValueError("Unknown ownership mode")
    ranks, samples = [], []
    for group in layout.groups:
        cp = len(group.ranks)
        if (not cp or cp & (cp - 1) or group.ranks[0] % cp
                or group.ranks != tuple(range(group.ranks[0], group.ranks[0] + cp))):
            raise ValueError("CP groups must be aligned dyadic rank intervals")
        if not group.samples:
            raise ValueError("Every executing group must have nonempty work")
        for sample in group.samples:
            if sample not in lengths or type(lengths[sample]) is not int or lengths[sample] <= 0:
                raise ValueError("Unknown sample or invalid length")
            if cp > 1 and lengths[sample] % (2 * cp):
                raise ValueError("Unpadded backend requires sequence alignment to 2*CP")
        ranks.extend(group.ranks)
        samples.extend(group.samples)
    if sorted(ranks) != list(range(world)):
        raise ValueError("Each rank must execute exactly one group per layer")
    if len(samples) != len(set(samples)):
        raise ValueError("Layout duplicates a sample")
    return frozenset(samples)


def validate_plan(plan, lengths):
    if not plan.world or plan.world & (plan.world - 1):
        raise ValueError("World must be a positive power of two")
    if not plan.pattern or set(plan.pattern) - set("G*"):
        raise ValueError("Pattern must contain GDN G and/or GQA *")
    seen = set()
    for batch in plan.microbatches:
        ids = validate_layout(batch.ingress, lengths, plan.world)
        if batch.ingress.mode != "zigzag" or len(batch.layers) != len(plan.pattern):
            raise ValueError("Invalid ingress or layer count")
        if seen & ids:
            raise ValueError("Microbatches duplicate samples")
        seen.update(ids)
        for symbol, (attention, mlp) in zip(plan.pattern, batch.layers):
            if attention.mode != ("contiguous" if symbol == "G" else "zigzag"):
                raise ValueError("Attention ownership mode disagrees with kernel")
            for layout in (attention, mlp):
                if validate_layout(layout, lengths, plan.world) != ids:
                    raise ValueError("Layer ownership changes the microbatch sample set")
    if seen != set(lengths):
        raise ValueError("Plan does not cover the complete input exactly once")


def spans(layout, lengths, world):
    """Per-rank (sample, logical start, local offset, length), without token arrays."""
    result = [[] for _ in range(world)]
    for group in layout.groups:
        cp = len(group.ranks)
        total = sum(lengths[s] for s in group.samples)
        width = total // cp
        for index, rank in enumerate(group.ranks):
            offset, pack_offset = 0, 0
            for sample in group.samples:
                n = lengths[sample]
                if layout.mode == "zigzag":
                    intervals = [(0, n)] if cp == 1 else [
                        (index * (n // (2 * cp)), n // (2 * cp)),
                        ((2 * cp - 1 - index) * (n // (2 * cp)), n // (2 * cp))]
                else:
                    begin = max(pack_offset, index * width)
                    end = min(pack_offset + n, (index + 1) * width)
                    intervals = [(begin - pack_offset, end - begin)] if end > begin else []
                for begin, count in intervals:
                    result[rank].append((sample, begin, offset, count))
                    offset += count
                pack_offset += n
    return tuple(tuple(row) for row in result)


def route(source, target, lengths, world):
    """Edges (src rank, dst rank, src offset, dst offset, count); exact permutation."""
    if validate_layout(source, lengths, world) != validate_layout(target, lengths, world):
        raise ValueError("Routing requires identical sample sets")
    src, dst = spans(source, lengths, world), spans(target, lengths, world)
    by_sample = {}
    for rank, row in enumerate(dst):
        for sample, start, offset, count in row:
            by_sample.setdefault(sample, []).append((rank, start, offset, count))
    edges = []
    for rank, row in enumerate(src):
        for sample, start, offset, count in row:
            for other, begin, dest, size in by_sample[sample]:
                lo, hi = max(start, begin), min(start + count, begin + size)
                if hi > lo:
                    edges.append((rank, other, offset + lo - start, dest + lo - begin, hi - lo))
    return tuple(sorted(edges))


def identity_route(edges):
    return all(src == dst and start == end for src, dst, start, end, _ in edges)
