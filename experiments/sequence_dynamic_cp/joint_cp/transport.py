"""Interval-based all-to-all permutation with an immutable inverse autograd route.

No process groups are created here. All ranks call the same globally compiled
route, including ranks with no remote traffic. Only CPU geometry is cached.
"""

from functools import lru_cache

from .ir import identity_route, route


class Transport:
    def __init__(self, lengths, world, group):
        self.lengths, self.world, self.group = dict(lengths), world, group

    @lru_cache(maxsize=128)
    def edges(self, source, target):
        return route(source, target, self.lengths, self.world)

    def __call__(self, value, source, target):
        if source == target:
            return value
        edges = self.edges(source, target)
        if identity_route(edges):
            return value
        return redistribute(value, edges, self.world, self.group)


def _exchange(value, edges, world, group):
    import torch
    import torch.distributed as dist

    rank = dist.get_rank(group)
    if dist.get_world_size(group) != world:
        raise ValueError("Route world disagrees with supplied process group")
    outgoing = sorted((e for e in edges if e[0] == rank), key=lambda e: (e[1], e[2], e[3]))
    incoming = sorted((e for e in edges if e[1] == rank), key=lambda e: (e[0], e[2], e[3]))
    if value.shape[0] != sum(e[4] for e in outgoing):
        raise ValueError("Tensor does not match compiled route")
    send = torch.cat([value.narrow(0, e[2], e[4]) for e in outgoing], dim=0).contiguous()
    shape = (sum(e[4] for e in incoming), *value.shape[1:])
    received = value.new_empty(shape)
    if any(e[0] != e[1] for e in edges):
        send_sizes = [sum(e[4] for e in outgoing if e[1] == peer) for peer in range(world)]
        recv_sizes = [sum(e[4] for e in incoming if e[0] == peer) for peer in range(world)]
        dist.all_to_all_single(received, send, recv_sizes, send_sizes, group=group)
    else:
        received.copy_(send)
    output = value.new_empty(shape)
    offset = 0
    for _, _, _, destination, count in incoming:
        output.narrow(0, destination, count).copy_(received.narrow(0, offset, count))
        offset += count
    return output


def redistribute(value, edges, world, group):
    import torch

    class Permute(torch.autograd.Function):
        @staticmethod
        def forward(ctx, tensor):
            ctx.inverse = tuple((dst, src, end, start, count) for src, dst, start, end, count in edges)
            return _exchange(tensor, edges, world, group)

        @staticmethod
        def backward(ctx, gradient):
            return _exchange(gradient, ctx.inverse, world, group)

    return Permute.apply(value)
