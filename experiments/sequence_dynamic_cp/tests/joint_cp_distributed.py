"""torchrun --standalone --nproc-per-node=2 .../joint_cp_distributed.py

CPU/Gloo route + MLP placement gradients against a centralized dense reference.
Does not exercise TE/FLA kernels, BF16 numerics, NCCL, or HBM admission.
"""

from pathlib import Path
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from joint_cp import Group, Layout
from joint_cp.adapter import mlp_placement
from joint_cp.ir import spans
from joint_cp.transport import Transport


def indices(layout, rank, lengths):
    starts, position = {}, 0
    for sample, size in lengths.items():
        starts[sample], position = position, position + size
    return torch.tensor([starts[s] + begin + k for s, begin, _, count in spans(layout, lengths, 2)[rank]
                         for k in range(count)], dtype=torch.long)


class Dense(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(9, dtype=torch.float64).reshape(3, 3) / 20)

    def forward(self, hidden_states):
        return torch.tanh(hidden_states @ self.weight), None


def main():
    dist.init_process_group("gloo")
    rank, group = dist.get_rank(), dist.group.WORLD
    lengths = {0: 16, 1: 8, 2: 8}
    source = Layout((Group((0,), (0,)), Group((1,), (1, 2))), "zigzag")
    packed = Layout((Group((0, 1), (0, 1, 2)),), "contiguous")
    uneven = Layout((Group((0,), (1,)), Group((1,), (0, 2))), "zigzag")
    zigzag = packed.with_mode("zigzag")
    transfer = Transport(lengths, 2, group)
    full = torch.arange(96, dtype=torch.float64).reshape(32, 1, 3) / 100
    original = full[indices(source, rank, lengths)].clone().requires_grad_()
    assert transfer(original, source, source.with_mode("contiguous")) is original
    value = original
    for layout in (packed, uneven, zigzag, source):
        value = transfer(value, source if layout == packed else previous, layout)
        torch.testing.assert_close(value, full[indices(layout, rank, lengths)], rtol=0, atol=0)
        previous = layout
    (value.square().sum()).backward()
    torch.testing.assert_close(original.grad, 2 * original.detach(), rtol=0, atol=0)

    # Three layout changes plus MLP round trip and residual; parameter gradients
    # are compared after SUM reduction, matching unique logical-token ownership.
    local = original.detach().clone().requires_grad_()
    reference = full.clone().requires_grad_()
    layer, reference_layer = Dense(), Dense()
    hidden = transfer(local, source, packed)
    with mlp_placement(layer, transfer, packed, uneven):
        output, _ = layer(hidden)
    output = transfer(hidden + output, packed, source)
    weights = (torch.arange(32, dtype=torch.float64) + 1).reshape(32, 1, 1)
    (output * weights[indices(source, rank, lengths)]).sum().backward()
    expected = reference + reference_layer(reference)[0]
    (expected * weights).sum().backward()
    dist.all_reduce(layer.weight.grad, group=group)
    torch.testing.assert_close(output, expected[indices(source, rank, lengths)], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(local.grad, reference.grad[indices(source, rank, lengths)], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(layer.weight.grad, reference_layer.weight.grad, rtol=1e-12, atol=1e-12)
    if rank == 0:
        print("PASS: exact token permutation, inverse gradient, MLP residual, input and parameter gradients (CPU/Gloo)")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
