"""CPU contracts for the mixed executor; actual TE/FLA checks run on GPUs."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from native import MEGATRON_ROOT
sys.path.insert(0, str(MEGATRON_ROOT))

from mixed_plan import coalesce_rounds
from planner import Group

from megatron.core.mixed_context_parallel import (
    merge_packed_batches,
    mixed_attention_forward,
    mixed_gdn_forward,
)
from megatron.core.packed_seq_params import PackedSeqParams


class CP:
    def __init__(self, size):
        self.n = size

    def size(self):
        return self.n


def part(lengths, cp, start):
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    n = sum(lengths) // cp
    p = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=cu,
        cu_seqlens_kv_padded=cu,
        max_seqlen_q=max(lengths),
        max_seqlen_kv=max(lengths),
        local_cp_size=cp,
        cp_group=CP(cp),
        pad_between_seqs=False,
    )
    x = torch.arange(start, start + n)[None, :]
    return x, x + 1, torch.ones_like(x), None, x + 10, p, torch.zeros_like(x, dtype=torch.bool)


def config():
    return SimpleNamespace(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        sequence_parallel=False,
        recompute_granularity=None,
        attention_output_gate=False,
        head_wise_attn_gate=False,
        no_rope_freq=None,
        linear_cp_mode="chunkwise",
    )


def test_r8_coalescing_preserves_ownership_and_capacity():
    lengths = {0: 65536, **{i: 2048 for i in range(1, 17)}}
    plan = (tuple(Group((r,), (r + 1, r + 9)) for r in range(8)), (Group(tuple(range(8)), (0,)),))
    assert coalesce_rounds(plan, lengths, 8, 32768) == ((0, 1),)
    assert coalesce_rounds(plan, lengths, 8, 8192) == ((0,), (1,))
    assert coalesce_rounds(plan, lengths, 8, 32768, 2) == ((0,), (1,))


def test_merged_thd_keeps_child_global_boundaries_and_local_fields():
    parts = [part([4, 4], 1, 0), part([64], 8, 100)]
    batch = merge_packed_batches(parts, CP(1))
    assert batch[0].tolist() == [list(range(8)) + list(range(100, 108))]
    assert batch[5].cu_seqlens_q.tolist() == [0, 4, 8, 16]
    assert batch[5].max_seqlen_q == 64
    assert [(s.start, s.end, s.params.local_cp_size) for s in batch[5].mixed_cp_segments] == [(0, 8, 1), (8, 16, 8)]
    assert batch[5].mixed_cp_segments[1].params.cu_seqlens_q.tolist() == [0, 64]
    for i in (0, 1, 2, 4, 6):
        assert torch.equal(batch[i], torch.cat([p[i] for p in parts], dim=-1))
    assert merge_packed_batches(parts[:1], CP(1)) is parts[0]
    parts[1][5].local_cp_size = 4
    with pytest.raises(ValueError, match="metadata"):
        merge_packed_batches(parts, CP(1))


@pytest.mark.parametrize("padded", [True, False])
def test_attention_projects_once_and_accumulates_same_gradients(monkeypatch, padded):
    import megatron.core.models.common.embeddings.rope_utils as rope

    torch.manual_seed(4)
    p = merge_packed_batches([part([4, 4], 1, 0), part([64], 8, 100)], CP(1))[5]
    if not padded:
        for s in p.mixed_cp_segments:
            s.params.cu_seqlens_q_padded = s.params.cu_seqlens_kv_padded = None
    rope_calls = []

    def apply_rope(t, emb, *, cu_seqlens, cp_group, **kwargs):
        rope_calls.append((cp_group.size(), cu_seqlens.tolist()))
        return t

    monkeypatch.setattr(rope, "apply_rotary_pos_emb", apply_rope)

    class Model:
        config = config()
        attention_type = "self"
        checkpoint_core_attention = False
        offload_qkv_linear = offload_core_attention = offload_attn_proj = False
        rotary_pos_emb = None
        _yarn_concentration_factor = 1.0
        attn_mask_type = "causal"

        def __init__(self):
            self.w = torch.randn(3, 9, requires_grad=True)
            self.o = torch.randn(3, 2, requires_grad=True)
            self.calls = []

        def get_query_key_value_tensors(self, x, **kwargs):
            self.calls.append("qkv")
            return tuple(t.unsqueeze(2) for t in (x @ self.w).chunk(3, dim=-1))

        def _run_core_attention(self, q, k, v, *args, packed_seq_params, **kwargs):
            self.calls.append(packed_seq_params.local_cp_size)
            return (q * k).sigmoid() * v

        def linear_proj(self, x):
            self.calls.append("out")
            return x @ self.o, None

    m = Model()
    x = torch.randn(16, 1, 3, requires_grad=True)
    y, _ = mixed_attention_forward(m, x, None, torch.zeros(64, 1, 1, 3), p)
    got = torch.autograd.grad(y.square().sum(), (x, m.w, m.o))
    ref = []
    for s in p.mixed_cp_segments:
        q, k, v = (x[s.start : s.end] @ m.w).chunk(3, dim=-1)
        ref.append(((q * k).sigmoid() * v) @ m.o)
    expected = torch.autograd.grad(torch.cat(ref).square().sum(), (x, m.w, m.o))
    assert m.calls == ["qkv", 1, 8, "out"]
    assert rope_calls == [(1, [0, 4, 8])] * 2 + [(8, [0, 64])] * 2
    for a, b in zip(got, expected):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("convert_layout", [True, False])
def test_gdn_projects_and_normalizes_once_with_layout_roundtrip(monkeypatch, convert_layout):
    import megatron.core.context_parallel_layout as layout

    class Reverse:
        def convert(self, x, **kwargs):
            return x.flip(0)

    def convert(*, hidden_states, cp_group, **kwargs):
        return (hidden_states.flip(0), Reverse()) if convert_layout and cp_group.size() > 1 else (hidden_states, None)

    monkeypatch.setattr(layout, "convert_module_input_tensors_cp_partition_mode", convert)
    p = merge_packed_batches([part([4, 4], 1, 0), part([64], 8, 100)], CP(1))[5]

    class Model:
        config = config()
        recompute_gdn = recompute_norm_out = False
        tp_group = None
        in_proj_split_sections = [2, 2, 2, 2, 1, 1]
        value_head_dim = 2

        def __init__(self):
            self.w = torch.randn(3, 10, requires_grad=True)
            self.o = torch.randn(2, 3, requires_grad=True)
            self.calls = []

        def in_proj(self, x):
            self.calls.append("in")
            return x @ self.w, None

        def _forward_single_batch(self, x, **kwargs):
            assert kwargs["projected_core_only"]
            self.calls.append("core")
            return x[..., :2].cumsum(0), None

        def _apply_gated_norm(self, x, g):
            self.calls.append("norm")
            return torch.tanh(x) * torch.sigmoid(g)

        def out_proj(self, x):
            self.calls.append("out")
            return x @ self.o, None

    torch.manual_seed(5)
    m = Model()
    x = torch.randn(16, 1, 3, requires_grad=True)
    y, _ = mixed_gdn_forward(m, x, p, SimpleNamespace())
    actual = torch.autograd.grad(y.square().sum(), (x, m.w, m.o))
    expected = []
    for s in p.mixed_cp_segments:
        inp = x[s.start : s.end]
        if convert_layout and s.params.local_cp_size > 1:
            inp = inp.flip(0)
        proj = inp @ m.w
        out = (proj[..., :2].cumsum(0).tanh() * proj[..., 6:8].sigmoid()) @ m.o
        if convert_layout and s.params.local_cp_size > 1:
            out = out.flip(0)
        expected.append(out)
    expected = torch.autograd.grad(torch.cat(expected).square().sum(), (x, m.w, m.o))
    assert m.calls == ["in", "core", "core", "norm", "out"]
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)


def test_consecutive_gdn_layers_share_layout_exchanges(monkeypatch):
    import megatron.core.context_parallel_layout as layout
    from megatron.core.mixed_context_parallel import mixed_parent_contiguous_forward

    exchanges = []

    class Converter:
        def __init__(self, **kwargs):
            self.kw = kwargs

        def convert(self, x, **kwargs):
            if self.kw["cp_group"].size() == 1:
                return x
            exchanges.append(self.kw["target_partition_mode"])
            self.kw["packed_seq_params"].cp_partition_mode = self.kw["target_partition_mode"]
            return x.flip(0)

    monkeypatch.setattr(layout, "CpPartitionModeConverter", Converter)
    p = merge_packed_batches([part([4, 4], 1, 0), part([64], 8, 100)], CP(1))[5]
    calls = []

    def layer(*, hidden_states, packed_seq_params, **kwargs):
        calls.append(tuple(s.params.cp_partition_mode for s in packed_seq_params.mixed_cp_segments))
        return hidden_states * 2, None

    block = SimpleNamespace(
        config=config(),
        pg_collection=SimpleNamespace(tp=None),
        _packed_sequence_linear_pattern=[1, 1, 1, 0],
        layers=[layer] * 4,
    )
    x = torch.randn(16, 1, 3, requires_grad=True)
    y = mixed_parent_contiguous_forward(block, x, p, None, None)
    torch.testing.assert_close(y, 16 * x)
    assert exchanges == ["contiguous", "zigzag"]
    assert calls == [("contiguous", "contiguous")] * 3 + [("zigzag", "zigzag")]
    assert all(s.params.cp_partition_mode == "zigzag" for s in p.mixed_cp_segments)
    torch.testing.assert_close(torch.autograd.grad(y.sum(), x)[0], torch.full_like(x, 16))
