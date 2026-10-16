"""P0/P1 isolated CP1 Megatron modules. No optimizer or parameter collectives.

Run through profile_campaign.py; every timing iteration consumes a fresh graph.
Reference checks share BF16 outer projections but use independent eager cores.
"""
import argparse
import copy
import csv
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault('NVTE_ALLOW_NONDETERMINISTIC_ALGO', '0')
os.environ.setdefault('NVTE_FLASH_ATTN', '1')
os.environ.setdefault('NVTE_FUSED_ATTN', '0')
sys.path.insert(0, os.environ.get('MEGATRON_ROOT', str(Path(__file__).resolve().parents[3] / 'Megatron-LM')))
import torch
import torch.distributed as dist
import torch.nn.functional as F

# One independent launch visits 11 shapes plus reference shapes. Keep the
# native @jit_fuser path from silently falling back after eight recompiles.
torch._dynamo.config.recompile_limit = 64
torch._dynamo.config.accumulated_recompile_limit = 256

HERE = Path(__file__).resolve().parent
THRESHOLDS = {'native_repeat': {'Y': 0.005, 'dX': 0.005, 'dW': 0.01},
              'eager_core': {'Y': 0.02, 'dX': 0.02, 'dW': 0.03},
              'packed_vs_separate': {'Y': 0.005, 'dX': 0.005, 'dW': 0.01}}


def append_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    exists = path.exists()
    with path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def initialize(seed, directory):
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    torch.cuda.set_device(0)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group('nccl', init_method=f'file://{directory.resolve()}/rendezvous',
                            rank=0, world_size=1, device_id=torch.device('cuda:0'))
    parallel_state.initialize_model_parallel()
    model_parallel_cuda_manual_seed(seed)
    from megatron.core.process_groups_config import ProcessGroupCollection
    return ProcessGroupCollection(tp=parallel_state.get_tensor_model_parallel_group(),
                                  cp=parallel_state.get_context_parallel_group())


def build_module(kind, pg, cp=1, cp_comm_type=None):
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.transformer.spec_utils import ModuleSpec
    from megatron.core.extensions.transformer_engine import (
        TELayerNormColumnParallelLinear, TERowParallelLinear, TEDotProductAttention, TENorm)
    config = TransformerConfig(
        num_layers=1, hidden_size=4096, ffn_hidden_size=16384,
        num_attention_heads=32, num_query_groups=8, kv_channels=128,
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=cp,
        bf16=True, params_dtype=torch.bfloat16, attention_dropout=0, hidden_dropout=0,
        add_bias_linear=False, normalization='RMSNorm', layernorm_epsilon=1e-5,
        activation_func=F.silu, gated_linear_unit=True, bias_activation_fusion=True,
        gradient_accumulation_fusion=False, apply_rope_fusion=False,
        experimental_attention_variant='gated_delta_net' if kind == 'gdn' else None,
        linear_num_key_heads=16, linear_num_value_heads=16,
        linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
        linear_attention_freq=1 if kind == 'gdn' else None,
        linear_cp_mode='chunkwise', gdn_pre_gated_delta_rule_fusion=False)
    if kind == 'gqa':
        from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
        from megatron.core.transformer.enums import AttnMaskType
        module = SelfAttention(config, SelfAttentionSubmodules(
            linear_qkv=TELayerNormColumnParallelLinear,
            core_attention=TEDotProductAttention, linear_proj=TERowParallelLinear),
            layer_number=1, attn_mask_type=AttnMaskType.causal, pg_collection=pg,
            cp_comm_type=cp_comm_type)
    elif kind == 'gdn':
        from megatron.core.ssm.gated_delta_net import GatedDeltaNet, GatedDeltaNetSubmodules
        module = GatedDeltaNet(config, GatedDeltaNetSubmodules(
            in_proj=ModuleSpec(module=TELayerNormColumnParallelLinear),
            out_norm=ModuleSpec(module=TENorm), out_proj=ModuleSpec(module=TERowParallelLinear)),
            layer_number=1, pg_collection=pg)
    else:
        from megatron.core.transformer.mlp import MLP, MLPSubmodules
        module = MLP(config, MLPSubmodules(linear_fc1=TELayerNormColumnParallelLinear,
                                          linear_fc2=TERowParallelLinear), tp_group=pg.tp)
    return module.cuda().train()


def metadata(lengths, group):
    from megatron.core.packed_seq_params import PackedSeqParams
    cu = torch.tensor([0] + list(__import__('itertools').accumulate(lengths)),
                      dtype=torch.int32, device='cuda')
    return PackedSeqParams(qkv_format='thd', cu_seqlens_q=cu, cu_seqlens_kv=cu,
        max_seqlen_q=max(lengths), max_seqlen_kv=max(lengths), total_tokens=sum(lengths),
        cp_group=group, local_cp_size=1, cp_partition_mode='contiguous')


def prepare(kind, lengths, group):
    packed = metadata(lengths, group)
    rope = None
    if kind == 'gqa':
        from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
        rope = RotaryEmbedding(128, 1.0, rotary_base=10000, cp_group=group)(max(lengths))
    return packed, rope


def forward(module, kind, x, prepared):
    packed, rope = prepared
    if kind == 'gqa':
        y, bias = module(x, attention_mask=None, packed_seq_params=packed, rotary_pos_emb=rope)
    elif kind == 'gdn':
        y, bias = module(x, attention_mask=None, packed_seq_params=packed)
    else:
        y, bias = module(x)
    return y if bias is None else y + bias


def eager_attention(query, key, value, attention_mask=None, packed_seq_params=None, **kwargs):
    """Small-shape FP32 causal attention, with independent per-sample masks."""
    cu = packed_seq_params.cu_seqlens_q.tolist()
    outputs = []
    for a, b in zip(cu, cu[1:]):
        q = query[a:b].float().transpose(0, 1)
        k = key[a:b].float().repeat_interleave(q.shape[0] // key.shape[1], dim=1).transpose(0, 1)
        v = value[a:b].float().repeat_interleave(q.shape[0] // value.shape[1], dim=1).transpose(0, 1)
        scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
        mask = torch.ones((b-a, b-a), device=q.device, dtype=torch.bool).triu(1)
        out = scores.masked_fill(mask, -float('inf')).softmax(-1) @ v
        outputs.append(out.transpose(0, 1).reshape(b-a, -1))
    return torch.cat(outputs).to(query.dtype)


def eager_delta(q, k, v, g, beta, cu_seqlens=None, **kwargs):
    """Independent differentiable FP32 recurrent delta-rule reference."""
    cu = cu_seqlens.tolist()
    outputs = []
    for a, b in zip(cu, cu[1:]):
        state = torch.zeros((q.shape[2], q.shape[3], v.shape[3]), device=q.device)
        for t in range(a, b):
            qt, kt, vt = q[0, t].float(), k[0, t].float(), v[0, t].float()
            state = state * g[0, t].float().exp()[:, None, None]
            delta = (vt - torch.einsum('hk,hkv->hv', kt, state)) * beta[0, t].float()[:, None]
            state = state + kt[:, :, None] * delta[:, None, :]
            outputs.append(torch.einsum('hk,hkv->hv', qt / math.sqrt(q.shape[3]), state))
    return torch.stack(outputs)[None].to(v.dtype), None


def eager_mlp(module, x):
    # Same BF16 GEMM inputs/weights; independently expressed norm and SwiGLU.
    p = module.linear_fc1
    z = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5))
    z = (z * p.layer_norm_weight.float()).to(x.dtype)
    z = F.linear(z, p.weight)
    a, b = z.chunk(2, dim=-1)
    z = (F.silu(a.float()) * b.float()).to(x.dtype)
    return F.linear(z, module.linear_fc2.weight)


def snapshot(module, kind, x, dy, prepared, reference=False):
    module.zero_grad(set_to_none=True)
    x = x.detach().clone().requires_grad_()
    y = eager_mlp(module, x) if reference and kind == 'mlp' else forward(module, kind, x, prepared)
    y.backward(dy)
    grads = {}
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            raise RuntimeError(f'missing trainable gradient: {name}')
        grads[name] = parameter.grad.detach().clone()
    return y.detach().clone(), x.grad.detach().clone(), grads


def compare(left, right, mode, kind, seed):
    records = []
    pairs = [('Y', '', left[0], right[0]), ('dX', '', left[1], right[1])]
    pairs += [('dW', n, left[2][n], right[2][n]) for n in left[2]]
    for tensor, name, actual, expected in pairs:
        actual, expected = actual.float(), expected.float()
        delta = actual - expected
        relative = (delta.norm() / expected.norm().clamp_min(1e-12)).item()
        finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
        threshold = THRESHOLDS[mode][tensor]
        records.append(dict(module=kind, seed=seed, check=mode, tensor=tensor, parameter=name,
                            finite=finite, relative_l2=relative, max_abs=delta.abs().max().item(),
                            threshold=threshold, passed=finite and relative <= threshold))
    return records


def validate(module, kind, group, directory):
    records = []
    # Nonuniform upstream gradients and unequal sequence lengths in both seeds.
    lengths = [128, 64]
    prepared = prepare(kind, lengths, group)
    for seed in (1701, 2702):
        torch.manual_seed(seed)
        x = torch.randn((sum(lengths), 1, 4096), device='cuda', dtype=torch.bfloat16) * .2
        dy = torch.randn_like(x) * .1
        native = snapshot(module, kind, x, dy, prepared)
        repeat = snapshot(module, kind, x, dy, prepared)
        records.extend(compare(native, repeat, 'native_repeat', kind, seed))
        del repeat
        if kind == 'gqa':
            original = module.core_attention.forward
            module.core_attention.forward = eager_attention
        elif kind == 'gdn':
            original = module.gated_delta_rule
            module.gated_delta_rule = eager_delta
        try:
            reference = snapshot(module, kind, x, dy, prepared, reference=True)
        finally:
            if kind == 'gqa':
                module.core_attention.forward = original
            elif kind == 'gdn':
                module.gated_delta_rule = original
        records.extend(compare(native, reference, 'eager_core', kind, seed))
        del reference
        # Per-sequence execution tests causal isolation, conv/recurrence reset,
        # RoPE position reset, and the sum of local parameter contributions.
        module.zero_grad(set_to_none=True)
        sx = x.detach().clone().requires_grad_()
        outputs = []
        offset = 0
        for length in lengths:
            y = forward(module, kind, sx[offset:offset+length], prepare(kind, [length], group))
            y.backward(dy[offset:offset+length])
            outputs.append(y.detach())
            offset += length
        separate = (torch.cat(outputs), sx.grad, {n:p.grad for n,p in module.named_parameters()})
        records.extend(compare(native, separate, 'packed_vs_separate', kind, seed))
        del native, separate, outputs
    append_csv(directory / 'checks.csv', records)
    module.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    if not all(r['passed'] for r in records):
        raise RuntimeError('P0 numerical gate failed; see checks.csv. No performance rows emitted.')


def measure(module, kind, case, group, args, directory):
    lengths = case['lengths']
    torch.manual_seed(args.seed + 9)
    x = (torch.randn((sum(lengths), 1, 4096), device='cuda', dtype=torch.bfloat16) * .2).requires_grad_()
    dy = torch.randn_like(x) * .1
    prepared = prepare(kind, lengths, group)
    rows = []
    init_end = time.perf_counter()
    for iteration in range(-args.warmup, args.iterations):
        module.zero_grad(set_to_none=True)
        x.grad = None
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        begin, middle, end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        host_start = time.perf_counter()
        begin.record()
        label = f"case={case['case']}/task=0/block=0/module={kind}/group=cp1/layout=thd"
        with torch.cuda.nvtx.range(label + '/phase=forward'):
            y = forward(module, kind, x, prepared)
        middle.record()
        with torch.cuda.nvtx.range(label + '/phase=backward'):
            y.backward(dy)
        end.record()
        end.synchronize()
        host_ms = (time.perf_counter() - host_start) * 1000
        if iteration >= 0:
            for phase, milliseconds in [('forward', begin.elapsed_time(middle)),
                ('backward', middle.elapsed_time(end)), ('fwd_bwd', begin.elapsed_time(end))]:
                rows.append(dict(run=args.run, launch=args.launch, iteration=iteration, rank=0,
                    module=kind, implementation=getattr(module, 'benchmark_implementation', 'megatron_te_fla_cp1'), phase=phase,
                    case=case['case'], split=case['split'], cp=1, topology='single_h800',
                    lengths=json.dumps(lengths), local_tokens=sum(lengths),
                    s1_ki=sum(lengths)/1024, s2_ki2=sum(n*n for n in lengths)/1024**2,
                    causal_pairs=sum(n*(n+1)//2 for n in lengths), milliseconds=milliseconds,
                    host_elapsed_ms=host_ms, allocated_bytes=torch.cuda.max_memory_allocated(),
                    reserved_bytes=torch.cuda.max_memory_reserved(), status='success', provenance='measured'))
        del y
    append_csv(directory / 'measurements.csv', rows)
    module.zero_grad(set_to_none=True)
    x.grad = None
    return time.perf_counter() - init_end


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module', choices=['gqa','gdn','mlp'], required=True)
    parser.add_argument('--cases', default='u02,u04,u08,u16,u32,u64,p04,p16,h24,h48,h_mix')
    parser.add_argument('--inputs', type=Path, default=HERE/'configs'/'inputs.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run', default='standalone')
    parser.add_argument('--launch', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    startup = time.perf_counter()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; execute outside device-restricted sandbox')
        pg = initialize(args.seed, args.output)
        module = build_module(args.module, pg)
        (args.output/'module.json').write_text(json.dumps(dict(
            module=args.module, config=repr(module.config), parameter_dtypes={n:str(p.dtype) for n,p in module.named_parameters()},
            gradient_accumulation='parameter.grad; BF16 for BF16 parameters; no fused main_grad',
            dynamo_recompile_limit=64, dynamo_accumulated_recompile_limit=256,
            initialization_seconds=time.perf_counter()-startup, trainable_parameters=sum(p.numel() for p in module.parameters()),
            outer_norm_included=True, reference_scope='shared BF16 outer projections, independent eager core; packed-vs-separate full module'), indent=2))
        validate(module, args.module, pg.cp, args.output)
        print(f'{args.module}: P0 passed', flush=True)
        if not args.validate_only:
            cases = {c['case']:c for c in json.loads(args.inputs.read_text())}
            for name in args.cases.split(','):
                try:
                    elapsed = measure(module, args.module, cases[name], pg.cp, args, args.output)
                    print(f'{args.module}/{name}: {elapsed:.2f}s', flush=True)
                except torch.OutOfMemoryError as exc:
                    append_csv(args.output/'failures.csv', [dict(module=args.module, case=name,
                        status='OOM', error=str(exc), allocated_bytes=torch.cuda.memory_allocated(),
                        reserved_bytes=torch.cuda.memory_reserved())])
                    # Exit: a fresh process is required after OOM (no contaminated tail).
                    raise
    except Exception as exc:
        (args.output/'error.json').write_text(json.dumps(dict(type=type(exc).__name__,
            error=str(exc), traceback=traceback.format_exc()), indent=2))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
