"""Benchmark fixed and heterogeneous CP for DSv4 CSA/HCA layers.

This entrypoint targets the contiguous THD context-parallel path on the current
Megatron-LM ``dev`` branch.  It intentionally models a restricted but useful
case: every sample has the same sequence length and CP groups are contiguous in
global-rank order.  With a fixed global batch, each rank then owns the same
global contiguous token interval for every valid CP degree.  A CP transition is
therefore an autograd-visible identity on the activation tensor; only
``PackedSeqParams`` and the active CP group change at the next layer.

The benchmark excludes optimizer work, indexer auxiliary loss, and data-parallel
gradient synchronization.  Its purpose is the first-pass CSA/HCA CP sweet-spot
calibration, not a full training-throughput claim.
"""

import argparse
import os
import random
import statistics
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.distributed as dist


_PROFILES = {
    "flash": {
        "hidden_size": 4096,
        "num_attention_heads": 64,
        "q_lora_rank": 1024,
        "v_head_dim": 512,
        "qk_pos_emb_head_dim": 64,
        "o_groups": 8,
        "o_lora_rank": 1024,
        "csa_compress_rotary_base": 40000,
        "dsa_indexer_topk": 512,
        "dsa_indexer_n_heads": 64,
        # cuDNN SM90 currently needs 128 for this DSA indexer path.
        "dsa_indexer_head_dim": 128,
    },
    "pro": {
        "hidden_size": 7168,
        "num_attention_heads": 128,
        "q_lora_rank": 1536,
        "v_head_dim": 512,
        "qk_pos_emb_head_dim": 64,
        "o_groups": 16,
        "o_lora_rank": 1024,
        "csa_compress_rotary_base": 160000,
        "dsa_indexer_topk": 1024,
        "dsa_indexer_n_heads": 64,
        "dsa_indexer_head_dim": 128,
    },
    # Constructor/single-GPU smoke profile; do not use it for performance claims.
    "tiny": {
        "hidden_size": 256,
        "num_attention_heads": 16,
        "q_lora_rank": 64,
        "v_head_dim": 64,
        "qk_pos_emb_head_dim": 32,
        "o_groups": 8,
        "o_lora_rank": 64,
        "csa_compress_rotary_base": 40000,
        "dsa_indexer_topk": 8,
        "dsa_indexer_n_heads": 8,
        "dsa_indexer_head_dim": 64,
    },
}


@dataclass(frozen=True)
class LayerLayout:
    """One operator and the CP degree used to execute it."""

    kind: str
    cp_size: int


class EqualLengthTHDContiguousTransition(torch.autograd.Function):
    """Identity transition for the equal-length contiguous-THD special case.

    For global batch ``G``, sequence length ``S``, world size ``W`` and CP size
    ``C``, a rank stores ``(G / (W/C)) * S / C == G*S/W`` rows.  For contiguous
    CP groups, global rank ``r`` owns rows ``[r*G*S/W, (r+1)*G*S/W)`` regardless
    of ``C``.  Consequently no P2P activation movement is required.
    """

    @staticmethod
    def forward(ctx, hidden, src_cp, dst_cp, global_batch, seq_length, world_size):
        del ctx
        expected_rows = equal_length_local_rows(global_batch, seq_length, world_size)
        if hidden.ndim != 3 or hidden.shape[1] != 1:
            raise ValueError(
                "THD hidden states must have shape [local_tokens, 1, hidden_size]."
            )
        if hidden.shape[0] != expected_rows:
            raise ValueError(
                f"Transition CP{src_cp}->CP{dst_cp} expected {expected_rows} local "
                f"THD rows, got {hidden.shape[0]}."
            )
        validate_layout(src_cp, global_batch, seq_length, world_size)
        validate_layout(dst_cp, global_batch, seq_length, world_size)
        global_rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        source_interval = rank_token_interval(
            global_rank, src_cp, global_batch, seq_length, world_size
        )
        destination_interval = rank_token_interval(
            global_rank, dst_cp, global_batch, seq_length, world_size
        )
        if source_interval != destination_interval:
            raise ValueError(
                f"Transition CP{src_cp}->CP{dst_cp} changes rank {global_rank} ownership "
                f"from {source_interval} to {destination_interval}; a real redistribution "
                "kernel is required."
            )
        return hidden

    @staticmethod
    def backward(ctx, grad_output):
        del ctx
        return grad_output, None, None, None, None, None


def transition_equal_length_thd(
    hidden: torch.Tensor,
    src_cp: int,
    dst_cp: int,
    global_batch: int,
    seq_length: int,
    world_size: int,
) -> torch.Tensor:
    """Apply the metadata-only THD transition after validating its assumptions."""
    return EqualLengthTHDContiguousTransition.apply(
        hidden, src_cp, dst_cp, global_batch, seq_length, world_size
    )


def equal_length_local_rows(global_batch: int, seq_length: int, world_size: int) -> int:
    """Return invariant local THD rows for a fixed global batch."""
    total_rows = global_batch * seq_length
    if total_rows % world_size:
        raise ValueError(
            f"global_batch_size * seq_length ({total_rows}) must divide evenly across "
            f"world_size={world_size}."
        )
    return total_rows // world_size


def validate_layout(cp_size: int, global_batch: int, seq_length: int, world_size: int) -> None:
    """Validate one DP x CP layout under the equal-length THD assumptions."""
    if cp_size < 1 or world_size % cp_size:
        raise ValueError(f"CP size {cp_size} must divide world_size={world_size}.")
    dp_size = world_size // cp_size
    if global_batch < 1 or global_batch % dp_size:
        raise ValueError(
            f"global_batch_size={global_batch} must divide DP={dp_size} for CP={cp_size}."
        )
    local_batch = global_batch // dp_size
    if local_batch * seq_length % cp_size:
        raise ValueError(
            f"local_batch({local_batch}) * seq_length({seq_length}) must divide CP={cp_size}."
        )
    equal_length_local_rows(global_batch, seq_length, world_size)


def rank_token_interval(
    global_rank: int,
    cp_size: int,
    global_batch: int,
    seq_length: int,
    world_size: int,
) -> Tuple[int, int]:
    """Return one rank's global flattened-token interval for contiguous THD CP."""
    validate_layout(cp_size, global_batch, seq_length, world_size)
    if not 0 <= global_rank < world_size:
        raise ValueError(f"global_rank={global_rank} is outside world_size={world_size}.")
    dp_rank = global_rank // cp_size
    cp_rank = global_rank % cp_size
    local_batch = global_batch // (world_size // cp_size)
    local_rows = equal_length_local_rows(global_batch, seq_length, world_size)
    start = dp_rank * local_batch * seq_length + cp_rank * local_rows
    return start, start + local_rows


def make_layer_kinds(num_csa_layers: int, num_hca_layers: int, seed: int) -> List[str]:
    """Create one reproducibly shuffled CSA/HCA layer multiset."""
    if num_csa_layers < 0 or num_hca_layers < 0:
        raise ValueError("Layer counts must be non-negative.")
    kinds = ["csa"] * num_csa_layers + ["hca"] * num_hca_layers
    if not kinds:
        raise ValueError("At least one CSA or HCA layer is required.")
    random.Random(seed).shuffle(kinds)
    return kinds


def make_plan(kinds: Sequence[str], args, mode: str) -> List[LayerLayout]:
    """Build a fixed or operator-specific CP plan for the same layer schedule."""
    if mode == "fixed":
        requested = [
            args.csa_cp_size if "csa" in kinds else 1,
            args.hca_cp_size if "hca" in kinds else 1,
        ]
        fixed_cp = args.fixed_cp_size or max(requested)
        return [LayerLayout(kind, fixed_cp) for kind in kinds]
    return [
        LayerLayout(kind, args.csa_cp_size if kind == "csa" else args.hca_cp_size)
        for kind in kinds
    ]


def validate_plan(plan: Sequence[LayerLayout], args, world_size: int) -> None:
    """Reject plans that violate CP, batch, or compression preconditions."""
    for layer in plan:
        validate_layout(layer.cp_size, args.global_batch_size, args.seq_length, world_size)
        ratio = args.csa_compress_ratio if layer.kind == "csa" else args.hca_compress_ratio
        if args.seq_length < ratio or args.seq_length % ratio:
            raise ValueError(
                f"seq_length={args.seq_length} must be at least and divisible by the "
                f"{layer.kind.upper()} compression ratio {ratio}."
            )


def make_cp_group_cache(cp_sizes: Iterable[int]):
    """Create contiguous CP groups in identical order on every rank."""
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    cache = {}
    for cp_size in sorted(set(cp_sizes)):
        for start in range(0, world_size, cp_size):
            ranks = list(range(start, start + cp_size))
            group = dist.new_group(ranks)
            if rank in ranks:
                cache[cp_size] = (group, ranks)
    return cache


def make_local_tp_group():
    """Create the TP1 group required by DSv4HybridSelfAttention."""
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_group = None
    for global_rank in range(world_size):
        group = dist.new_group([global_rank])
        if rank == global_rank:
            local_group = group
    return local_group


def make_packed_seq_params(args, cp_size: int, cp_group, device):
    """Build full logical-microbatch THD metadata shared by one CP group."""
    from megatron.core.packed_seq_params import PackedSeqParams

    world_size = dist.get_world_size()
    dp_size = world_size // cp_size
    local_batch = args.global_batch_size // dp_size
    cu = torch.arange(
        0,
        (local_batch + 1) * args.seq_length,
        args.seq_length,
        dtype=torch.int32,
        device=device,
    )
    packed = PackedSeqParams(
        cu_seqlens_q=cu,
        cu_seqlens_q_padded=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_kv_padded=cu,
        max_seqlen_q=args.seq_length,
        max_seqlen_kv=args.seq_length,
        qkv_format="thd",
        cp_partition_mode="contiguous",
        tokens_per_sample=args.seq_length,
    )
    packed.local_cp_size = cp_size
    packed.cp_group = cp_group
    return packed


def make_config(args, kind: str, cp_size: int):
    """Construct the current Megatron-LM DSv4 hybrid attention config."""
    from megatron.core.transformer.transformer_config import MLATransformerConfig

    shape = _PROFILES[args.profile]
    ratio = args.csa_compress_ratio if kind == "csa" else args.hca_compress_ratio
    return MLATransformerConfig(
        num_layers=1,
        hidden_size=shape["hidden_size"],
        num_attention_heads=shape["num_attention_heads"],
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        add_bias_linear=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        q_lora_rank=shape["q_lora_rank"],
        kv_lora_rank=shape["v_head_dim"] - shape["qk_pos_emb_head_dim"],
        qk_head_dim=shape["v_head_dim"] - shape["qk_pos_emb_head_dim"],
        qk_pos_emb_head_dim=shape["qk_pos_emb_head_dim"],
        v_head_dim=shape["v_head_dim"],
        o_groups=shape["o_groups"],
        o_lora_rank=shape["o_lora_rank"],
        rope_type="yarn" if args.rope_fusion else "rope",
        rotary_base=10000,
        rotary_percent=1.0,
        multi_latent_attention=True,
        experimental_attention_variant="dsv4_hybrid",
        csa_compress_ratios=[ratio],
        csa_window_size=args.csa_window_size,
        csa_dense_mode=False,
        csa_compress_rotary_base=shape["csa_compress_rotary_base"],
        dsa_indexer_n_heads=shape["dsa_indexer_n_heads"],
        dsa_indexer_head_dim=(
            args.dsa_indexer_head_dim
            if args.dsa_indexer_head_dim is not None
            else shape["dsa_indexer_head_dim"]
        ),
        dsa_indexer_topk=shape["dsa_indexer_topk"],
        dsa_indexer_loss_coeff=0.0,
        dsa_indexer_use_sparse_loss=False,
        dsa_kernel_backend=args.dsa_kernel_backend,
        apply_rope_fusion=args.rope_fusion,
        context_parallel_size=cp_size,
        cp_partition_mode="contiguous",
        sequence_packing_scheduler="dp_balanced" if cp_size > 1 else None,
        layernorm_epsilon=1e-6,
        normalization="RMSNorm",
        qk_layernorm=True,
        layernorm_zero_centered_gamma=False,
        expert_model_parallel_size=1,
        gradient_accumulation_fusion=False,
    )


def build_attention_layer(args, layout, layer_number, tp_group, cp_cache, device):
    """Build one full DSv4 hybrid layer and bind its THD metadata."""
    from megatron.core.extensions.transformer_engine import HAVE_TE
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_dsv4_hybrid_module_spec_for_backend,
    )
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.spec_utils import build_module

    if not HAVE_TE:
        raise RuntimeError(
            "DSv4 CSA/HCA benchmark execution requires Transformer Engine; "
            "use --validate-only for the dependency-free transition check."
        )

    cp_group = cp_cache[layout.cp_size][0]
    pg_collection = ProcessGroupCollection()
    pg_collection.tp = tp_group
    pg_collection.cp = cp_group
    config = make_config(args, layout.kind, layout.cp_size)
    spec = get_dsv4_hybrid_module_spec_for_backend(config=config, backend=TESpecProvider())
    module = build_module(
        spec,
        config=config,
        layer_number=1,
        pg_collection=pg_collection,
    ).to(device=device)
    module.train()
    packed = make_packed_seq_params(args, layout.cp_size, cp_group, device)

    def forward(hidden):
        output, _ = module(
            hidden_states=hidden,
            attention_mask=None,
            packed_seq_params=packed,
        )
        return output

    forward.module = module
    forward.kind = layout.kind
    forward.cp_size = layout.cp_size
    forward.layer_number = layer_number
    return forward


def synchronize_parameters(layers) -> None:
    """Start every rank from identical replicated layer parameters."""
    if dist.get_world_size() == 1:
        return
    for layer in layers:
        for parameter in layer.module.parameters():
            dist.broadcast(parameter.data, src=0)


def make_stack(args, plan, tp_group, cp_cache, device):
    """Build a CSA/HCA stack with metadata-only transitions between CP layouts."""
    layers = []
    for index, layout in enumerate(plan):
        torch.manual_seed(args.layer_seed + index)
        torch.cuda.manual_seed(args.layer_seed + index)
        layers.append(
            build_attention_layer(args, layout, index + 1, tp_group, cp_cache, device)
        )
    synchronize_parameters(layers)

    def forward(hidden):
        current_cp = plan[0].cp_size
        for layout, layer in zip(plan, layers):
            if layout.cp_size != current_cp:
                hidden = transition_equal_length_thd(
                    hidden,
                    current_cp,
                    layout.cp_size,
                    args.global_batch_size,
                    args.seq_length,
                    dist.get_world_size(),
                )
                current_cp = layout.cp_size
            hidden = layer(hidden)
        return hidden

    return forward, layers


def zero_parameter_grads(layers) -> None:
    for layer in layers:
        layer.module.zero_grad(set_to_none=True)


def peak_memory_gib(world_size: int) -> Tuple[float, float]:
    peaks = torch.tensor(
        [torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()],
        dtype=torch.float64,
        device="cuda",
    )
    if world_size > 1:
        dist.all_reduce(peaks, op=dist.ReduceOp.MAX)
    return tuple((peaks / (1024**3)).tolist())


def benchmark_plan(args, label, plan, tp_group, cp_cache, rank, world_size, device):
    """Measure one full forward/backward stack without DP gradient synchronization."""
    stack, layers = make_stack(args, plan, tp_group, cp_cache, device)
    local_rows = equal_length_local_rows(
        args.global_batch_size, args.seq_length, world_size
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(args.input_seed + rank)
    hidden = torch.randn(
        local_rows,
        1,
        _PROFILES[args.profile]["hidden_size"],
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).requires_grad_(True)

    for _ in range(args.warmup):
        zero_parameter_grads(layers)
        hidden.grad = None
        stack(hidden).sum().backward()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    samples = []
    for _ in range(args.iters):
        zero_parameter_grads(layers)
        hidden.grad = None
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        stack(hidden).sum().backward()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))

    sample_tensor = torch.tensor(samples, dtype=torch.float64, device=device)
    if world_size > 1:
        dist.all_reduce(sample_tensor, op=dist.ReduceOp.MAX)
    samples = sample_tensor.cpu().tolist()
    alloc_gib, reserved_gib = peak_memory_gib(world_size)
    mean_ms = statistics.mean(samples)
    result = {
        "mean_ms": mean_ms,
        "std_ms": statistics.pstdev(samples),
        "tokens_per_second": args.global_batch_size * args.seq_length / (mean_ms / 1000),
        "alloc_gib": alloc_gib,
        "reserved_gib": reserved_gib,
    }
    if rank == 0:
        schedule = "-".join(layer.kind.upper() for layer in plan)
        layouts = "-".join(f"DP{world_size // layer.cp_size}/CP{layer.cp_size}" for layer in plan)
        transitions = sum(a.cp_size != b.cp_size for a, b in zip(plan, plan[1:]))
        print(f"\n{label}: {schedule}")
        print(f"  layouts={layouts}; metadata-only transitions={transitions}")
        print(
            f"  fwd+bwd={mean_ms:.2f} ms (std={result['std_ms']:.2f}); "
            f"tokens/s={result['tokens_per_second']:.0f}; "
            f"peak allocated={alloc_gib:.2f} GiB; reserved={reserved_gib:.2f} GiB"
        )

    del hidden, stack, layers
    torch.cuda.empty_cache()
    return result


def setup_distributed():
    """Initialize one-process or torchrun NCCL execution."""
    if dist.is_initialized():
        return
    device_count = torch.cuda.device_count()
    if device_count < 1:
        raise RuntimeError("Benchmark execution requires at least one CUDA device.")
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    local_rank %= device_count
    torch.cuda.set_device(local_rank)
    requested_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    backend = "nccl" if dist.is_nccl_available() else "gloo"
    if backend != "nccl" and requested_world_size > 1:
        raise RuntimeError("Multi-GPU benchmark execution requires an NCCL-enabled PyTorch build.")
    os.environ.setdefault("USE_LIBUV", "0")
    init_kwargs = {"backend": backend}
    if backend == "nccl":
        init_kwargs["device_id"] = torch.device("cuda", local_rank)
    if "RANK" in os.environ:
        dist.init_process_group(**init_kwargs)
    else:
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(rank=0, world_size=1, **init_kwargs)

    from megatron.core.tensor_parallel.random import (
        _DATA_PARALLEL_RNG_TRACKER_NAME,
        _MODEL_PARALLEL_RNG_TRACKER_NAME,
        get_cuda_rng_tracker,
        initialize_rng_tracker,
    )

    initialize_rng_tracker(use_te_rng_tracker=True)
    tracker = get_cuda_rng_tracker()
    tracker.reset()
    torch.cuda.manual_seed(1234)
    tracker.add(_DATA_PARALLEL_RNG_TRACKER_NAME, 1234)
    tracker.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, 3952 + dist.get_rank())


def run_validation_only(args) -> None:
    """Exercise plan arithmetic and transition autograd without Megatron or CUDA."""
    world_size = args.validation_world_size
    kinds = make_layer_kinds(args.num_csa_layers, args.num_hca_layers, args.layer_seed)
    modes = [args.mode] if args.mode != "both" else ["fixed", "mixed"]
    for mode in modes:
        plan = make_plan(kinds, args, mode)
        validate_plan(plan, args, world_size)
        for source, destination in zip(plan, plan[1:]):
            if source.cp_size == destination.cp_size:
                continue
            for global_rank in range(world_size):
                source_interval = rank_token_interval(
                    global_rank,
                    source.cp_size,
                    args.global_batch_size,
                    args.seq_length,
                    world_size,
                )
                destination_interval = rank_token_interval(
                    global_rank,
                    destination.cp_size,
                    args.global_batch_size,
                    args.seq_length,
                    world_size,
                )
                if source_interval != destination_interval:
                    raise AssertionError(
                        f"rank {global_rank} ownership changes: "
                        f"{source_interval} != {destination_interval}"
                    )
        rows = equal_length_local_rows(args.global_batch_size, args.seq_length, world_size)
        hidden = torch.arange(rows, dtype=torch.float32).view(rows, 1, 1).requires_grad_(True)
        current_cp = plan[0].cp_size
        value = hidden
        for layout in plan[1:]:
            if layout.cp_size != current_cp:
                value = transition_equal_length_thd(
                    value,
                    current_cp,
                    layout.cp_size,
                    args.global_batch_size,
                    args.seq_length,
                    world_size,
                )
                current_cp = layout.cp_size
        value.sum().backward()
        if not torch.equal(value, hidden) or not torch.equal(hidden.grad, torch.ones_like(hidden)):
            raise AssertionError("THD metadata-only transition failed identity/autograd validation.")
        schedule = "-".join(layer.kind.upper() for layer in plan)
        layouts = "-".join(f"DP{world_size // layer.cp_size}/CP{layer.cp_size}" for layer in plan)
        print(f"validated {mode}: {schedule}; {layouts}; local_rows={rows}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Benchmark contiguous-THD fixed/mixed CP for DSv4 CSA and HCA."
    )
    parser.add_argument("--seq-length", type=int, required=True)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--num-csa-layers", type=int, default=1)
    parser.add_argument("--num-hca-layers", type=int, default=1)
    parser.add_argument("--csa-cp-size", type=int, default=2)
    parser.add_argument("--hca-cp-size", type=int, default=8)
    parser.add_argument("--fixed-cp-size", type=int, default=None)
    parser.add_argument("--csa-compress-ratio", type=int, default=4, choices=[4])
    parser.add_argument("--hca-compress-ratio", type=int, default=128, choices=[128])
    parser.add_argument("--csa-window-size", type=int, default=128)
    parser.add_argument("--profile", choices=sorted(_PROFILES), default="flash")
    parser.add_argument(
        "--dsa-indexer-head-dim",
        type=int,
        default=None,
        help="Override the selected profile's DSA indexer head dimension.",
    )
    parser.add_argument("--dsa-kernel-backend", choices=["cudnn", "none"], default="cudnn")
    parser.add_argument(
        "--rope-fusion", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--mode", choices=["fixed", "mixed", "both"], default="both")
    parser.add_argument("--layer-seed", type=int, default=1234)
    parser.add_argument("--input-seed", type=int, default=5678)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--megatron-path",
        default=os.environ.get("MEGATRON_LM_PATH", "/workspace/repos/Megatron-LM"),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate layout/transition invariants on CPU without importing Megatron-LM.",
    )
    parser.add_argument("--validation-world-size", type=int, default=8)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.seq_length < 1:
        raise ValueError("seq_length must be positive.")
    if args.warmup < 0 or args.iters < 1:
        raise ValueError("warmup must be non-negative and iters must be positive.")
    if args.validate_only:
        run_validation_only(args)
        return

    sys.path.insert(0, args.megatron_path)
    from megatron.core.extensions.transformer_engine import HAVE_TE

    if not HAVE_TE:
        raise RuntimeError(
            "DSv4 CSA/HCA benchmark execution requires Transformer Engine; "
            "use --validate-only for the dependency-free transition check."
        )
    setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", torch.cuda.current_device())
    kinds = make_layer_kinds(args.num_csa_layers, args.num_hca_layers, args.layer_seed)
    modes = [args.mode] if args.mode != "both" else ["fixed", "mixed"]
    plans: Dict[str, List[LayerLayout]] = {
        mode: make_plan(kinds, args, mode) for mode in modes
    }
    for plan in plans.values():
        validate_plan(plan, args, world_size)

    cp_sizes = [layer.cp_size for plan in plans.values() for layer in plan]
    cp_cache = make_cp_group_cache(cp_sizes)
    tp_group = make_local_tp_group()
    if rank == 0:
        print(
            "Scope: DSv4 full attention layer, equal-length contiguous THD, "
            "indexer auxiliary loss=0, no DP gradient synchronization."
        )
        print(
            f"profile={args.profile}; world={world_size}; global_batch={args.global_batch_size}; "
            f"seq={args.seq_length}; local_rows={equal_length_local_rows(args.global_batch_size, args.seq_length, world_size)}"
        )
        print(
            "dsa_indexer_head_dim="
            f"{args.dsa_indexer_head_dim or _PROFILES[args.profile]['dsa_indexer_head_dim']}"
        )

    results = {}
    for mode in modes:
        results[mode] = benchmark_plan(
            args, mode, plans[mode], tp_group, cp_cache, rank, world_size, device
        )
        dist.barrier()

    if rank == 0 and args.mode == "both":
        fixed = results["fixed"]
        mixed = results["mixed"]
        print("\nfixed vs mixed")
        print(
            f"  E2E speedup={fixed['mean_ms'] / mixed['mean_ms']:.3f}x; "
            f"allocated delta={mixed['alloc_gib'] - fixed['alloc_gib']:+.2f} GiB; "
            f"reserved delta={mixed['reserved_gib'] - fixed['reserved_gib']:+.2f} GiB"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
