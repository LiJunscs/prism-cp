"""Validate fixed and mixed CP outputs and reduced parameter gradients."""

import os

import torch
import torch.distributed as dist

from utils import (
    _ATTN_BACKEND_ENV,
    benchmark_parameters,
    build_stack,
    make_cp_group_cache,
    make_layer_plan,
    make_layer_types,
    parse_benchmark_args,
    partition_for_cp,
    setup_dist,
    synchronize_parameters,
    validate_plan,
)


def add_correctness_args(parser):
    """Add fixed/mixed correctness-specific arguments."""
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--num-attention-layers", type=int, default=3)
    parser.add_argument("--num-gdn-layers", type=int, default=3)
    parser.add_argument("--layer-seed", type=int, default=1234)
    parser.add_argument("--attn-cp-size", type=int, default=2)
    parser.add_argument("--gdn-cp-size", type=int, default=4)
    parser.add_argument("--fixed-cp-size", type=int, default=None)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    return parser


def parse_args(extra_args_provider=None, argv=None):
    """Parse common arguments plus correctness and caller-provided arguments."""

    def provider(parser):
        add_correctness_args(parser)
        return extra_args_provider(parser) if extra_args_provider is not None else parser

    return parse_benchmark_args(
        provider,
        argv=argv,
        description="Compare fixed and mixed CP outputs and parameter gradients.",
    )


def reset_model_rng(seed):
    """Reset torch and Megatron/TE RNG streams before building one plan."""
    from megatron.core.tensor_parallel.random import (
        _DATA_PARALLEL_RNG_TRACKER_NAME,
        _MODEL_PARALLEL_RNG_TRACKER_NAME,
        get_cuda_rng_tracker,
    )

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    tracker = get_cuda_rng_tracker()
    tracker.reset()
    tracker.add(_DATA_PARALLEL_RNG_TRACKER_NAME, seed)
    tracker.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, seed + 2718 + dist.get_rank())


def local_layout(canonical, cp_size, cp_cache):
    """Slice the DP batch on CPU, then move it to CUDA and partition for CP."""
    if canonical.device.type != "cpu":
        raise ValueError(
            f"The canonical input must stay on CPU, but is on {canonical.device}."
        )
    world_size = dist.get_world_size()
    dp_size = world_size // cp_size
    if canonical.shape[1] % dp_size:
        raise ValueError(
            f"Global batch {canonical.shape[1]} must be divisible by DP={dp_size}."
        )
    local_batch = canonical.shape[1] // dp_size
    dp_rank = dist.get_rank() // cp_size
    batch_start = dp_rank * local_batch
    local_full_sequence = canonical[
        :, batch_start : batch_start + local_batch
    ].contiguous()
    local_full_sequence = local_full_sequence.to(device="cuda")
    return partition_for_cp(local_full_sequence, cp_cache[cp_size][0])


class PlanRuntime:
    """A plan-specific model and communication setup built once per test."""

    def __init__(self, mode, layer_plan, cp_cache, fn, layers, grad_buffer):
        self.mode = mode
        self.layer_plan = layer_plan
        self.cp_cache = cp_cache
        self.fn = fn
        self.layers = layers
        self.grad_buffer = grad_buffer

    @property
    def cp_sizes(self):
        return [cp_size for _, cp_size in self.layer_plan]

    def close(self):
        if self.grad_buffer is not None:
            self.grad_buffer.close()
            self.grad_buffer = None
        self.fn = None
        self.layers = None


def build_plan_runtime(args, mode, layer_types, world_size, tp_group):
    """Build one fixed or mixed plan before executing either correctness run."""
    layer_plan = make_layer_plan(layer_types, args, mode)
    validate_plan(layer_plan, args, world_size)
    cp_sizes = [cp_size for _, cp_size in layer_plan]
    cp_cache = make_cp_group_cache(cp_sizes)
    reset_model_rng(args.layer_seed)
    fn, layers = build_stack(args, layer_plan, tp_group, cp_cache)
    grad_buffer = benchmark_parameters(layers, world_size)
    return PlanRuntime(mode, layer_plan, cp_cache, fn, layers, grad_buffer)


def copy_parameters(source_params, destination_params):
    """Copy exact fixed-plan parameters into the already-built mixed plan."""
    if len(source_params) != len(destination_params):
        raise ValueError(
            f"Parameter count differs: fixed={len(source_params)}, "
            f"mixed={len(destination_params)}."
        )
    with torch.no_grad():
        for index, (source, destination) in enumerate(
            zip(source_params, destination_params)
        ):
            if source.shape != destination.shape or source.dtype != destination.dtype:
                raise ValueError(
                    f"Parameter {index} differs: fixed={tuple(source.shape)}/{source.dtype}, "
                    f"mixed={tuple(destination.shape)}/{destination.dtype}."
                )
            destination.copy_(source)


def gather_canonical_output(local_output, cp_size, seq_length, global_batch_size):
    """Reconstruct a global output without invoking the transition under test."""
    local_output = local_output.contiguous()
    world_size = dist.get_world_size()
    dp_size = world_size // cp_size
    local_batch = global_batch_size // dp_size
    sequence_chunk = seq_length // (2 * cp_size)
    expected_shape = (seq_length // cp_size, local_batch, local_output.shape[-1])
    if tuple(local_output.shape) != expected_shape:
        raise ValueError(
            f"CP{cp_size} output shape {tuple(local_output.shape)} does not match "
            f"expected local shape {expected_shape}."
        )

    gathered = [torch.empty_like(local_output) for _ in range(world_size)]
    dist.all_gather(gathered, local_output)
    canonical = torch.empty(
        seq_length,
        global_batch_size,
        local_output.shape[-1],
        dtype=local_output.dtype,
        device="cpu",
    )
    for global_rank, rank_output in enumerate(gathered):
        rank_output = rank_output.cpu()
        dp_rank = global_rank // cp_size
        cp_rank = global_rank % cp_size
        batch_start = dp_rank * local_batch
        batch_slice = slice(batch_start, batch_start + local_batch)
        front_start = cp_rank * sequence_chunk
        back_start = (2 * cp_size - cp_rank - 1) * sequence_chunk
        canonical[front_start : front_start + sequence_chunk, batch_slice].copy_(
            rank_output[:sequence_chunk]
        )
        canonical[back_start : back_start + sequence_chunk, batch_slice].copy_(
            rank_output[sequence_chunk:]
        )
    return canonical


def capture_correctness_state(
    fn, x, final_cp, seq_length, global_batch_size, grad_buffer
):
    """Run one backward and retain global output and reduced parameter gradients."""
    grad_buffer.zero_grad()
    if x.grad is not None:
        x.grad = None

    out = fn(x)
    canonical_out = gather_canonical_output(
        out.detach(), final_cp, seq_length, global_batch_size
    )
    out.sum().backward()
    grad_buffer.all_reduce()
    return {
        "out": canonical_out,
        "grad_buffer": grad_buffer.grad_data.detach().cpu(),
        "grad_numels": tuple(param.numel() for param in grad_buffer.params),
        "grad_shapes": tuple(tuple(param.shape) for param in grad_buffer.params),
    }


def run_plan(
    args,
    runtime,
    canonical_input,
):
    """Execute an already-built plan from its direct first-CP input layout."""
    reset_model_rng(args.layer_seed)
    cp_sizes = runtime.cp_sizes
    first_cp = cp_sizes[0]
    x = (
        local_layout(canonical_input, first_cp, runtime.cp_cache)
        .detach()
        .requires_grad_(True)
    )
    if dist.get_rank() == 0:
        print(
            f"[correctness][{runtime.mode}] first_cp={first_cp} "
            f"final_cp={cp_sizes[-1]} "
            f"global_input_shape={tuple(canonical_input.shape)} "
            f"local_input_shape={tuple(x.shape)} transitions="
            f"{sum(src != dst for src, dst in zip(cp_sizes, cp_sizes[1:]))}",
            flush=True,
        )

    state = capture_correctness_state(
        runtime.fn,
        x,
        cp_sizes[-1],
        args.seq_length,
        args.global_batch_size,
        runtime.grad_buffer,
    )
    del x
    return state


def tensor_difference(reference, actual, atol, rtol, chunk_size=1 << 20):
    """Compute allclose and maximum errors without a full-sized FP32 temporary."""
    if reference.shape != actual.shape:
        return False, float("inf"), float("inf")

    reference = reference.reshape(-1)
    actual = actual.reshape(-1)
    is_close = True
    max_abs = 0.0
    max_rel = 0.0
    for offset in range(0, reference.numel(), chunk_size):
        ref_chunk = reference[offset : offset + chunk_size].float()
        actual_chunk = actual[offset : offset + chunk_size].float()
        abs_error = (actual_chunk - ref_chunk).abs()
        abs_error = torch.nan_to_num(abs_error, nan=float("inf"), posinf=float("inf"))
        tolerance = atol + rtol * ref_chunk.abs()
        is_close = is_close and bool(torch.all(abs_error <= tolerance))
        max_abs = max(max_abs, abs_error.max().item())
        relative_error = abs_error / ref_chunk.abs().clamp_min(atol)
        relative_error = torch.nan_to_num(
            relative_error, nan=float("inf"), posinf=float("inf")
        )
        max_rel = max(max_rel, relative_error.max().item())
    return is_close, max_abs, max_rel


def gradient_norm_difference(
    reference, actual, param_numels, rtol, chunk_size=1 << 20
):
    """Compare every parameter gradient with a BF16-robust relative L2 metric."""
    if reference.shape != actual.shape:
        return False, float("inf"), -1, float("inf"), float("nan"), float("inf")
    if sum(param_numels) != reference.numel():
        raise ValueError(
            f"Gradient metadata covers {sum(param_numels)} elements, but the buffer "
            f"contains {reference.numel()}."
        )

    reference = reference.reshape(-1)
    actual = actual.reshape(-1)
    offset = 0
    max_param_rel_l2 = 0.0
    worst_param = -1
    total_diff_sq = 0.0
    total_reference_sq = 0.0
    total_actual_sq = 0.0
    total_dot = 0.0
    max_reference = 0.0
    max_abs = 0.0

    for param_index, numel in enumerate(param_numels):
        diff_sq = 0.0
        reference_sq = 0.0
        actual_sq = 0.0
        for chunk_offset in range(0, numel, chunk_size):
            start = offset + chunk_offset
            end = start + min(chunk_size, numel - chunk_offset)
            ref = reference[start:end].float()
            cur = actual[start:end].float()
            diff = cur - ref
            chunk_diff_sq = torch.dot(diff, diff).item()
            chunk_reference_sq = torch.dot(ref, ref).item()
            chunk_actual_sq = torch.dot(cur, cur).item()
            diff_sq += chunk_diff_sq
            reference_sq += chunk_reference_sq
            actual_sq += chunk_actual_sq
            total_dot += torch.dot(ref, cur).item()
            max_reference = max(max_reference, ref.abs().max().item())
            max_abs = max(max_abs, diff.abs().max().item())

        denominator = max(reference_sq, actual_sq, torch.finfo(torch.float32).tiny)
        relative_l2 = (diff_sq / denominator) ** 0.5
        if relative_l2 > max_param_rel_l2:
            max_param_rel_l2 = relative_l2
            worst_param = param_index

        total_diff_sq += diff_sq
        total_reference_sq += reference_sq
        total_actual_sq += actual_sq
        offset += numel

    norm_product = (total_reference_sq * total_actual_sq) ** 0.5
    cosine = total_dot / norm_product if norm_product else 1.0
    global_rel_l2 = (
        total_diff_sq
        / max(total_reference_sq, total_actual_sq, torch.finfo(torch.float32).tiny)
    ) ** 0.5
    normalized_max = max_abs / max(max_reference, torch.finfo(torch.float32).tiny)
    return (
        max_param_rel_l2 <= rtol,
        max_param_rel_l2,
        worst_param,
        global_rel_l2,
        cosine,
        normalized_max,
    )


def compare_states(fixed_state, mixed_state, rank, atol, rtol):
    """Compare canonical outputs and gradient buffers across every rank."""
    results = {}
    for name in ("out", "grad_buffer"):
        elementwise_close, max_abs, max_rel = tensor_difference(
            fixed_state[name], mixed_state[name], atol, rtol
        )
        norm_details = None
        if name == "grad_buffer":
            if fixed_state["grad_numels"] != mixed_state["grad_numels"]:
                raise ValueError("Fixed and mixed parameter-buffer layouts differ.")
            if fixed_state["grad_shapes"] != mixed_state["grad_shapes"]:
                raise ValueError("Fixed and mixed parameter shapes differ.")
            norm_details = gradient_norm_difference(
                fixed_state[name],
                mixed_state[name],
                fixed_state["grad_numels"],
                rtol,
            )
            is_close = norm_details[0]
        else:
            is_close = elementwise_close
        errors = torch.tensor([max_abs, max_rel], dtype=torch.float32, device="cuda")
        passed = torch.tensor(int(is_close), dtype=torch.int32, device="cuda")
        dist.all_reduce(errors, op=dist.ReduceOp.MAX)
        dist.all_reduce(passed, op=dist.ReduceOp.MIN)
        results[name] = bool(passed.item())

        if rank == 0:
            status = "PASS" if results[name] else "FAIL"
            print(
                f"  {name:<12} {status} | max_abs={errors[0].item():.6e} | "
                f"max_rel={errors[1].item():.6e} | atol={atol:.1e} | rtol={rtol:.1e}"
            )
            if norm_details is not None:
                (
                    _,
                    max_param_rel_l2,
                    worst_param,
                    global_rel_l2,
                    cosine,
                    normalized_max,
                ) = norm_details
                worst_shape = (
                    fixed_state["grad_shapes"][worst_param]
                    if worst_param >= 0
                    else None
                )
                print(
                    f"  {'':<12} elementwise={'PASS' if elementwise_close else 'FAIL'} | "
                    f"max_param_rel_l2={max_param_rel_l2:.6e} "
                    f"(param_index={worst_param}, shape={worst_shape}) | "
                    f"global_rel_l2={global_rel_l2:.6e} | "
                    f"cosine={cosine:.8f} | "
                    f"normalized_max={normalized_max:.6e}"
                )

    passed = all(results.values())
    if rank == 0:
        print(f"  overall      {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    args = parse_args()
    for key, value in _ATTN_BACKEND_ENV[args.attention_backend].items():
        os.environ[key] = value

    setup_dist()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    for index in range(world_size):
        tp_group = dist.new_group([index])
        if rank == index:
            local_tp_group = tp_group

    layer_types = make_layer_types(
        args.num_attention_layers,
        args.num_gdn_layers,
        args.layer_seed,
    )
    fixed_runtime = None
    mixed_runtime = None
    passed = False
    try:
        fixed_runtime = build_plan_runtime(
            args, "fixed", layer_types, world_size, local_tp_group
        )
        mixed_runtime = build_plan_runtime(
            args, "mixed", layer_types, world_size, local_tp_group
        )
        synchronize_parameters(fixed_runtime.grad_buffer.params)
        copy_parameters(
            fixed_runtime.grad_buffer.params,
            mixed_runtime.grad_buffer.params,
        )

        input_generator = torch.Generator(device="cpu")
        input_generator.manual_seed(args.layer_seed + 1)
        canonical_input = torch.randn(
            args.seq_length,
            args.global_batch_size,
            args.hidden_size,
            generator=input_generator,
            device="cpu",
            dtype=torch.bfloat16,
        )

        fixed_state = run_plan(args, fixed_runtime, canonical_input)
        dist.barrier()
        mixed_state = run_plan(args, mixed_runtime, canonical_input)
        dist.barrier()

        if rank == 0:
            print("Fixed vs mixed correctness:")
        passed = compare_states(fixed_state, mixed_state, rank, args.atol, args.rtol)
        del canonical_input, fixed_state, mixed_state
    finally:
        if mixed_runtime is not None:
            mixed_runtime.close()
        if fixed_runtime is not None:
            fixed_runtime.close()
        torch.cuda.empty_cache()
        dist.destroy_process_group()
    if not passed:
        raise AssertionError("Fixed/mixed output or reduced gradient comparison failed.")


if __name__ == "__main__":
    main()
