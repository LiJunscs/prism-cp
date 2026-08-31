"""Validate fixed and mixed CP outputs and reduced parameter gradients."""

import os

import torch
import torch.distributed as dist

from bench_fixed_mixed import (
    _temporal_loss,
    make_repeat_stack,
    make_temporal_stack,
    repeat_logical_cp,
    validate_temporal_plan,
)

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
    transmition_cp_size,
    validate_plan,
)


def add_correctness_args(parser):
    """Add fixed/mixed correctness-specific arguments."""
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--num-attention-layers", type=int, default=3)
    parser.add_argument("--num-gdn-layers", type=int, default=3)
    parser.add_argument("--layer-seed", type=int, default=1234)
    parser.add_argument("--attn-cp-size", type=int, default=4)
    parser.add_argument("--gdn-cp-size", type=int, default=1)
    parser.add_argument("--fixed-cp-size", type=int, default=4)
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-2,
        help="Absolute tolerance for output and elementwise gradient diagnostics.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-2,
        help="Relative tolerance for output and elementwise gradient diagnostics.",
    )
    parser.add_argument("--input-grad-rel-l2-tol", type=float, default=5e-3)
    parser.add_argument("--input-grad-cosine-min", type=float, default=0.9999)
    parser.add_argument("--input-grad-normalized-max-tol", type=float, default=1e-2)
    parser.add_argument("--param-grad-rel-l2-tol", type=float, default=1e-2)
    parser.add_argument("--param-grad-global-rel-l2-tol", type=float, default=1e-2)
    parser.add_argument("--param-grad-cosine-min", type=float, default=0.9999)
    parser.add_argument("--param-grad-normalized-max-tol", type=float, default=2e-2)
    parser.add_argument(
        "--correctness-strategy",
        default="all",
        choices=["all", "transition", "legacy", "bundle", "repeat"],
        help=(
            "Run the repeat transition algebra check, original fixed/mixed "
            "redistribution check, bundle check, layered repeat check, or all checks."
        ),
    )
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


def validate_correctness_thresholds(args):
    """Reject invalid or accidentally permissive correctness thresholds."""
    positive = {
        "atol": args.atol,
        "rtol": args.rtol,
        "input_grad_rel_l2_tol": args.input_grad_rel_l2_tol,
        "input_grad_normalized_max_tol": args.input_grad_normalized_max_tol,
        "param_grad_rel_l2_tol": args.param_grad_rel_l2_tol,
        "param_grad_global_rel_l2_tol": args.param_grad_global_rel_l2_tol,
        "param_grad_normalized_max_tol": args.param_grad_normalized_max_tol,
    }
    for name, value in positive.items():
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name} must be in (0, 1), got {value}.")

    for name in ("input_grad_cosine_min", "param_grad_cosine_min"):
        value = getattr(args, name)
        if not -1.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [-1, 1], got {value}.")


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
    grad_buffer = benchmark_parameters(
        layers,
        world_size,
    )
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


def transition_exercising_layer_types(args, round_trip=True):
    """Build a short pattern that crosses both CP layouts deterministically."""
    if args.attn_cp_size == args.gdn_cp_size:
        return ["attention", "gdn", "attention"] if round_trip else ["attention", "gdn"]
    if args.attn_cp_size > args.gdn_cp_size:
        large_cp_type, small_cp_type = "attention", "gdn"
    else:
        large_cp_type, small_cp_type = "gdn", "attention"
    if round_trip:
        return [large_cp_type, small_cp_type, large_cp_type]
    return [large_cp_type, small_cp_type]


def explicit_temporal_plans(args, layer_types):
    """Return a max-CP reference plan and the heterogeneous plan under test."""
    max_cp = max(args.attn_cp_size, args.gdn_cp_size)
    fixed_plan = [(layer_type, max_cp) for layer_type in layer_types]
    mixed_plan = [
        (
            layer_type,
            args.attn_cp_size if layer_type == "attention" else args.gdn_cp_size,
        )
        for layer_type in layer_types
    ]
    return fixed_plan, mixed_plan


def build_temporal_runtime(
    args,
    label,
    layer_plan,
    world_size,
    tp_group,
    strategy,
):
    """Build one explicit plan using the same bundle/repeat path as the benchmark."""
    accumulation_steps = validate_temporal_plan(
        layer_plan, args, world_size, strategy=strategy
    )
    cp_sizes = [cp_size for _, cp_size in layer_plan]
    cp_cache = make_cp_group_cache(cp_sizes)
    reset_model_rng(args.layer_seed)
    _, layers = build_stack(args, layer_plan, tp_group, cp_cache)
    grad_buffer = benchmark_parameters(
        layers,
        world_size,
    )

    if strategy == "bundle":
        first_cp = cp_sizes[0]
        logical_global_batch = args.micro_batch_size * (world_size // first_cp)
        fn = make_temporal_stack(
            layers,
            cp_cache,
            world_size,
            logical_global_batch,
        )
    else:
        fn = make_repeat_stack(layers, cp_cache)

    return (
        PlanRuntime(label, layer_plan, cp_cache, fn, layers, grad_buffer),
        accumulation_steps,
    )


def make_temporal_inputs_from_canonical(
    canonical_input,
    first_cp,
    cp_cache,
    per_step_global_batch,
    accumulation_steps,
):
    """Slice one canonical optimizer batch into deterministic logical microbatches."""
    expected = per_step_global_batch * accumulation_steps
    if canonical_input.shape[1] != expected:
        raise ValueError(
            f"Canonical batch={canonical_input.shape[1]} does not match "
            f"per_step_global_batch={per_step_global_batch} * "
            f"accumulation_steps={accumulation_steps}."
        )
    inputs = []
    for accumulation_index in range(accumulation_steps):
        start = accumulation_index * per_step_global_batch
        step = canonical_input[
            :, start : start + per_step_global_batch
        ].contiguous()
        tensor = (
            local_layout(step, first_cp, cp_cache)
            .detach()
            .requires_grad_(True)
        )
        inputs.append((tensor, 1))
    return inputs


def make_repeat_inputs_from_canonical(
    canonical_input,
    first_cp,
    cp_cache,
    logical_cp,
    per_step_global_batch,
    accumulation_steps,
):
    """Create repeat inputs, duplicating logical samples only across spatial replicas."""
    world_size = dist.get_world_size()
    if logical_cp % first_cp:
        raise ValueError(
            f"logical_cp={logical_cp} must be divisible by first_cp={first_cp}."
        )
    logical_dp = world_size // logical_cp
    if per_step_global_batch % logical_dp:
        raise ValueError(
            f"per_step_global_batch={per_step_global_batch} must divide "
            f"logical_dp={logical_dp}."
        )
    logical_local_batch = per_step_global_batch // logical_dp
    expected = per_step_global_batch * accumulation_steps
    if canonical_input.shape[1] != expected:
        raise ValueError(
            f"Canonical batch={canonical_input.shape[1]} does not match "
            f"repeat optimizer batch={expected}."
        )

    logical_dp_rank = dist.get_rank() // logical_cp
    inputs = []
    for accumulation_index in range(accumulation_steps):
        step_start = accumulation_index * per_step_global_batch
        batch_start = step_start + logical_dp_rank * logical_local_batch
        local_full = canonical_input[
            :, batch_start : batch_start + logical_local_batch
        ].contiguous().to(device="cuda")
        tensor = (
            partition_for_cp(local_full, cp_cache[first_cp][0])
            if first_cp > 1
            else local_full
        )
        inputs.append((tensor.detach().requires_grad_(True), 1))
    return inputs


def _reconstruct_cp_replica(
    gathered,
    group_start,
    cp_size,
    seq_length,
    local_batch,
):
    """Reconstruct one full-sequence physical DP replica from gathered rank tensors."""
    hidden = gathered[group_start].shape[-1]
    sequence_chunk = seq_length // (2 * cp_size)
    full = torch.empty(
        seq_length,
        local_batch,
        hidden,
        dtype=gathered[group_start].dtype,
        device="cpu",
    )
    for cp_rank in range(cp_size):
        rank_tensor = gathered[group_start + cp_rank].cpu()
        front_start = cp_rank * sequence_chunk
        back_start = (2 * cp_size - cp_rank - 1) * sequence_chunk
        full[front_start : front_start + sequence_chunk].copy_(
            rank_tensor[:sequence_chunk]
        )
        full[back_start : back_start + sequence_chunk].copy_(
            rank_tensor[sequence_chunk:]
        )
    return full


def gather_repeat_canonical_output(
    local_output,
    cp_size,
    logical_cp,
    seq_length,
    logical_global_batch,
    atol,
    rtol,
):
    """Recover unique repeat-mode samples and verify physical replicas agree."""
    world_size = dist.get_world_size()
    logical_dp = world_size // logical_cp
    if logical_global_batch % logical_dp:
        raise ValueError(
            f"logical_global_batch={logical_global_batch} must divide logical DP={logical_dp}."
        )
    local_batch = logical_global_batch // logical_dp
    expected_shape = (seq_length // cp_size, local_batch, local_output.shape[-1])
    if tuple(local_output.shape) != expected_shape:
        raise ValueError(
            f"repeat CP{cp_size} output shape {tuple(local_output.shape)} does not "
            f"match expected {expected_shape}."
        )
    if logical_cp % cp_size:
        raise ValueError(
            f"logical_cp={logical_cp} must be divisible by final CP={cp_size}."
        )

    gathered = [torch.empty_like(local_output) for _ in range(world_size)]
    dist.all_gather(gathered, local_output.contiguous())
    replication = logical_cp // cp_size
    canonical = torch.empty(
        seq_length,
        logical_global_batch,
        local_output.shape[-1],
        dtype=local_output.dtype,
        device="cpu",
    )
    replica_ok = True
    max_replica_abs = 0.0
    max_replica_rel = 0.0
    for logical_dp_rank in range(logical_dp):
        logical_group_start = logical_dp_rank * logical_cp
        replicas = [
            _reconstruct_cp_replica(
                gathered,
                logical_group_start + replica * cp_size,
                cp_size,
                seq_length,
                local_batch,
            )
            for replica in range(replication)
        ]
        reference = replicas[0]
        for replica in replicas[1:]:
            close, max_abs, max_rel = tensor_difference(
                reference, replica, atol, rtol
            )
            replica_ok = replica_ok and close
            max_replica_abs = max(max_replica_abs, max_abs)
            max_replica_rel = max(max_replica_rel, max_rel)
        batch_start = logical_dp_rank * local_batch
        canonical[:, batch_start : batch_start + local_batch].copy_(reference)

    replica_passed = torch.tensor(
        int(replica_ok), dtype=torch.int32, device="cuda"
    )
    replica_errors = torch.tensor(
        [max_replica_abs, max_replica_rel], dtype=torch.float32, device="cuda"
    )
    dist.all_reduce(replica_passed, op=dist.ReduceOp.MIN)
    dist.all_reduce(replica_errors, op=dist.ReduceOp.MAX)
    if dist.get_rank() == 0:
        print(
            f"    repeat replica consistency: "
            f"{'PASS' if replica_passed.item() else 'FAIL'} | "
            f"max_abs={replica_errors[0].item():.6e} | "
            f"max_rel={replica_errors[1].item():.6e}"
        )
    if not replica_passed.item():
        raise AssertionError("Repeat physical replicas diverged before canonical gather.")
    return canonical


def gather_repeat_canonical_input_grad(
    local_grad,
    cp_size,
    logical_cp,
    seq_length,
    logical_global_batch,
):
    """Sum physical repeat-leaf gradients back into one logical input gradient."""
    world_size = dist.get_world_size()
    logical_dp = world_size // logical_cp
    if logical_global_batch % logical_dp:
        raise ValueError(
            f"logical_global_batch={logical_global_batch} must divide logical DP={logical_dp}."
        )
    local_batch = logical_global_batch // logical_dp
    if logical_cp % cp_size:
        raise ValueError(
            f"logical_cp={logical_cp} must be divisible by input CP={cp_size}."
        )
    expected_shape = (seq_length // cp_size, local_batch, local_grad.shape[-1])
    if tuple(local_grad.shape) != expected_shape:
        raise ValueError(
            f"repeat input grad shape {tuple(local_grad.shape)} does not match "
            f"expected {expected_shape}."
        )

    gathered = [torch.empty_like(local_grad) for _ in range(world_size)]
    dist.all_gather(gathered, local_grad.contiguous())
    replication = logical_cp // cp_size
    canonical = torch.empty(
        seq_length,
        logical_global_batch,
        local_grad.shape[-1],
        dtype=local_grad.dtype,
        device="cpu",
    )
    for logical_dp_rank in range(logical_dp):
        logical_group_start = logical_dp_rank * logical_cp
        logical_grad = None
        for replica in range(replication):
            replica_grad = _reconstruct_cp_replica(
                gathered,
                logical_group_start + replica * cp_size,
                cp_size,
                seq_length,
                local_batch,
            )
            logical_grad = (
                replica_grad
                if logical_grad is None
                else logical_grad + replica_grad
            )
        batch_start = logical_dp_rank * local_batch
        canonical[:, batch_start : batch_start + local_batch].copy_(logical_grad)
    return canonical


def run_repeat_transition_correctness(args, rank, world_size):
    """Validate repeat CP remeshing and its adjoints without model kernels."""
    logical_cp = max(args.attn_cp_size, args.gdn_cp_size)
    low_cp = min(args.attn_cp_size, args.gdn_cp_size)
    if logical_cp == low_cp:
        if rank == 0:
            print("Repeat transition algebra: SKIP (only one CP size)")
        return True
    if world_size % logical_cp:
        raise ValueError(
            f"repeat transition logical CP={logical_cp} must divide world_size={world_size}."
        )

    cp_cache = make_cp_group_cache([low_cp, logical_cp])
    logical_dp = world_size // logical_cp
    logical_global_batch = args.micro_batch_size * logical_dp
    input_generator = torch.Generator(device="cpu")
    input_generator.manual_seed(args.layer_seed + 73)
    canonical_input = torch.randn(
        args.seq_length,
        logical_global_batch,
        args.hidden_size,
        generator=input_generator,
        device="cpu",
        dtype=torch.bfloat16,
    )

    cases = (
        ("roundtrip", logical_cp, (logical_cp, low_cp, logical_cp)),
        ("terminal_replication", logical_cp, (logical_cp, low_cp)),
        ("initial_replication", low_cp, (low_cp, logical_cp)),
    )
    all_passed = True
    for case_name, first_cp, path in cases:
        inputs = make_repeat_inputs_from_canonical(
            canonical_input,
            first_cp,
            cp_cache,
            logical_cp,
            logical_global_batch,
            accumulation_steps=1,
        )
        if len(inputs) != 1:
            raise RuntimeError(
                f"repeat transition algebra expects one input, got {len(inputs)}."
            )
        tensor = inputs[0][0]
        current_cp = path[0]
        for next_cp in path[1:]:
            tensor = transmition_cp_size(
                tensor,
                current_cp,
                next_cp,
                cp_cache,
                strategy="repeat",
            )
            current_cp = next_cp

        canonical_output = gather_repeat_canonical_output(
            tensor.detach(),
            current_cp,
            logical_cp,
            args.seq_length,
            logical_global_batch,
            args.atol,
            args.rtol,
        )
        replication_factor = logical_cp // current_cp
        (tensor.sum() / replication_factor).backward()
        if inputs[0][0].grad is None:
            raise RuntimeError(
                f"repeat transition algebra case {case_name} produced no input gradient."
            )
        canonical_input_grad = gather_repeat_canonical_input_grad(
            inputs[0][0].grad.detach(),
            first_cp,
            logical_cp,
            args.seq_length,
            logical_global_batch,
        )

        out_close, out_max_abs, out_max_rel = tensor_difference(
            canonical_input, canonical_output, args.atol, args.rtol
        )
        expected_grad = torch.ones_like(canonical_input_grad)
        grad_close, grad_max_abs, grad_max_rel = tensor_difference(
            expected_grad, canonical_input_grad, args.atol, args.rtol
        )
        local_checks = torch.tensor(
            [int(out_close), int(grad_close)], dtype=torch.int32, device="cuda"
        )
        errors = torch.tensor(
            [out_max_abs, out_max_rel, grad_max_abs, grad_max_rel],
            dtype=torch.float32,
            device="cuda",
        )
        dist.all_reduce(errors, op=dist.ReduceOp.MAX)
        dist.all_reduce(local_checks, op=dist.ReduceOp.MIN)
        out_passed = bool(local_checks[0].item())
        grad_passed = bool(local_checks[1].item())
        case_passed = out_passed and grad_passed
        all_passed = all_passed and case_passed
        if rank == 0:
            print(
                f"Repeat transition algebra ({case_name}, path={list(path)}):"
            )
            print(
                f"  out          {'PASS' if out_passed else 'FAIL'} | "
                f"max_abs={errors[0].item():.6e} | max_rel={errors[1].item():.6e}"
            )
            print(
                f"  input_grad   {'PASS' if grad_passed else 'FAIL'} | "
                f"max_abs={errors[2].item():.6e} | max_rel={errors[3].item():.6e}"
            )
            print(f"  overall      {'PASS' if case_passed else 'FAIL'}")
        del inputs, tensor, canonical_output, canonical_input_grad
        dist.barrier()

    return all_passed


def capture_temporal_correctness_state(
    args,
    runtime,
    inputs,
    strategy,
    per_step_global_batch,
    accumulation_steps,
    logical_cp,
):
    """Run one temporal/repeat optimizer step and retain outputs, input grads, params."""
    reset_model_rng(args.layer_seed)
    runtime.grad_buffer.zero_grad()
    for tensor, _ in inputs:
        if tensor.grad is not None:
            tensor.grad = None

    outputs = runtime.fn(inputs)
    final_cp = runtime.cp_sizes[-1]
    if len(outputs) != accumulation_steps or any(width != 1 for _, width in outputs):
        raise RuntimeError(
            f"{strategy} correctness pattern must finish with one width-1 output "
            f"per logical microbatch; got widths={[width for _, width in outputs]}."
        )

    canonical_outputs = []
    for tensor, _ in outputs:
        if strategy == "repeat":
            canonical = gather_repeat_canonical_output(
                tensor.detach(),
                final_cp,
                logical_cp,
                args.seq_length,
                per_step_global_batch,
                args.atol,
                args.rtol,
            )
        else:
            canonical = gather_canonical_output(
                tensor.detach(),
                final_cp,
                args.seq_length,
                per_step_global_batch,
            )
        canonical_outputs.append(canonical)

    replication_factor = logical_cp // final_cp if strategy == "repeat" else 1
    _temporal_loss(
        outputs,
        accumulation_steps,
        replication_factor=replication_factor,
    ).backward()

    first_cp = runtime.cp_sizes[0]
    canonical_input_grads = []
    for tensor, _ in inputs:
        if tensor.grad is None:
            raise RuntimeError("Temporal correctness input gradient was not produced.")
        if strategy == "repeat":
            canonical_input_grads.append(
                gather_repeat_canonical_input_grad(
                    tensor.grad.detach(),
                    first_cp,
                    logical_cp,
                    args.seq_length,
                    per_step_global_batch,
                )
            )
        else:
            canonical_input_grads.append(
                gather_canonical_output(
                    tensor.grad.detach(),
                    first_cp,
                    args.seq_length,
                    per_step_global_batch,
                )
            )

    runtime.grad_buffer.all_reduce()
    return {
        "out": torch.cat(canonical_outputs, dim=1),
        "input_grad": torch.cat(canonical_input_grads, dim=1),
        "input_grad_scale": accumulation_steps,
        "grad_buffer": runtime.grad_buffer.grad_data.detach().cpu(),
        "grad_numels": tuple(param.numel() for param in runtime.grad_buffer.params),
        "grad_shapes": tuple(tuple(param.shape) for param in runtime.grad_buffer.params),
    }


def run_temporal_correctness_case(
    args,
    strategy,
    layer_types,
    rank,
    world_size,
    tp_group,
    case_name,
):
    """Compare one bundle/repeat mixed execution against a max-CP reference."""
    fixed_plan, mixed_plan = explicit_temporal_plans(args, layer_types)
    fixed_runtime = mixed_runtime = None
    try:
        fixed_runtime, fixed_accumulation = build_temporal_runtime(
            args,
            f"{strategy}_{case_name}_fixed",
            fixed_plan,
            world_size,
            tp_group,
            strategy,
        )
        mixed_runtime, mixed_accumulation = build_temporal_runtime(
            args,
            f"{strategy}_{case_name}_mixed",
            mixed_plan,
            world_size,
            tp_group,
            strategy,
        )
        if fixed_accumulation != mixed_accumulation:
            raise ValueError(
                f"{strategy} correctness requires matching accumulation counts; "
                f"fixed={fixed_accumulation}, mixed={mixed_accumulation}."
            )
        accumulation_steps = fixed_accumulation

        synchronize_parameters(fixed_runtime.grad_buffer.params)
        copy_parameters(
            fixed_runtime.grad_buffer.params,
            mixed_runtime.grad_buffer.params,
        )

        logical_cp = max(args.attn_cp_size, args.gdn_cp_size)
        if strategy == "repeat":
            fixed_logical_cp = repeat_logical_cp(args, fixed_plan)
            mixed_logical_cp = repeat_logical_cp(args, mixed_plan)
            if fixed_logical_cp != mixed_logical_cp:
                raise ValueError(
                    f"repeat logical CP differs: fixed={fixed_logical_cp}, "
                    f"mixed={mixed_logical_cp}."
                )
            logical_cp = fixed_logical_cp

        per_step_global_batch = args.micro_batch_size * (world_size // logical_cp)
        if per_step_global_batch * accumulation_steps != args.global_batch_size:
            raise ValueError(
                f"{strategy} correctness expects global batch "
                f"{per_step_global_batch * accumulation_steps}, got "
                f"{args.global_batch_size}."
            )

        input_generator = torch.Generator(device="cpu")
        input_generator.manual_seed(args.layer_seed + 100 + len(case_name))
        canonical_input = torch.randn(
            args.seq_length,
            args.global_batch_size,
            args.hidden_size,
            generator=input_generator,
            device="cpu",
            dtype=torch.bfloat16,
        )

        def make_inputs(runtime):
            if strategy == "repeat":
                return make_repeat_inputs_from_canonical(
                    canonical_input,
                    runtime.cp_sizes[0],
                    runtime.cp_cache,
                    logical_cp,
                    per_step_global_batch,
                    accumulation_steps,
                )
            return make_temporal_inputs_from_canonical(
                canonical_input,
                runtime.cp_sizes[0],
                runtime.cp_cache,
                per_step_global_batch,
                accumulation_steps,
            )

        fixed_inputs = make_inputs(fixed_runtime)
        fixed_state = capture_temporal_correctness_state(
            args,
            fixed_runtime,
            fixed_inputs,
            strategy,
            per_step_global_batch,
            accumulation_steps,
            logical_cp,
        )
        del fixed_inputs
        dist.barrier()

        mixed_inputs = make_inputs(mixed_runtime)
        mixed_state = capture_temporal_correctness_state(
            args,
            mixed_runtime,
            mixed_inputs,
            strategy,
            per_step_global_batch,
            accumulation_steps,
            logical_cp,
        )
        del mixed_inputs
        dist.barrier()

        if rank == 0:
            print(
                f"{strategy.capitalize()} correctness ({case_name}, "
                f"layout={fixed_runtime.cp_sizes}->{mixed_runtime.cp_sizes}):"
            )
        passed = compare_states(
            fixed_state,
            mixed_state,
            rank,
            args,
        )
        del canonical_input, fixed_state, mixed_state
        return passed
    finally:
        if mixed_runtime is not None:
            mixed_runtime.close()
        if fixed_runtime is not None:
            fixed_runtime.close()
        torch.cuda.empty_cache()


def run_bundle_correctness(args, rank, world_size, tp_group):
    """Exercise a max-CP -> min-CP -> max-CP temporal-folding round trip."""
    layer_types = transition_exercising_layer_types(args, round_trip=True)
    return run_temporal_correctness_case(
        args,
        "bundle",
        layer_types,
        rank,
        world_size,
        tp_group,
        "roundtrip",
    )


def run_repeat_correctness(args, rank, world_size, tp_group):
    """Validate repeat adjoints, terminal scaling, and initially replicated input."""
    roundtrip = run_temporal_correctness_case(
        args,
        "repeat",
        transition_exercising_layer_types(args, round_trip=True),
        rank,
        world_size,
        tp_group,
        "roundtrip",
    )
    dist.barrier()
    terminal = run_temporal_correctness_case(
        args,
        "repeat",
        transition_exercising_layer_types(args, round_trip=False),
        rank,
        world_size,
        tp_group,
        "terminal_replication",
    )
    dist.barrier()
    initial_types = list(
        reversed(transition_exercising_layer_types(args, round_trip=False))
    )
    initial = run_temporal_correctness_case(
        args,
        "repeat",
        initial_types,
        rank,
        world_size,
        tp_group,
        "initial_replication",
    )
    return roundtrip and terminal and initial


def tensor_error_statistics(reference, actual, atol, rtol, chunk_size=1 << 20):
    """Compute strict elementwise error statistics in bounded FP32 chunks."""
    if reference.shape != actual.shape:
        return {
            "shape_match": False,
            "numel": max(reference.numel(), actual.numel()),
            "violation_count": max(reference.numel(), actual.numel()),
            "violation_fraction": 1.0,
            "nonfinite_count": max(reference.numel(), actual.numel()),
            "max_abs": float("inf"),
            "max_rel": float("inf"),
            "max_scaled_error": float("inf"),
            "near_zero_count": 0,
            "near_zero_violation_count": 0,
            "non_near_zero_count": 0,
            "non_near_zero_violation_count": 0,
            "max_abs_index": -1,
            "max_abs_reference": float("nan"),
            "max_abs_actual": float("nan"),
            "max_scaled_index": -1,
            "max_scaled_reference": float("nan"),
            "max_scaled_actual": float("nan"),
        }

    reference = reference.reshape(-1)
    actual = actual.reshape(-1)
    violation_count = 0
    nonfinite_count = 0
    max_abs = 0.0
    max_rel = 0.0
    max_scaled_error = 0.0
    near_zero_count = 0
    near_zero_violation_count = 0
    non_near_zero_count = 0
    non_near_zero_violation_count = 0
    max_abs_index = -1
    max_abs_reference = float("nan")
    max_abs_actual = float("nan")
    max_scaled_index = -1
    max_scaled_reference = float("nan")
    max_scaled_actual = float("nan")
    tiny = torch.finfo(torch.float32).tiny
    for offset in range(0, reference.numel(), chunk_size):
        ref_chunk = reference[offset : offset + chunk_size].float()
        actual_chunk = actual[offset : offset + chunk_size].float()
        finite = torch.isfinite(ref_chunk) & torch.isfinite(actual_chunk)
        chunk_nonfinite = int((~finite).sum().item())
        nonfinite_count += chunk_nonfinite
        safe_ref = torch.where(finite, ref_chunk, torch.zeros_like(ref_chunk))
        safe_actual = torch.where(finite, actual_chunk, torch.zeros_like(actual_chunk))
        abs_error = (safe_actual - safe_ref).abs()
        tolerance = atol + rtol * safe_ref.abs()
        violations = (~finite) | (abs_error > tolerance)
        violation_count += int(violations.sum().item())
        near_zero = finite & (safe_ref.abs() <= atol)
        non_near_zero = finite & (~near_zero)
        near_zero_count += int(near_zero.sum().item())
        near_zero_violation_count += int((violations & near_zero).sum().item())
        non_near_zero_count += int(non_near_zero.sum().item())
        non_near_zero_violation_count += int(
            (violations & non_near_zero).sum().item()
        )

        chunk_max_abs, chunk_max_abs_index = abs_error.max(dim=0)
        chunk_max_abs = chunk_max_abs.item()
        if chunk_max_abs > max_abs:
            local_index = int(chunk_max_abs_index.item())
            max_abs = chunk_max_abs
            max_abs_index = offset + local_index
            max_abs_reference = ref_chunk[local_index].item()
            max_abs_actual = actual_chunk[local_index].item()
        relative_error = abs_error / safe_ref.abs().clamp_min(atol)
        max_rel = max(max_rel, relative_error.max().item())
        scaled_error = abs_error / tolerance.clamp_min(tiny)
        chunk_max_scaled, chunk_max_scaled_index = scaled_error.max(dim=0)
        chunk_max_scaled = chunk_max_scaled.item()
        if chunk_max_scaled > max_scaled_error:
            local_index = int(chunk_max_scaled_index.item())
            max_scaled_error = chunk_max_scaled
            max_scaled_index = offset + local_index
            max_scaled_reference = ref_chunk[local_index].item()
            max_scaled_actual = actual_chunk[local_index].item()

    if nonfinite_count:
        max_abs = max_rel = max_scaled_error = float("inf")
    numel = reference.numel()
    return {
        "shape_match": True,
        "numel": numel,
        "violation_count": violation_count,
        "violation_fraction": violation_count / max(numel, 1),
        "nonfinite_count": nonfinite_count,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "max_scaled_error": max_scaled_error,
        "near_zero_count": near_zero_count,
        "near_zero_violation_count": near_zero_violation_count,
        "non_near_zero_count": non_near_zero_count,
        "non_near_zero_violation_count": non_near_zero_violation_count,
        "max_abs_index": max_abs_index,
        "max_abs_reference": max_abs_reference,
        "max_abs_actual": max_abs_actual,
        "max_scaled_index": max_scaled_index,
        "max_scaled_reference": max_scaled_reference,
        "max_scaled_actual": max_scaled_actual,
    }


def tensor_difference(reference, actual, atol, rtol, chunk_size=1 << 20):
    """Return exact allclose status plus legacy maximum-error diagnostics."""
    stats = tensor_error_statistics(reference, actual, atol, rtol, chunk_size)
    return (
        stats["shape_match"] and stats["violation_count"] == 0,
        stats["max_abs"],
        stats["max_rel"],
    )


def tensor_norm_difference(reference, actual, rtol, chunk_size=1 << 20):
    """Compare a tensor pair with scale-invariant FP32 norm diagnostics."""
    if reference.shape != actual.shape:
        return (
            False,
            float("inf"),
            float("nan"),
            float("inf"),
            float("nan"),
            float("nan"),
        )

    reference = reference.reshape(-1)
    actual = actual.reshape(-1)
    diff_sq = 0.0
    reference_sq = 0.0
    actual_sq = 0.0
    dot = 0.0
    max_reference = 0.0
    max_abs = 0.0
    has_nonfinite = False
    for offset in range(0, reference.numel(), chunk_size):
        ref = reference[offset : offset + chunk_size].float()
        cur = actual[offset : offset + chunk_size].float()
        finite = torch.isfinite(ref) & torch.isfinite(cur)
        if not bool(torch.all(finite)):
            has_nonfinite = True
        ref = torch.where(finite, ref, torch.zeros_like(ref))
        cur = torch.where(finite, cur, torch.zeros_like(cur))
        diff = cur - ref
        diff_sq += torch.dot(diff, diff).item()
        reference_sq += torch.dot(ref, ref).item()
        actual_sq += torch.dot(cur, cur).item()
        dot += torch.dot(ref, cur).item()
        max_reference = max(max_reference, ref.abs().max().item())
        max_abs = max(max_abs, diff.abs().max().item())

    tiny = torch.finfo(torch.float32).tiny
    denominator = max(reference_sq, actual_sq, tiny)
    global_rel_l2 = (diff_sq / denominator) ** 0.5
    norm_product = (reference_sq * actual_sq) ** 0.5
    if norm_product:
        cosine = dot / norm_product
    else:
        cosine = 1.0 if reference_sq == 0.0 and actual_sq == 0.0 else 0.0
    normalized_max = max_abs / max(max_reference, tiny)
    if has_nonfinite:
        return (
            False,
            float("inf"),
            float("nan"),
            float("inf"),
            reference_sq**0.5,
            actual_sq**0.5,
        )
    return (
        global_rel_l2 <= rtol,
        global_rel_l2,
        cosine,
        normalized_max,
        reference_sq**0.5,
        actual_sq**0.5,
    )


def gradient_norm_difference(
    reference, actual, param_numels, rtol, chunk_size=1 << 20
):
    """Compare every parameter gradient with a BF16-robust relative L2 metric."""
    if reference.shape != actual.shape:
        return (
            False,
            float("inf"),
            -1,
            float("inf"),
            float("nan"),
            float("inf"),
            float("inf"),
            -1,
        )
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
    max_param_normalized_max = 0.0
    worst_normalized_param = -1
    has_nonfinite = False

    for param_index, numel in enumerate(param_numels):
        diff_sq = 0.0
        reference_sq = 0.0
        actual_sq = 0.0
        param_max_reference = 0.0
        param_max_abs = 0.0
        for chunk_offset in range(0, numel, chunk_size):
            start = offset + chunk_offset
            end = start + min(chunk_size, numel - chunk_offset)
            ref = reference[start:end].float()
            cur = actual[start:end].float()
            finite = torch.isfinite(ref) & torch.isfinite(cur)
            if not bool(torch.all(finite)):
                has_nonfinite = True
            ref = torch.where(finite, ref, torch.zeros_like(ref))
            cur = torch.where(finite, cur, torch.zeros_like(cur))
            diff = cur - ref
            chunk_diff_sq = torch.dot(diff, diff).item()
            chunk_reference_sq = torch.dot(ref, ref).item()
            chunk_actual_sq = torch.dot(cur, cur).item()
            diff_sq += chunk_diff_sq
            reference_sq += chunk_reference_sq
            actual_sq += chunk_actual_sq
            total_dot += torch.dot(ref, cur).item()
            chunk_max_reference = ref.abs().max().item()
            chunk_max_abs = diff.abs().max().item()
            param_max_reference = max(param_max_reference, chunk_max_reference)
            param_max_abs = max(param_max_abs, chunk_max_abs)
            max_reference = max(max_reference, chunk_max_reference)
            max_abs = max(max_abs, chunk_max_abs)

        denominator = max(reference_sq, actual_sq, torch.finfo(torch.float32).tiny)
        relative_l2 = (diff_sq / denominator) ** 0.5
        if relative_l2 > max_param_rel_l2:
            max_param_rel_l2 = relative_l2
            worst_param = param_index
        param_normalized_max = param_max_abs / max(
            param_max_reference, torch.finfo(torch.float32).tiny
        )
        if param_normalized_max > max_param_normalized_max:
            max_param_normalized_max = param_normalized_max
            worst_normalized_param = param_index

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
    if has_nonfinite:
        return (
            False,
            float("inf"),
            worst_param,
            float("inf"),
            float("nan"),
            float("inf"),
            float("inf"),
            worst_normalized_param,
        )
    return (
        max_param_rel_l2 <= rtol,
        max_param_rel_l2,
        worst_param,
        global_rel_l2,
        cosine,
        normalized_max,
        max_param_normalized_max,
        worst_normalized_param,
    )


def _distributed_max(values):
    """Return worst-rank maxima for a short list of scalar diagnostics."""
    metrics = torch.tensor(values, dtype=torch.float64, device="cuda")
    dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
    return metrics.cpu().tolist()


def _distributed_min(values):
    """Return worst-rank minima for a short list of scalar diagnostics."""
    metrics = torch.tensor(values, dtype=torch.float64, device="cuda")
    dist.all_reduce(metrics, op=dist.ReduceOp.MIN)
    return metrics.cpu().tolist()


def compare_states(fixed_state, mixed_state, rank, args):
    """Compare output and gradients with BF16-appropriate correctness gates."""
    results = {}
    names = ["out"]
    if "input_grad" in fixed_state or "input_grad" in mixed_state:
        if "input_grad" not in fixed_state or "input_grad" not in mixed_state:
            raise ValueError("Only one correctness state contains input_grad.")
        if fixed_state.get("input_grad_scale") != mixed_state.get("input_grad_scale"):
            raise ValueError(
                "Fixed and mixed input gradients use different accumulation scales."
            )
        names.append("input_grad")
    names.append("grad_buffer")

    for name in names:
        error_stats = tensor_error_statistics(
            fixed_state[name], mixed_state[name], args.atol, args.rtol
        )
        shape_match = bool(
            _distributed_min([int(error_stats["shape_match"])])[0]
        )
        max_abs, max_rel, violation_count, nonfinite_count = _distributed_max(
            [
                error_stats["max_abs"],
                error_stats["max_rel"],
                error_stats["violation_count"],
                error_stats["nonfinite_count"],
            ]
        )
        violation_count = int(violation_count)
        nonfinite_count = int(nonfinite_count)

        if name == "out":
            gate_checks = {
                "elementwise_allclose": shape_match and violation_count == 0,
                "finite": nonfinite_count == 0,
                "shape": shape_match,
            }
            details = (
                f"max_abs={max_abs:.6e} | max_rel={max_rel:.6e} | "
                f"violations={violation_count}"
            )
        elif name == "input_grad":
            (
                _,
                global_rel_l2,
                cosine,
                normalized_max,
                reference_l2,
                actual_l2,
            ) = tensor_norm_difference(
                fixed_state[name],
                mixed_state[name],
                args.input_grad_rel_l2_tol,
            )
            global_rel_l2, normalized_max, reference_l2, actual_l2 = (
                _distributed_max(
                    [global_rel_l2, normalized_max, reference_l2, actual_l2]
                )
            )
            cosine = _distributed_min([cosine])[0]
            gate_checks = {
                "global_rel_l2": (
                    global_rel_l2 <= args.input_grad_rel_l2_tol
                ),
                "cosine": cosine >= args.input_grad_cosine_min,
                "normalized_max": (
                    normalized_max <= args.input_grad_normalized_max_tol
                ),
                "finite": nonfinite_count == 0,
                "shape": shape_match,
            }
            details = (
                f"global_rel_l2={global_rel_l2:.6e}"
                f"<={args.input_grad_rel_l2_tol:.1e} | "
                f"cosine={cosine:.8f}>={args.input_grad_cosine_min:.6f} | "
                f"normalized_max={normalized_max:.6e}"
                f"<={args.input_grad_normalized_max_tol:.1e} | "
                f"max_abs={max_abs:.6e} | "
                f"reference_l2={reference_l2:.6e} | actual_l2={actual_l2:.6e}"
            )
        else:
            if fixed_state["grad_numels"] != mixed_state["grad_numels"]:
                raise ValueError("Fixed and mixed parameter-buffer layouts differ.")
            if fixed_state["grad_shapes"] != mixed_state["grad_shapes"]:
                raise ValueError("Fixed and mixed parameter shapes differ.")

            (
                _,
                max_param_rel_l2,
                worst_param,
                global_rel_l2,
                cosine,
                global_normalized_max,
                max_param_normalized_max,
                worst_normalized_param,
            ) = gradient_norm_difference(
                fixed_state[name],
                mixed_state[name],
                fixed_state["grad_numels"],
                args.param_grad_rel_l2_tol,
            )
            (
                max_param_rel_l2,
                global_rel_l2,
                global_normalized_max,
                max_param_normalized_max,
            ) = _distributed_max(
                [
                    max_param_rel_l2,
                    global_rel_l2,
                    global_normalized_max,
                    max_param_normalized_max,
                ]
            )
            cosine = _distributed_min([cosine])[0]
            gate_checks = {
                "max_param_rel_l2": (
                    max_param_rel_l2 <= args.param_grad_rel_l2_tol
                ),
                "global_rel_l2": (
                    global_rel_l2 <= args.param_grad_global_rel_l2_tol
                ),
                "cosine": cosine >= args.param_grad_cosine_min,
                "normalized_max": (
                    max_param_normalized_max
                    <= args.param_grad_normalized_max_tol
                ),
                "finite": nonfinite_count == 0,
                "shape": shape_match,
            }
            worst_shape = (
                fixed_state["grad_shapes"][worst_param]
                if worst_param >= 0
                else None
            )
            worst_normalized_shape = (
                fixed_state["grad_shapes"][worst_normalized_param]
                if worst_normalized_param >= 0
                else None
            )
            details = (
                f"max_param_rel_l2={max_param_rel_l2:.6e}"
                f"<={args.param_grad_rel_l2_tol:.1e} "
                f"(param={worst_param}, shape={worst_shape}) | "
                f"global_rel_l2={global_rel_l2:.6e}"
                f"<={args.param_grad_global_rel_l2_tol:.1e} | "
                f"cosine={cosine:.8f}>={args.param_grad_cosine_min:.6f} | "
                f"max_param_normalized_max={max_param_normalized_max:.6e}"
                f"<={args.param_grad_normalized_max_tol:.1e} "
                f"(param={worst_normalized_param}, "
                f"shape={worst_normalized_shape}) | "
                f"global_normalized_max={global_normalized_max:.6e} | "
                f"max_abs={max_abs:.6e}"
            )

        is_close = all(gate_checks.values())
        passed = torch.tensor(int(is_close), dtype=torch.int32, device="cuda")
        dist.all_reduce(passed, op=dist.ReduceOp.MIN)
        results[name] = bool(passed.item())

        if rank == 0:
            status = "PASS" if results[name] else "FAIL"
            print(f"  {name:<12} {status} | {details}")
            if not results[name]:
                failed = [
                    check
                    for check, check_passed in gate_checks.items()
                    if not check_passed
                ]
                print(f"  {'':<12} failed_gates={','.join(failed)}")

    passed = all(results.values())
    if rank == 0:
        print(f"  overall      {'PASS' if passed else 'FAIL'}")
    return passed

def main():
    args = parse_args()
    validate_correctness_thresholds(args)
    for key, value in _ATTN_BACKEND_ENV[args.attention_backend].items():
        os.environ[key] = value

    setup_dist()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if rank == 0:
        print(
            "Correctness gates: "
            f"output(atol={args.atol:.1e}, rtol={args.rtol:.1e}); "
            f"input(rel_l2<={args.input_grad_rel_l2_tol:.1e}, "
            f"cosine>={args.input_grad_cosine_min:.6f}, "
            f"normalized_max<={args.input_grad_normalized_max_tol:.1e}); "
            f"param(per_tensor_rel_l2<={args.param_grad_rel_l2_tol:.1e}, "
            f"global_rel_l2<={args.param_grad_global_rel_l2_tol:.1e}, "
            f"cosine>={args.param_grad_cosine_min:.6f}, "
            f"normalized_max<={args.param_grad_normalized_max_tol:.1e}); "
            "all tensors require matching shapes and finite values"
        )
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
    passed = True
    try:
        if args.correctness_strategy in ("all", "transition", "repeat"):
            passed = run_repeat_transition_correctness(
                args, rank, world_size
            ) and passed
            dist.barrier()

        if args.correctness_strategy in ("all", "legacy"):
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
                print("Fixed vs mixed redistribution correctness:")
            passed = compare_states(
                fixed_state, mixed_state, rank, args
            ) and passed
            del canonical_input, fixed_state, mixed_state
            mixed_runtime.close()
            fixed_runtime.close()
            mixed_runtime = fixed_runtime = None
            torch.cuda.empty_cache()
            dist.barrier()

        if args.correctness_strategy in ("all", "bundle"):
            passed = run_bundle_correctness(
                args, rank, world_size, local_tp_group
            ) and passed
            dist.barrier()

        if args.correctness_strategy in ("all", "repeat"):
            passed = run_repeat_correctness(
                args, rank, world_size, local_tp_group
            ) and passed
            dist.barrier()
    finally:
        if mixed_runtime is not None:
            mixed_runtime.close()
        if fixed_runtime is not None:
            fixed_runtime.close()
        torch.cuda.empty_cache()
        dist.destroy_process_group()
    if not passed:
        raise AssertionError("One or more CP correctness comparisons failed.")


if __name__ == "__main__":
    main()
