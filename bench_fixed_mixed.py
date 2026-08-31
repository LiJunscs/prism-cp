# bench_fixed_mixed.py
"""Profile fixed/mixed CP stacks and simple temporal-folding prototypes."""

import math
import os
import statistics

import torch
import torch.distributed as dist

from utils import (
    _ATTN_BACKEND_ENV,
    benchmark_parameters,
    build_stack,
    describe_plan,
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


def add_fixed_mixed_args(parser):
    """Add end-to-end mixed-plan profiling arguments."""
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=1,
        help="Global batch represented by one optimizer-step benchmark.",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=1,
        help=(
            "Per-DP-replica microbatch used by temporal modes. Gradient "
            "accumulation is global_batch_size / (micro_batch_size * first_dp)."
        ),
    )
    parser.add_argument("--num-attention-layers", type=int, default=3)
    parser.add_argument("--num-gdn-layers", type=int, default=3)
    parser.add_argument(
        "--layer-seed",
        type=int,
        default=1234,
        help="Seed used to shuffle the Attention/GDN layer multiset.",
    )
    parser.add_argument("--attn-cp-size", type=int, default=2)
    parser.add_argument("--gdn-cp-size", type=int, default=4)
    parser.add_argument("--fixed-cp-size", type=int, default=None)
    parser.add_argument(
        "--mode",
        default="both",
        choices=["mixed", "fixed", "both"],
        help="Run mixed, fixed, or both plans with the same layer schedule.",
    )
    parser.add_argument(
        "--temporal-strategy",
        default="none",
        choices=["none", "bundle", "repeat", "both"],
        help=(
            "Use optimizer-step temporal execution instead of the legacy single "
            "spatial batch. 'bundle' uses distinct accumulated microbatches; "
            "'repeat' keeps distinct accumulated microbatches but replicates a "
            "logical microbatch spatially only when CP contraction expands DP."
        ),
    )
    return parser


def parse_args(extra_args_provider=None, argv=None):
    """Parse common arguments plus fixed/mixed-specific and caller arguments."""

    def provider(parser):
        add_fixed_mixed_args(parser)
        return extra_args_provider(parser) if extra_args_provider is not None else parser

    return parse_benchmark_args(
        provider,
        argv=argv,
        description="Profile end-to-end fixed/mixed CP plans and transition overhead.",
    )


def reset_memory_tracking():
    """Start a phase-level peak-memory measurement after releasing cache blocks."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def peak_memory_gib(world_size):
    """Return worst-rank allocated and reserved CUDA peaks in GiB."""
    torch.cuda.synchronize()
    peaks = torch.tensor(
        [torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()],
        dtype=torch.float64,
        device="cuda",
    )
    if world_size > 1:
        dist.all_reduce(peaks, op=dist.ReduceOp.MAX)
    return (peaks / (1024**3)).tolist()


class InlineForwardProfiler:
    """Record layer and transition forward time inside real stack executions."""

    def __init__(self, layer_plan, iterations):
        self.descriptors = []
        current_cp = layer_plan[0][1]
        for layer_index, (layer_type, cp_size) in enumerate(layer_plan):
            if cp_size != current_cp:
                self.descriptors.append(
                    {
                        "kind": "transition",
                        "index": layer_index - 1,
                        "src_cp": current_cp,
                        "dst_cp": cp_size,
                    }
                )
                current_cp = cp_size
            self.descriptors.append(
                {
                    "kind": "layer",
                    "index": layer_index,
                    "layer_type": layer_type,
                    "cp_size": cp_size,
                }
            )

        self.events = [
            [
                (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for _ in self.descriptors
            ]
            for _ in range(iterations)
        ]
        self.active_iteration = None
        self.call_index = 0

    def begin_iteration(self, iteration):
        self.active_iteration = iteration
        self.call_index = 0

    def measure(self, operation):
        if self.active_iteration is None:
            return operation()
        if self.call_index >= len(self.descriptors):
            raise RuntimeError("Inline profiler observed more calls than expected.")
        start, end = self.events[self.active_iteration][self.call_index]
        start.record()
        result = operation()
        end.record()
        self.call_index += 1
        return result

    def end_iteration(self):
        if self.call_index != len(self.descriptors):
            raise RuntimeError(
                f"Inline profiler observed {self.call_index} calls, expected "
                f"{len(self.descriptors)}."
            )
        self.active_iteration = None

    def summarize(self):
        profiles = []
        for call_index, descriptor in enumerate(self.descriptors):
            samples = [
                iteration_events[call_index][0].elapsed_time(
                    iteration_events[call_index][1]
                )
                for iteration_events in self.events
            ]
            profiles.append(
                {
                    **descriptor,
                    "mean_ms": statistics.mean(samples),
                    "std_ms": statistics.pstdev(samples),
                }
            )
        return profiles


class InlineBackwardProfiler:
    """Record per-component backward time through boundary-tensor grad hooks.

    The stack graph is a single chain ``v_0 -> c_0 -> v_1 -> ... -> c_{M-1} ->
    v_M``, so backward executes strictly serially on the current stream in
    reverse order. A hook on boundary tensor ``v_k`` fires exactly when
    ``grad(v_k)`` is produced, i.e. when the backward of the component that
    produced ``v_k`` finishes. Component ``c_k`` (forward index ``k``) then
    occupies the stream interval ``[B[k+1], B[k]]``.
    """

    def __init__(self, num_components, iterations):
        self.num_boundaries = num_components + 1
        self.boundary_events = [
            [
                torch.cuda.Event(enable_timing=True)
                for _ in range(self.num_boundaries)
            ]
            for _ in range(iterations)
        ]
        self.backward_start_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(iterations)
        ]
        self.sync_events = [
            (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for _ in range(iterations)
        ]
        self.active_iteration = None
        self._input_hook_installed = False

    def begin_iteration(self, iteration):
        self.active_iteration = iteration

    def end_iteration(self):
        self.active_iteration = None

    def register_boundary_hook(self, tensor, boundary_index):
        def grad_hook(*unused):
            if self.active_iteration is not None:
                self.boundary_events[self.active_iteration][boundary_index].record()

        tensor.register_hook(grad_hook)

    def register_input_hook_once(self, tensor):
        if not self._input_hook_installed:
            self.register_boundary_hook(tensor, 0)
            self._input_hook_installed = True

    def mark_backward_start(self):
        self.backward_start_events[self.active_iteration].record()

    def mark_sync_start(self):
        self.sync_events[self.active_iteration][0].record()

    def mark_sync_end(self):
        self.sync_events[self.active_iteration][1].record()

    def summarize(self):
        if not all(
            event.query()
            for iteration_events in self.boundary_events
            for event in iteration_events
        ):
            raise RuntimeError(
                "Inline backward profiler did not record every boundary event; "
                "the autograd graph shape does not match the layer plan."
            )

        means, stds = [], []
        for boundary_index in range(self.num_boundaries - 1):
            samples = [
                iteration_events[boundary_index + 1].elapsed_time(
                    iteration_events[boundary_index]
                )
                for iteration_events in self.boundary_events
            ]
            means.append(statistics.mean(samples))
            stds.append(statistics.pstdev(samples))

        loss_samples = [
            start.elapsed_time(iteration_events[-1])
            for start, iteration_events in zip(
                self.backward_start_events, self.boundary_events
            )
        ]
        if self.sync_events and self.sync_events[0][0].query():
            sync_samples = [
                start.elapsed_time(end) for start, end in self.sync_events
            ]
            grad_sync_ms = statistics.mean(sync_samples)
            grad_sync_std_ms = statistics.pstdev(sync_samples)
        else:
            grad_sync_ms = grad_sync_std_ms = 0.0

        return {
            "mean_ms": means,
            "std_ms": stds,
            "loss_bwd_ms": statistics.mean(loss_samples),
            "loss_bwd_std_ms": statistics.pstdev(loss_samples),
            "grad_sync_ms": grad_sync_ms,
            "grad_sync_std_ms": grad_sync_std_ms,
        }


def make_profiled_stack(layers, cp_cache, profiler, backward_profiler):
    """Wrap the actual stack forward with inline layer/transition events."""

    def forward(x):
        current_cp = layers[0][1]
        component = 0
        backward_profiler.register_input_hook_once(x)
        for _, cp_size, layer_fn in layers:
            if cp_size != current_cp:
                source_cp = current_cp
                x = profiler.measure(
                    lambda value=x, src=source_cp, dst=cp_size: transmition_cp_size(
                        value, src, dst, cp_cache
                    )
                )
                component += 1
                backward_profiler.register_boundary_hook(x, component)
                current_cp = cp_size
            x = profiler.measure(lambda value=x, fn=layer_fn: fn(value))
            component += 1
            backward_profiler.register_boundary_hook(x, component)
        return x

    return forward


def timed_profiled_fwd_bwd(
    fn,
    x,
    warmup,
    iters,
    profiler,
    backward_profiler,
    zero_grad=None,
    post_backward=None,
):
    """Time real fwd+bwd iterations while profiling both paths inline."""
    for _ in range(warmup):
        if zero_grad is not None:
            zero_grad()
        if x.grad is not None:
            x.grad = None
        out = fn(x)
        out.sum().backward()
        if post_backward is not None:
            post_backward()
    torch.cuda.synchronize()

    times = []
    for iteration in range(iters):
        if zero_grad is not None:
            zero_grad()
        if x.grad is not None:
            x.grad = None
        profiler.begin_iteration(iteration)
        backward_profiler.begin_iteration(iteration)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn(x)
        backward_profiler.mark_backward_start()
        out.sum().backward()
        if post_backward is not None:
            backward_profiler.mark_sync_start()
            post_backward()
            backward_profiler.mark_sync_end()
        end.record()
        torch.cuda.synchronize()
        profiler.end_iteration()
        backward_profiler.end_iteration()
        times.append(start.elapsed_time(end))
    return times, profiler.summarize(), backward_profiler.summarize()


def bench_stack(
    label, fn, x, args, rank, world_size, grad_buffer, profiler, backward_profiler
):
    global_tokens = args.global_batch_size * args.seq_length

    # Execute only the real end-to-end path. Its forward and backward carry the
    # intrusive layer/transition events; no extra stack pass is used for the
    # breakdown.
    reset_memory_tracking()
    t_both, forward_profiles, backward_summary = timed_profiled_fwd_bwd(
        fn,
        x,
        args.warmup,
        args.iters,
        profiler,
        backward_profiler,
        zero_grad=grad_buffer.zero_grad,
        post_backward=grad_buffer.all_reduce,
    )
    avg_both = statistics.mean(t_both)
    tokens_both = global_tokens / (avg_both / 1000)
    mem_both, mem_both_reserved = peak_memory_gib(world_size)

    module_forward = sum(
        profile["mean_ms"]
        for profile in forward_profiles
        if profile["kind"] == "layer"
    )
    attention_forward = sum(
        profile["mean_ms"]
        for profile in forward_profiles
        if profile["kind"] == "layer" and profile["layer_type"] == "attention"
    )
    gdn_forward = sum(
        profile["mean_ms"]
        for profile in forward_profiles
        if profile["kind"] == "layer" and profile["layer_type"] == "gdn"
    )
    transition_forward = sum(
        profile["mean_ms"]
        for profile in forward_profiles
        if profile["kind"] == "transition"
    )
    avg_fwd = module_forward + transition_forward

    layer_backward = sum(
        bwd
        for profile, bwd in zip(forward_profiles, backward_summary["mean_ms"])
        if profile["kind"] == "layer"
    )
    transition_backward = sum(
        bwd
        for profile, bwd in zip(forward_profiles, backward_summary["mean_ms"])
        if profile["kind"] == "transition"
    )
    loss_backward = backward_summary["loss_bwd_ms"]
    grad_sync = backward_summary["grad_sync_ms"]
    avg_bwd = layer_backward + transition_backward + loss_backward + grad_sync
    bwd_residual = avg_both - avg_fwd - avg_bwd
    tokens_fwd = global_tokens / (avg_fwd / 1000)
    tokens_bwd = global_tokens / (avg_bwd / 1000) if avg_bwd > 0 else float("inf")

    if rank == 0:
        print(
            f"  {label + ' fwd (inline)':<20} {avg_fwd:9.2f} "
            f"{tokens_fwd:14.0f} {'n/a':>10}"
        )
        print(
            f"  {label + ' bwd (measured)':<20} {avg_bwd:9.2f} "
            f"{tokens_bwd:14.0f} {'n/a':>10}"
        )
        print(
            f"  {label + ' both':<20} {avg_both:9.2f} "
            f"{tokens_both:14.0f} {mem_both:10.2f}"
        )
        print(
            f"  latency std (ms): both={statistics.pstdev(t_both):.2f}; "
            "forward and backward components are measured inline"
        )

        print("  inline forward breakdown (inside measured fwd+bwd iterations):")
        for profile in forward_profiles:
            if profile["kind"] == "layer":
                description = (
                    f"layer[{profile['index']}] {profile['layer_type']} "
                    f"CP{profile['cp_size']}"
                )
            else:
                description = (
                    f"transition[{profile['index']}] "
                    f"CP{profile['src_cp']}->CP{profile['dst_cp']}"
                )
            print(
                f"    {description:<34} {profile['mean_ms']:9.2f} ms "
                f"(std={profile['std_ms']:.2f})"
            )

        print("  inline backward breakdown (inside measured fwd+bwd iterations):")
        for profile, bwd_mean, bwd_std in zip(
            forward_profiles,
            backward_summary["mean_ms"],
            backward_summary["std_ms"],
        ):
            if profile["kind"] == "layer":
                description = (
                    f"layer[{profile['index']}] {profile['layer_type']} "
                    f"CP{profile['cp_size']}"
                )
            else:
                description = (
                    f"transition[{profile['index']}] "
                    f"CP{profile['src_cp']}->CP{profile['dst_cp']}"
                )
            print(
                f"    {description:<34} {bwd_mean:9.2f} ms "
                f"(std={bwd_std:.2f})"
            )
        print(
            f"    {'loss bwd':<34} {loss_backward:9.2f} ms "
            f"(std={backward_summary['loss_bwd_std_ms']:.2f})"
        )
        print(
            f"    {'grad sync (all_reduce)':<34} {grad_sync:9.2f} ms "
            f"(std={backward_summary['grad_sync_std_ms']:.2f})"
        )

        print(
            "  measured E2E decomposition: "
            f"attention_fwd={attention_forward:.2f} ms | "
            f"gdn_fwd={gdn_forward:.2f} ms | "
            f"transition_fwd={transition_forward:.2f} ms | "
            f"layer_bwd={layer_backward:.2f} ms | "
            f"transition_bwd={transition_backward:.2f} ms | "
            f"loss_bwd={loss_backward:.2f} ms | "
            f"grad_sync={grad_sync:.2f} ms | "
            f"bwd_residual={bwd_residual:.2f} ms | "
            f"E2E={avg_both:.2f} ms"
        )

    return {
        "fwd": avg_fwd,
        "bwd": avg_bwd,
        "both": avg_both,
        "both_std": statistics.pstdev(t_both),
        "tokens_fwd": tokens_fwd,
        "tokens_bwd": tokens_bwd,
        "tokens_both": tokens_both,
        "mem_both": mem_both,
        "mem_both_reserved": mem_both_reserved,
        "attention_forward": attention_forward,
        "gdn_forward": gdn_forward,
        "transition_forward": transition_forward,
        "layer_backward": layer_backward,
        "transition_backward": transition_backward,
        "loss_backward": loss_backward,
        "grad_sync": grad_sync,
        "bwd_residual": bwd_residual,
        "forward_profiles": forward_profiles,
        "backward_profiles": [
            {
                **profile,
                "bwd_mean_ms": bwd_mean,
                "bwd_std_ms": bwd_std,
            }
            for profile, bwd_mean, bwd_std in zip(
                forward_profiles,
                backward_summary["mean_ms"],
                backward_summary["std_ms"],
            )
        ],
    }


def temporal_gradient_accumulation_steps(args, first_cp, world_size):
    """Validate optimizer-step batching and return its accumulation count."""
    if args.micro_batch_size < 1:
        raise ValueError("micro_batch_size must be positive.")
    first_dp = world_size // first_cp
    logical_global_batch = args.micro_batch_size * first_dp
    if args.global_batch_size % logical_global_batch:
        raise ValueError(
            f"global_batch_size={args.global_batch_size} must be divisible by "
            f"micro_batch_size={args.micro_batch_size} * first_dp={first_dp}."
        )
    return args.global_batch_size // logical_global_batch


def repeat_logical_cp(args, layer_plan):
    """Return the non-replicated CP domain used to define repeat-mode samples."""
    candidates = [cp_size for _, cp_size in layer_plan]
    candidates.extend([args.attn_cp_size, args.gdn_cp_size])
    if args.fixed_cp_size is not None:
        candidates.append(args.fixed_cp_size)
    logical_cp = max(candidates)
    world_size = dist.get_world_size()
    if world_size % logical_cp:
        raise ValueError(
            f"repeat logical CP={logical_cp} must divide world_size={world_size}."
        )
    for _, cp_size in layer_plan:
        if logical_cp % cp_size:
            raise ValueError(
                f"repeat logical CP={logical_cp} must be divisible by every plan CP; "
                f"got CP={cp_size}."
            )
    return logical_cp


def validate_temporal_plan(layer_plan, args, world_size, strategy="bundle"):
    """Validate optimizer-step legality for bundle or spatial-repeat execution."""
    if args.global_batch_size < 1:
        raise ValueError("global_batch_size must be positive.")
    if strategy not in ("bundle", "repeat"):
        raise ValueError(f"Unknown temporal strategy {strategy!r}.")

    for _, cp_size in layer_plan:
        if cp_size < 1 or world_size % cp_size:
            raise ValueError(
                f"CP size {cp_size} must divide world_size={world_size}."
            )
        if args.seq_length % (2 * cp_size) or args.seq_length // cp_size < 128:
            raise ValueError(
                f"Invalid seq_length={args.seq_length} for CP={cp_size}."
            )
        dp_size = world_size // cp_size
        if strategy == "bundle" and args.global_batch_size % dp_size:
            raise ValueError(
                f"optimizer-step global_batch_size={args.global_batch_size} must "
                f"be divisible by DP={dp_size} for CP={cp_size}."
            )

    for (_, src_cp), (_, dst_cp) in zip(layer_plan, layer_plan[1:]):
        if src_cp == dst_cp:
            continue
        large_cp, small_cp = max(src_cp, dst_cp), min(src_cp, dst_cp)
        if large_cp % small_cp:
            raise ValueError(
                f"CP transition {src_cp}->{dst_cp} requires nested CP sizes."
            )

    if strategy == "bundle":
        return temporal_gradient_accumulation_steps(
            args, layer_plan[0][1], world_size
        )

    logical_cp = repeat_logical_cp(args, layer_plan)
    logical_dp = world_size // logical_cp
    logical_microbatch = args.micro_batch_size * logical_dp
    if args.global_batch_size % logical_microbatch:
        raise ValueError(
            f"repeat global_batch_size={args.global_batch_size} must be divisible by "
            f"micro_batch_size={args.micro_batch_size} * logical_dp={logical_dp}."
        )
    return args.global_batch_size // logical_microbatch


def _minimum_temporal_width(logical_global_batch, dp_size):
    """Minimum logical-microbatch cohort that is integral under ``dp_size``."""
    return dp_size // math.gcd(dp_size, logical_global_batch)


def temporal_remesh_payloads(
    payloads,
    src_cp,
    dst_cp,
    cp_cache,
    world_size,
    logical_global_batch,
):
    """Remesh ``(tensor, logical_width)`` payloads while preserving identities.

    A payload width is the number of original logical microbatches represented by
    its tensor. Moving to a larger DP may require concatenating adjacent temporal
    payloads; moving back to a smaller DP splits them into the smallest legal
    cohorts for the destination layout.
    """
    if src_cp == dst_cp:
        return payloads

    src_dp = world_size // src_cp
    dst_dp = world_size // dst_cp
    dst_width = _minimum_temporal_width(logical_global_batch, dst_dp)

    for tensor, width in payloads:
        expected_numerator = logical_global_batch * width
        if expected_numerator % src_dp:
            raise ValueError(
                f"Payload width={width} is not spatially legal under DP={src_dp}."
            )
        expected_batch = expected_numerator // src_dp
        if tensor.shape[1] != expected_batch:
            raise ValueError(
                f"Payload batch={tensor.shape[1]} does not match width={width}, "
                f"logical_global_batch={logical_global_batch}, DP={src_dp}."
            )

    remeshed = []
    if dst_dp > src_dp:
        cohort_width = sum(width for _, width in payloads)
        if cohort_width % dst_width:
            raise ValueError(
                f"CP{src_cp}->CP{dst_cp} needs temporal cohort width "
                f"divisible by {dst_width}, but this optimizer step provides "
                f"width={cohort_width}."
            )
        packed = torch.cat([tensor for tensor, _ in payloads], dim=1)
        return [
            (
                transmition_cp_size(packed, src_cp, dst_cp, cp_cache),
                cohort_width,
            )
        ]

    target_batch = logical_global_batch * dst_width // dst_dp
    for tensor, width in payloads:
        if width % dst_width:
            raise ValueError(
                f"Payload width={width} cannot split into destination cohort "
                f"width={dst_width} for CP{dst_cp}."
            )
        transitioned = transmition_cp_size(
            tensor, src_cp, dst_cp, cp_cache
        )
        chunks = transitioned.split(target_batch, dim=1)
        expected_chunks = width // dst_width
        if len(chunks) != expected_chunks or any(
            chunk.shape[1] != target_batch for chunk in chunks
        ):
            raise RuntimeError(
                f"CP{src_cp}->CP{dst_cp} produced {len(chunks)} batch chunks; "
                f"expected {expected_chunks} chunks of size {target_batch}."
            )
        remeshed.extend((chunk.contiguous(), dst_width) for chunk in chunks)
    return remeshed


def make_temporal_stack(
    layers, cp_cache, world_size, logical_global_batch
):
    """Execute bundle-mode CP blocks over temporally foldable payloads."""
    blocks = []
    for _, cp_size, layer_fn in layers:
        if not blocks or blocks[-1][0] != cp_size:
            blocks.append((cp_size, [layer_fn]))
        else:
            blocks[-1][1].append(layer_fn)

    def forward(payloads):
        payloads = list(payloads)
        current_cp = blocks[0][0]
        for cp_size, block_fns in blocks:
            if cp_size != current_cp:
                payloads = temporal_remesh_payloads(
                    payloads,
                    current_cp,
                    cp_size,
                    cp_cache,
                    world_size,
                    logical_global_batch,
                )
                current_cp = cp_size

            block_outputs = []
            for tensor, width in payloads:
                for layer_fn in block_fns:
                    tensor = layer_fn(tensor)
                block_outputs.append((tensor, width))
            payloads = block_outputs
        return payloads

    return forward


def make_repeat_stack(layers, cp_cache):
    """Execute each logical microbatch independently with spatial repeat remeshing."""
    def forward(payloads):
        outputs = []
        for tensor, width in payloads:
            if width != 1:
                raise ValueError(
                    f"repeat mode expects width=1 logical microbatches, got {width}."
                )
            current_cp = layers[0][1]
            for _, cp_size, layer_fn in layers:
                if cp_size != current_cp:
                    tensor = transmition_cp_size(
                        tensor,
                        current_cp,
                        cp_size,
                        cp_cache,
                        strategy="repeat",
                    )
                    current_cp = cp_size
                tensor = layer_fn(tensor)
            outputs.append((tensor, width))
        return outputs

    return forward


def make_temporal_inputs(
    args,
    first_cp,
    cp_cache,
    world_size,
    accumulation_steps,
    strategy,
    logical_cp=None,
):
    """Create optimizer-step inputs for bundle or spatial-repeat execution."""
    if strategy not in ("bundle", "repeat"):
        raise ValueError(f"Unknown temporal input strategy {strategy!r}.")

    if strategy == "repeat":
        if logical_cp is None:
            raise ValueError("repeat inputs require logical_cp.")
        if logical_cp % first_cp:
            raise ValueError(
                f"logical_cp={logical_cp} must be divisible by first_cp={first_cp}."
            )
        seed_dp_rank = dist.get_rank() // logical_cp
    else:
        seed_dp_rank = dist.get_rank() // first_cp

    inputs = []
    for accumulation_index in range(accumulation_steps):
        x_full = torch.empty(
            args.seq_length,
            args.micro_batch_size,
            args.hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        torch.cuda.manual_seed(
            args.layer_seed
            + 1
            + seed_dp_rank * accumulation_steps
            + accumulation_index
        )
        if dist.get_rank(cp_cache[first_cp][0]) == 0:
            x_full.normal_()
        dist.broadcast(
            x_full,
            src=cp_cache[first_cp][1][0],
            group=cp_cache[first_cp][0],
        )
        x = (
            partition_for_cp(x_full, cp_cache[first_cp][0])
            if first_cp > 1
            else x_full
        )
        x = x.detach().requires_grad_(True)
        inputs.append((x, 1))
    return inputs


def _temporal_loss(outputs, accumulation_steps, replication_factor=1):
    loss = outputs[0][0].sum()
    for tensor, _ in outputs[1:]:
        loss = loss + tensor.sum()
    return loss / (accumulation_steps * replication_factor)


def timed_temporal_fwd_bwd(
    fn,
    inputs,
    warmup,
    iters,
    accumulation_steps,
    replication_factor=1,
    zero_grad=None,
    post_backward=None,
):
    """Time one optimizer-step-shaped temporal forward/backward workload."""

    def prepare_iteration():
        if zero_grad is not None:
            zero_grad()
        for tensor, _ in inputs:
            if tensor.grad is not None:
                tensor.grad = None

    for _ in range(warmup):
        prepare_iteration()
        outputs = fn(inputs)
        _temporal_loss(
            outputs, accumulation_steps, replication_factor
        ).backward()
        if post_backward is not None:
            post_backward()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        prepare_iteration()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        outputs = fn(inputs)
        _temporal_loss(
            outputs, accumulation_steps, replication_factor
        ).backward()
        if post_backward is not None:
            post_backward()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


def bench_temporal_stack(
    label,
    fn,
    inputs,
    args,
    rank,
    world_size,
    first_dp,
    accumulation_steps,
    strategy,
    grad_buffer,
    logical_cp,
    final_cp,
):
    """Measure the temporal prototype at optimizer-step granularity."""
    reset_memory_tracking()
    replication_factor = (
        logical_cp // final_cp if strategy == "repeat" else 1
    )
    times = timed_temporal_fwd_bwd(
        fn,
        inputs,
        args.warmup,
        args.iters,
        accumulation_steps,
        replication_factor=replication_factor,
        zero_grad=grad_buffer.zero_grad,
        post_backward=grad_buffer.all_reduce,
    )
    avg_both = statistics.mean(times)
    logical_tokens = args.global_batch_size * args.seq_length
    unique_batch = args.global_batch_size
    unique_tokens = unique_batch * args.seq_length
    logical_tokens_s = logical_tokens / (avg_both / 1000)
    unique_tokens_s = unique_tokens / (avg_both / 1000)
    mem_both, mem_both_reserved = peak_memory_gib(world_size)

    if rank == 0:
        semantic = (
            "distinct logical microbatches"
            if strategy == "bundle"
            else "distinct logical microbatches with transition-local spatial replication"
        )
        print(f"  {label}: {semantic}")
        print(
            f"  grad_accumulation_steps={accumulation_steps} | "
            f"logical_global_batch={args.global_batch_size} | "
            f"final_replication={replication_factor}"
        )
        print(
            f"  optimizer-step both={avg_both:.2f} ms "
            f"(std={statistics.pstdev(times):.2f}) | "
            f"logical_tokens/s={logical_tokens_s:.0f} | "
            f"unique_tokens/s={unique_tokens_s:.0f} | "
            f"alloc={mem_both:.2f} GiB"
        )
        if strategy == "repeat":
            print(
                "  repeat redundancy is layer-dependent; logical throughput counts "
                "each optimizer sample once even when lower-CP blocks execute replicas."
            )

    return {
        "both": avg_both,
        "both_std": statistics.pstdev(times),
        "executed_tokens_s": logical_tokens_s,
        "logical_tokens_s": logical_tokens_s,
        "unique_tokens_s": unique_tokens_s,
        "mem_both": mem_both,
        "mem_both_reserved": mem_both_reserved,
        "accumulation_steps": accumulation_steps,
        "unique_global_batch": unique_batch,
        "final_replication": replication_factor,
    }


def run_temporal_plan(
    args, mode, strategy, layer_types, rank, world_size, tp_group
):
    """Build and measure one fixed/mixed temporal optimizer-step plan."""
    layer_plan = make_layer_plan(layer_types, args, mode)
    accumulation_steps = validate_temporal_plan(
        layer_plan, args, world_size, strategy=strategy
    )
    cp_sizes = [cp_size for _, cp_size in layer_plan]
    cp_cache = make_cp_group_cache(cp_sizes)
    first_cp = cp_sizes[0]
    first_dp = world_size // first_cp
    if strategy == "repeat":
        logical_cp = repeat_logical_cp(args, layer_plan)
        logical_dp = world_size // logical_cp
        logical_global_batch = args.micro_batch_size * logical_dp
    else:
        logical_cp = first_cp
        logical_global_batch = args.micro_batch_size * first_dp

    torch.manual_seed(args.layer_seed)
    torch.cuda.manual_seed(args.layer_seed)
    _, layers = build_stack(args, layer_plan, tp_group, cp_cache)
    grad_buffer = benchmark_parameters(layers, world_size)
    synchronize_parameters(grad_buffer.params)
    inputs = make_temporal_inputs(
        args,
        first_cp,
        cp_cache,
        world_size,
        accumulation_steps,
        strategy,
        logical_cp=logical_cp,
    )
    if strategy == "repeat":
        fn = make_repeat_stack(layers, cp_cache)
    else:
        fn = make_temporal_stack(
            layers, cp_cache, world_size, logical_global_batch
        )

    schedule, layout_plan, transitions = describe_plan(layer_plan)
    label = f"temporal_{strategy}_{mode}"
    if rank == 0:
        print(f"\n  {label}: schedule={schedule}")
        print(
            f"  layout_plan={layout_plan} | transitions={transitions} | "
            f"first_dp={first_dp} | logical_cp={logical_cp} | "
            f"micro_batch_size={args.micro_batch_size}"
        )

    try:
        stats = bench_temporal_stack(
            label,
            fn,
            inputs,
            args,
            rank,
            world_size,
            first_dp,
            accumulation_steps,
            strategy,
            grad_buffer,
            logical_cp,
            cp_sizes[-1],
        )
    finally:
        grad_buffer.close()
    del grad_buffer, fn, layers, inputs
    torch.cuda.empty_cache()
    return stats
def run_randomized_plan(args, mode, layer_types, rank, world_size, tp_group):
    """Build and measure one plan for the shared randomized layer schedule."""
    layer_plan = make_layer_plan(layer_types, args, mode)
    validate_plan(layer_plan, args, world_size)
    cp_sizes = [cp for _, cp in layer_plan]
    cp_cache = make_cp_group_cache(cp_sizes)
    first_cp = cp_sizes[0]
    first_dp = world_size // first_cp
    torch.manual_seed(args.layer_seed)
    torch.cuda.manual_seed(args.layer_seed)
    _, layers = build_stack(args, layer_plan, tp_group, cp_cache)
    profiler = InlineForwardProfiler(layer_plan, args.iters)
    backward_profiler = InlineBackwardProfiler(
        len(profiler.descriptors), args.iters
    )
    fn = make_profiled_stack(layers, cp_cache, profiler, backward_profiler)
    grad_buffer = benchmark_parameters(layers, world_size)
    synchronize_parameters(grad_buffer.params)

    # CP ranks in one source DP replica start from the same local microbatch.
    # ``transmition_cp_size`` subsequently remeshes both sequence and batch
    # through its P2P autograd functions.
    microbatch_size = args.global_batch_size // first_dp
    x_full = torch.empty(
        args.seq_length,
        microbatch_size,
        args.hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    torch.cuda.manual_seed(args.layer_seed + 1)
    if dist.get_rank(cp_cache[first_cp][0]) == 0:
        x_full.normal_()
    dist.broadcast(
        x_full,
        src=cp_cache[first_cp][1][0],
        group=cp_cache[first_cp][0],
    )
    x = partition_for_cp(x_full, cp_cache[first_cp][0]) if first_cp > 1 else x_full
    x = x.detach().requires_grad_(True)

    schedule, layout_plan, transitions = describe_plan(layer_plan)
    label = f"random_{mode}"
    if rank == 0:
        print(f"\n  {label}: schedule={schedule}")
        print(
            f"  layout_plan={layout_plan} | transitions={transitions} | "
            f"starting local microbatch={microbatch_size}"
        )
        print(
            f"  {'phase':<20} {'ms':>9} {'tokens/s':>14} {'alloc GiB':>10}"
        )

    try:
        stats = bench_stack(
            label,
            fn,
            x,
            args,
            rank,
            world_size,
            grad_buffer,
            profiler,
            backward_profiler,
        )
    finally:
        grad_buffer.close()
    del grad_buffer
    del backward_profiler
    del profiler, fn, layers, x, x_full
    torch.cuda.empty_cache()
    return stats


def main():
    args = parse_args()
    for key, value in _ATTN_BACKEND_ENV[args.attention_backend].items():
        os.environ[key] = value

    setup_dist()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    for i in range(world_size):
        tp_group = dist.new_group([i])
        if rank == i:
            local_tp_group = tp_group
    layer_types = make_layer_types(
        args.num_attention_layers, args.num_gdn_layers, args.layer_seed
    )

    if rank == 0:
        print(
            f"Randomized layer multiset: attention={args.num_attention_layers}, "
            f"gdn={args.num_gdn_layers}, seed={args.layer_seed}"
        )

    modes = [args.mode] if args.mode != "both" else ["fixed", "mixed"]
    if args.temporal_strategy == "none":
        results = {}
        for mode in modes:
            results[mode] = run_randomized_plan(
                args, mode, layer_types, rank, world_size, local_tp_group
            )
            dist.barrier()

        if rank == 0 and args.mode == "both":
            fixed, mixed = results["fixed"], results["mixed"]
            print(
                "\n  Fixed vs mixed comparison (inline forward, derived backward, "
                "measured E2E):"
            )
            for phase in ("fwd", "bwd", "both"):
                speedup = fixed[phase] / mixed[phase]
                print(
                    f"  {phase:<5} fixed={fixed[phase]:.2f} ms | "
                    f"mixed={mixed[phase]:.2f} ms | speedup={speedup:.3f}x"
                )
            print(
                f"  both peak allocated: fixed={fixed['mem_both']:.2f} GiB | "
                f"mixed={mixed['mem_both']:.2f} GiB | "
                f"delta={mixed['mem_both'] - fixed['mem_both']:+.2f} GiB"
            )
            print(
                f"  both peak reserved:  fixed={fixed['mem_both_reserved']:.2f} GiB | "
                f"mixed={mixed['mem_both_reserved']:.2f} GiB | "
                f"delta={mixed['mem_both_reserved'] - fixed['mem_both_reserved']:+.2f} GiB"
            )
    else:
        strategies = (
            [args.temporal_strategy]
            if args.temporal_strategy != "both"
            else ["bundle", "repeat"]
        )
        temporal_results = {}
        for strategy in strategies:
            temporal_results[strategy] = {}
            for mode in modes:
                temporal_results[strategy][mode] = run_temporal_plan(
                    args,
                    mode,
                    strategy,
                    layer_types,
                    rank,
                    world_size,
                    local_tp_group,
                )
                dist.barrier()

            if rank == 0 and args.mode == "both":
                fixed = temporal_results[strategy]["fixed"]
                mixed = temporal_results[strategy]["mixed"]
                print(f"\n  Temporal {strategy}: fixed vs mixed")
                print(
                    f"  optimizer-step fixed={fixed['both']:.2f} ms | "
                    f"mixed={mixed['both']:.2f} ms | "
                    f"speedup={fixed['both'] / mixed['both']:.3f}x"
                )
                print(
                    f"  peak allocated fixed={fixed['mem_both']:.2f} GiB | "
                    f"mixed={mixed['mem_both']:.2f} GiB | "
                    f"delta={mixed['mem_both'] - fixed['mem_both']:+.2f} GiB"
                )

        if rank == 0 and args.temporal_strategy == "both":
            selected_mode = "mixed" if "mixed" in modes else modes[0]
            bundle = temporal_results["bundle"][selected_mode]
            repeat = temporal_results["repeat"][selected_mode]
            print(f"\n  Bundle vs repeat ({selected_mode} plan)")
            print(
                f"  bundle={bundle['both']:.2f} ms | repeat={repeat['both']:.2f} ms | "
                f"repeat/bundle latency={repeat['both'] / bundle['both']:.3f}x"
            )
            print(
                f"  bundle unique_tokens/s={bundle['unique_tokens_s']:.0f} | "
                f"repeat unique_tokens/s={repeat['unique_tokens_s']:.0f}"
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
