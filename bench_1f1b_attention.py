"""Minimal fixed-CP, Attention-only, non-interleaved 1F1B prototype.

The prototype intentionally excludes heterogeneous CP, GDN, DP, optimizer,
and recompute.  It establishes the pipeline schedule and the activation/gradient
P2P boundary that a later TMF integration can reuse.
"""

import os
import time

import torch
import torch.distributed as dist

from utils import (
    _ATTN_BACKEND_ENV,
    build_attn_te,
    parse_benchmark_args,
    partition_for_cp,
    setup_dist,
)


def add_1f1b_args(parser):
    parser.add_argument("--pipeline-parallel-size", type=int, default=2)
    parser.add_argument("--context-parallel-size", type=int, default=1)
    parser.add_argument("--micro-batches", type=int, default=4)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--layers-per-stage", type=int, default=2)
    parser.add_argument("--trace-schedule", action="store_true")
    return parser


def parse_args(argv=None):
    return parse_benchmark_args(
        add_1f1b_args,
        argv=argv,
        description=__doc__,
    )


def make_groups(pp_size, cp_size):
    """Create all groups in globally identical order and return this rank's groups."""
    rank = dist.get_rank()
    stage_cp_group = None
    stage_cp_ranks = None
    for stage in range(pp_size):
        ranks = list(range(stage * cp_size, (stage + 1) * cp_size))
        group = dist.new_group(ranks)
        if rank in ranks:
            stage_cp_group = group
            stage_cp_ranks = ranks

    pipeline_group = None
    for cp_rank in range(cp_size):
        ranks = [stage * cp_size + cp_rank for stage in range(pp_size)]
        group = dist.new_group(ranks)
        if rank in ranks:
            pipeline_group = group

    tp_group = None
    for global_rank in range(pp_size * cp_size):
        group = dist.new_group([global_rank])
        if rank == global_rank:
            tp_group = group
    return stage_cp_group, stage_cp_ranks, pipeline_group, tp_group


def collect_parameters(layer_fns):
    params = []
    seen = set()
    for fn in layer_fns:
        for module in getattr(fn, "_benchmark_modules", ()):
            for param in module.parameters():
                if param.requires_grad and id(param) not in seen:
                    seen.add(id(param))
                    params.append(param)
    return params


def build_stage(args, cp_group, tp_group):
    layers = [
        build_attn_te(
            args.hidden_size,
            args.num_attention_heads,
            args.num_query_groups,
            args.context_parallel_size,
            tp_group,
            cp_group,
        )
        for _ in range(args.layers_per_stage)
    ]

    def forward(x):
        for layer in layers:
            x = layer(x)
        return x

    return forward, collect_parameters(layers)


def synchronize_stage_parameters(params, cp_group, stage_root):
    for param in params:
        dist.broadcast(param.data, src=stage_root, group=cp_group)


def wait_ops(ops):
    if ops:
        for work in dist.batch_isend_irecv(ops):
            work.wait()


def send_forward(tensor, next_rank):
    dist.send(tensor.detach().contiguous(), dst=next_rank)


def recv_forward(shape, previous_rank):
    tensor = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
    dist.recv(tensor, src=previous_rank)
    return tensor.detach().requires_grad_(True)


def send_backward(tensor, previous_rank):
    dist.send(tensor.detach().contiguous(), dst=previous_rank)


def recv_backward(reference, next_rank):
    tensor = torch.empty_like(reference)
    dist.recv(tensor, src=next_rank)
    return tensor


def send_forward_recv_backward(output, next_rank):
    send = output.detach().contiguous()
    grad = torch.empty_like(output)
    wait_ops(
        [
            dist.P2POp(dist.isend, send, next_rank),
            dist.P2POp(dist.irecv, grad, next_rank),
        ]
    )
    return grad


def send_backward_recv_forward(input_grad, shape, previous_rank):
    send = input_grad.detach().contiguous()
    next_input = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
    wait_ops(
        [
            dist.P2POp(dist.isend, send, previous_rank),
            dist.P2POp(dist.irecv, next_input, previous_rank),
        ]
    )
    return next_input.detach().requires_grad_(True)


def make_stage0_input(args, cp_group, stage_root, microbatch_id):
    full = torch.empty(
        args.seq_length,
        args.micro_batch_size,
        args.hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    if dist.get_rank() == stage_root:
        generator = torch.Generator(device="cuda")
        generator.manual_seed(1234 + microbatch_id)
        full.normal_(generator=generator)
    dist.broadcast(full, src=stage_root, group=cp_group)
    local = (
        partition_for_cp(full, cp_group)
        if args.context_parallel_size > 1
        else full
    )
    return local.detach().requires_grad_(True)


def backward_microbatch(
    input_tensor,
    output_tensor,
    output_grad,
    is_last_stage,
    loss_denominator,
):
    if is_last_stage:
        loss = output_tensor.float().sum() / loss_denominator
        loss.backward()
        loss_value = loss.detach()
    else:
        torch.autograd.backward(output_tensor, output_grad)
        loss_value = None
    if input_tensor.grad is None:
        raise RuntimeError("pipeline input gradient was not produced")
    return input_tensor.grad, loss_value


def expected_stage_events(stage, pp_size, microbatches):
    """Return the canonical non-interleaved 1F1B event order for one stage."""
    warmup = min(pp_size - stage - 1, microbatches)
    remaining = microbatches - warmup
    events = [("F", microbatch_id) for microbatch_id in range(warmup)]
    for steady_index in range(remaining):
        events.append(("F", warmup + steady_index))
        events.append(("B", steady_index))
    events.extend(("B", microbatch_id) for microbatch_id in range(remaining, microbatches))
    return events


def run_1f1b(args, stage_forward, params, cp_group, stage_cp_ranks):
    rank = dist.get_rank()
    cp_size = args.context_parallel_size
    stage = rank // cp_size
    cp_rank = rank % cp_size
    pp_size = args.pipeline_parallel_size
    first_stage = stage == 0
    last_stage = stage == pp_size - 1
    previous_rank = None if first_stage else (stage - 1) * cp_size + cp_rank
    next_rank = None if last_stage else (stage + 1) * cp_size + cp_rank
    local_shape = (
        args.seq_length // cp_size,
        args.micro_batch_size,
        args.hidden_size,
    )
    stage_root = stage_cp_ranks[0]
    microbatches = args.micro_batches
    warmup = min(pp_size - stage - 1, microbatches)
    remaining = microbatches - warmup
    forward_queue = []
    events = []
    local_loss = torch.zeros((), device="cuda", dtype=torch.float32)
    loss_denominator = (
        microbatches
        * args.seq_length
        * args.micro_batch_size
        * args.hidden_size
    )

    for param in params:
        param.grad = None

    def receive_or_create(microbatch_id):
        if first_stage:
            return make_stage0_input(args, cp_group, stage_root, microbatch_id)
        return recv_forward(local_shape, previous_rank)

    # Warmup: fill the forward pipeline.
    for microbatch_id in range(warmup):
        input_tensor = receive_or_create(microbatch_id)
        output_tensor = stage_forward(input_tensor)
        forward_queue.append((microbatch_id, input_tensor, output_tensor))
        events.append(("F", microbatch_id))
        if not last_stage:
            send_forward(output_tensor, next_rank)

    current_input = receive_or_create(warmup) if remaining else None
    current_microbatch = warmup

    # Steady state: one new forward and one oldest backward per iteration.
    for steady_index in range(remaining):
        output_tensor = stage_forward(current_input)
        forward_queue.append((current_microbatch, current_input, output_tensor))
        events.append(("F", current_microbatch))

        output_grad = (
            None
            if last_stage
            else send_forward_recv_backward(output_tensor, next_rank)
        )
        backward_id, backward_input, backward_output = forward_queue.pop(0)
        input_grad, loss_value = backward_microbatch(
            backward_input,
            backward_output,
            output_grad,
            last_stage,
            loss_denominator,
        )
        events.append(("B", backward_id))
        if loss_value is not None:
            local_loss.add_(loss_value)

        final_steady = steady_index == remaining - 1
        if final_steady:
            if not first_stage:
                send_backward(input_grad, previous_rank)
            current_input = None
        elif first_stage:
            current_microbatch += 1
            current_input = receive_or_create(current_microbatch)
        else:
            current_microbatch += 1
            current_input = send_backward_recv_forward(
                input_grad, local_shape, previous_rank
            )

    # Cooldown: drain the outstanding forward queue.
    while forward_queue:
        backward_id, backward_input, backward_output = forward_queue.pop(0)
        output_grad = None if last_stage else recv_backward(backward_output, next_rank)
        input_grad, loss_value = backward_microbatch(
            backward_input,
            backward_output,
            output_grad,
            last_stage,
            loss_denominator,
        )
        events.append(("B", backward_id))
        if loss_value is not None:
            local_loss.add_(loss_value)
        if not first_stage:
            send_backward(input_grad, previous_rank)

    expected_events = expected_stage_events(stage, pp_size, microbatches)
    if events != expected_events:
        raise RuntimeError(
            f"stage {stage} schedule differs from canonical 1F1B: "
            f"actual={events}, expected={expected_events}"
        )

    # CP ranks hold different sequence contributions but replicated parameters.
    for param in params:
        if param.grad is None:
            raise RuntimeError("stage parameter gradient was not produced")
        dist.all_reduce(param.grad, group=cp_group)
        if not torch.isfinite(param.grad).all():
            raise RuntimeError("non-finite stage parameter gradient")
    return events, local_loss


def main():
    args = parse_args()
    for key, value in _ATTN_BACKEND_ENV[args.attention_backend].items():
        os.environ[key] = value
    if args.pipeline_parallel_size < 1 or args.context_parallel_size < 1:
        raise ValueError("PP and CP sizes must be positive")
    if args.micro_batches < 1 or args.layers_per_stage < 1:
        raise ValueError("micro-batches and layers-per-stage must be positive")
    if args.seq_length % (2 * args.context_parallel_size):
        raise ValueError("seq_length must be divisible by 2*context_parallel_size")

    setup_dist()
    rank = dist.get_rank()
    expected_world = args.pipeline_parallel_size * args.context_parallel_size
    if dist.get_world_size() != expected_world:
        raise ValueError(
            f"world_size={dist.get_world_size()} must equal PP*CP={expected_world}"
        )
    cp_group, cp_ranks, _pipeline_group, tp_group = make_groups(
        args.pipeline_parallel_size, args.context_parallel_size
    )
    stage = rank // args.context_parallel_size
    stage_forward, params = build_stage(args, cp_group, tp_group)
    synchronize_stage_parameters(params, cp_group, cp_ranks[0])

    dist.barrier()
    torch.cuda.synchronize()
    start = time.perf_counter()
    events, local_loss = run_1f1b(args, stage_forward, params, cp_group, cp_ranks)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    elapsed = torch.tensor(elapsed_ms, device="cuda")
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)

    loss = local_loss.clone()
    dist.all_reduce(loss)
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite pipeline loss")
    if args.trace_schedule:
        print(f"[rank {rank} stage {stage}] events={events}", flush=True)
    if rank == 0:
        ideal_fraction = args.micro_batches / (
            args.micro_batches + args.pipeline_parallel_size - 1
        )
        print(
            f"Attention-only 1F1B PASS | PP={args.pipeline_parallel_size} "
            f"CP={args.context_parallel_size} M={args.micro_batches} "
            f"layers/stage={args.layers_per_stage} | "
            f"worst_rank_step={elapsed.item():.3f} ms | "
            f"ideal_pipeline_utilization={ideal_fraction:.4f} | "
            f"loss={loss.item():.6e}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
