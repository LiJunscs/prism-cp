"""Non-interleaved 1F1B scheduler whose unit is one TMF cohort."""

import torch
import torch.distributed as dist

from runtime import reduce_stage_gradients


def expected_cohort_events(stage, pp_size, pipeline_cohorts):
    """Canonical non-interleaved 1F1B order at cohort granularity."""
    warmup = min(pp_size - stage - 1, pipeline_cohorts)
    remaining = pipeline_cohorts - warmup
    events = [("F", cohort_id) for cohort_id in range(warmup)]
    for steady_index in range(remaining):
        events.append(("F", warmup + steady_index))
        events.append(("B", steady_index))
    events.extend(
        ("B", cohort_id)
        for cohort_id in range(remaining, pipeline_cohorts)
    )
    return events


def run_cohort_1f1b(
    args,
    stage,
    adapter,
    stage_group,
    boundaries,
    trace,
    *,
    profiler=None,
    check_parameter_finite=True,
):
    """Run 1F1B while delegating all forward/backward P2P to ``adapter``."""
    rank = dist.get_rank()
    mesh_size = args.context_parallel_size
    stage_local_rank = rank % mesh_size
    pp_size = args.pipeline_parallel_size
    first_stage = stage == 0
    last_stage = stage == pp_size - 1
    previous_rank = (
        None if first_stage else (stage - 1) * mesh_size + stage_local_rank
    )
    next_rank = None if last_stage else (stage + 1) * mesh_size + stage_local_rank
    inbound_boundary = None if first_stage else boundaries[stage - 1]
    outbound_boundary = None if last_stage else boundaries[stage]
    pipeline_cohorts = adapter.execution_plan.pipeline_cohorts
    warmup = min(pp_size - stage - 1, pipeline_cohorts)
    remaining = pipeline_cohorts - warmup
    forward_queue = []
    events = []
    local_loss = torch.zeros((), device="cuda", dtype=torch.float32)
    loss_denominator = (
        args.micro_batches
        * args.seq_length
        * args.micro_batch_size
        * args.hidden_size
    )

    for param in adapter.model_chunk.params:
        param.grad = None

    trace.emit(
        "cohort_schedule_begin",
        pipeline_cohorts=pipeline_cohorts,
        cohort_width=adapter.cohort_width,
        warmup=warmup,
        remaining=remaining,
        previous_rank=previous_rank,
        next_rank=next_rank,
    )

    def receive_or_create(cohort_id):
        if first_stage:
            return adapter.make_stage0_cohort(cohort_id)
        return adapter.recv_forward(
            cohort_id,
            previous_rank,
            inbound_boundary,
        )

    for cohort_id in range(warmup):
        inputs = receive_or_create(cohort_id)
        state = adapter.forward_cohort(cohort_id, inputs)
        forward_queue.append(state)
        events.append(("F", cohort_id))
        if not last_stage:
            adapter.send_forward(state, next_rank, outbound_boundary)

    current_inputs = receive_or_create(warmup) if remaining else None
    current_cohort = warmup

    for steady_index in range(remaining):
        forward_state = adapter.forward_cohort(current_cohort, current_inputs)
        forward_queue.append(forward_state)
        events.append(("F", current_cohort))

        backward_state = forward_queue[0]
        output_gradients = (
            None
            if last_stage
            else adapter.send_forward_recv_backward(
                forward_state,
                backward_state,
                next_rank,
                outbound_boundary,
            )
        )
        popped_state = forward_queue.pop(0)
        if popped_state is not backward_state:
            raise RuntimeError("The cohort forward queue changed during P2P exchange.")
        input_gradients, loss_value = adapter.backward_cohort(
            backward_state,
            output_gradients,
            last_stage=last_stage,
            loss_denominator=loss_denominator,
        )
        events.append(("B", backward_state.cohort_id))
        if loss_value is not None:
            local_loss.add_(loss_value)

        final_steady = steady_index == remaining - 1
        if final_steady:
            if not first_stage:
                adapter.send_backward(
                    backward_state,
                    input_gradients,
                    previous_rank,
                    inbound_boundary,
                )
            current_inputs = None
        elif first_stage:
            current_cohort += 1
            current_inputs = receive_or_create(current_cohort)
        else:
            current_cohort += 1
            current_inputs = adapter.send_backward_recv_forward(
                backward_state,
                input_gradients,
                current_cohort,
                previous_rank,
                inbound_boundary,
            )

    while forward_queue:
        backward_state = forward_queue.pop(0)
        output_gradients = (
            None
            if last_stage
            else adapter.recv_backward(
                backward_state,
                next_rank,
                outbound_boundary,
            )
        )
        input_gradients, loss_value = adapter.backward_cohort(
            backward_state,
            output_gradients,
            last_stage=last_stage,
            loss_denominator=loss_denominator,
        )
        events.append(("B", backward_state.cohort_id))
        if loss_value is not None:
            local_loss.add_(loss_value)
        if not first_stage:
            adapter.send_backward(
                backward_state,
                input_gradients,
                previous_rank,
                inbound_boundary,
            )

    expected = expected_cohort_events(stage, pp_size, pipeline_cohorts)
    if events != expected:
        raise RuntimeError(
            f"Stage {stage} cohort schedule differs from canonical 1F1B: "
            f"actual={events}, expected={expected}."
        )
    reduce_stage_gradients(
        adapter.model_chunk.params,
        stage_group,
        trace,
        check_finite=check_parameter_finite,
        profiler=profiler,
    )
    trace.emit("cohort_schedule_end", events=events)
    return events, local_loss
