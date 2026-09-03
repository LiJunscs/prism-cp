"""Cohort-aware TMF model and bidirectional pipeline communication adapter."""

from dataclasses import dataclass

import torch
import torch.distributed as dist

from plan import local_payload_shape, temporal_width
from utils import (
    AsyncLaneGatherTransition,
    build_attn_te,
    build_gdn_te,
    partition_for_cp,
    transmition_cp_size,
)


@dataclass
class TemporalPayload:
    tensor: torch.Tensor
    logical_ids: tuple

    @property
    def width(self):
        return len(self.logical_ids)


@dataclass
class CohortState:
    cohort_id: int
    logical_ids: tuple
    inputs: tuple
    outputs: tuple


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


def _register_backward_trace(
    tensor,
    trace,
    event,
    cohort_id,
    *,
    callback=None,
    **fields,
):
    if (not trace.enabled and callback is None) or not tensor.requires_grad:
        return

    def hook(gradient):
        if trace.enabled:
            trace.emit(
                event,
                cohort_id,
                gradient_shape=tuple(gradient.shape),
                **fields,
            )
        if callback is not None:
            callback(gradient)
        return gradient

    tensor.register_hook(hook)


class StageModelChunk:
    """Modules belonging to one PP stage, grouped by consecutive CP layout."""

    def __init__(self, args, stage, layer_plan, cp_cache, tp_group, trace):
        self.args = args
        self.stage = stage
        self.layer_plan = layer_plan
        self.cp_cache = cp_cache
        self.trace = trace
        self.layers = []
        self.blocks = []

        for layer_index, layer in enumerate(layer_plan):
            cp_group = cp_cache[layer.cp_size][0]
            if layer.layer_type == "attention":
                layer_fn = build_attn_te(
                    args.hidden_size,
                    args.num_attention_heads,
                    args.num_query_groups,
                    layer.cp_size,
                    tp_group,
                    cp_group,
                )
            else:
                layer_fn = build_gdn_te(
                    args.hidden_size,
                    args.num_attention_heads,
                    args.num_query_groups,
                    layer.cp_size,
                    tp_group,
                    cp_group,
                    args,
                )
            self.layers.append(layer_fn)
            if not self.blocks or self.blocks[-1][0] != layer.cp_size:
                self.blocks.append((layer.cp_size, []))
            self.blocks[-1][1].append((layer_index, layer, layer_fn))

        self.params = collect_parameters(self.layers)


class TMFAdapter:
    """Own temporal remeshing, cohort graph lifetime, and PP P2P operations."""

    def __init__(
        self,
        args,
        execution_plan,
        stage,
        model_chunk,
        cp_cache,
        stage_group,
        stage_ranks,
        trace,
        profiler=None,
    ):
        self.args = args
        self.execution_plan = execution_plan
        self.stage = stage
        self.model_chunk = model_chunk
        self.cp_cache = cp_cache
        self.stage_group = stage_group
        self.stage_ranks = stage_ranks
        self.trace = trace
        self.profiler = profiler
        self.mesh_size = args.context_parallel_size
        self.logical_batch = args.micro_batch_size
        self.cohort_width = execution_plan.cohort_width
        self.input_cp = model_chunk.layer_plan[0].cp_size
        self.output_cp = model_chunk.layer_plan[-1].cp_size
        self.transition_sequence = 0
        self.transition_use_triton = (
            getattr(args, "tmf_transition_kernel", "triton") == "triton"
        )
        self.transition_overlap = bool(
            getattr(args, "tmf_transition_overlap", True)
        )
        self.transition_lane_group_size = int(
            getattr(args, "tmf_transition_lane_group_size", 1)
        )
        if self.transition_lane_group_size < 1:
            raise ValueError(
                "tmf_transition_lane_group_size must be positive, got "
                f"{self.transition_lane_group_size}."
            )
        self.transition_stream = (
            torch.cuda.Stream() if self.transition_overlap else None
        )

    def _profile_range(self, name, **fields):
        if self.profiler is None:
            from contextlib import nullcontext

            return nullcontext()
        return self.profiler.record(name, **fields)

    def _register_transition_backward(
        self,
        inputs,
        output,
        cohort_id,
        source_cp,
        destination_cp,
        logical_ids,
    ):
        transition_id = self.transition_sequence
        self.transition_sequence += 1
        operation = {"id": None, "remaining": len(inputs)}
        profile_fields = {
            "cohort": cohort_id,
            "transition_id": transition_id,
            "source_cp": source_cp,
            "destination_cp": destination_cp,
            "logical_ids": list(logical_ids),
        }

        def begin(_gradient):
            if self.profiler is not None:
                operation["id"] = self.profiler.begin(
                    "tmf_transition_backward", **profile_fields
                )

        def input_ready(input_ids):
            def callback(_gradient):
                operation["remaining"] -= 1
                if operation["remaining"] == 0:
                    self.trace.emit(
                        "tmf_transition_backward_end",
                        cohort_id,
                        transition_id=transition_id,
                        source_cp=source_cp,
                        destination_cp=destination_cp,
                        logical_ids=list(logical_ids),
                    )
                    if self.profiler is not None:
                        self.profiler.end(operation["id"])

            return callback

        _register_backward_trace(
            output,
            self.trace,
            "tmf_transition_backward_begin",
            cohort_id,
            callback=begin,
            transition_id=transition_id,
            source_cp=source_cp,
            destination_cp=destination_cp,
            logical_ids=list(logical_ids),
        )
        for tensor, input_ids in inputs:
            _register_backward_trace(
                tensor,
                self.trace,
                "tmf_transition_backward_input_gradient",
                cohort_id,
                callback=input_ready(input_ids),
                transition_id=transition_id,
                source_cp=source_cp,
                destination_cp=destination_cp,
                logical_ids=list(input_ids),
            )

    def logical_ids(self, cohort_id):
        start = cohort_id * self.cohort_width
        return tuple(range(start, start + self.cohort_width))

    def _expected_shape(self, cp_size, width):
        return local_payload_shape(
            self.args.seq_length,
            self.logical_batch,
            self.args.hidden_size,
            self.mesh_size,
            cp_size,
            width,
        )

    def _validate_payload(self, payload, cp_size, context):
        expected = self._expected_shape(cp_size, payload.width)
        if tuple(payload.tensor.shape) != expected:
            raise RuntimeError(
                f"stage={self.stage} {context} logical_ids={payload.logical_ids} "
                f"shape={tuple(payload.tensor.shape)}, expected CP{cp_size} "
                f"shape={expected}."
            )

    def _pack_groups(self, payloads, target_width):
        groups = []
        current = []
        current_width = 0
        for payload in payloads:
            if payload.width > target_width:
                raise RuntimeError(
                    f"Cannot pack payload width {payload.width} into target width "
                    f"{target_width}."
                )
            if current_width + payload.width > target_width:
                raise RuntimeError(
                    f"Payload widths do not align with target width {target_width}."
                )
            current.append(payload)
            current_width += payload.width
            if current_width == target_width:
                groups.append(current)
                current = []
                current_width = 0
        if current:
            raise RuntimeError(
                f"Incomplete temporal cohort width {current_width}/{target_width}."
            )
        return groups

    def _remesh_payloads(self, payloads, source_cp, destination_cp, cohort_id):
        if source_cp == destination_cp:
            return payloads
        for payload in payloads:
            self._validate_payload(payload, source_cp, "transition input")

        source_dp = self.mesh_size // source_cp
        destination_dp = self.mesh_size // destination_cp
        destination_width = temporal_width(
            self.logical_batch, self.mesh_size, destination_cp
        )
        self.trace.emit(
            "tmf_remesh_begin",
            cohort_id,
            source_cp=source_cp,
            destination_cp=destination_cp,
            input_widths=[payload.width for payload in payloads],
            destination_width=destination_width,
        )

        remeshed = []
        if destination_dp > source_dp:
            for group in self._pack_groups(payloads, destination_width):
                logical_ids = tuple(
                    logical_id
                    for payload in group
                    for logical_id in payload.logical_ids
                )
                packed = torch.cat([payload.tensor for payload in group], dim=1)
                with self._profile_range(
                    "tmf_transition_forward",
                    cohort=cohort_id,
                    source_cp=source_cp,
                    destination_cp=destination_cp,
                    logical_ids=list(logical_ids),
                ):
                    transitioned = transmition_cp_size(
                        packed,
                        source_cp,
                        destination_cp,
                        self.cp_cache,
                        strategy="redistribute",
                        use_triton=self.transition_use_triton,
                    )
                self._register_transition_backward(
                    [(payload.tensor, payload.logical_ids) for payload in group],
                    transitioned,
                    cohort_id,
                    source_cp,
                    destination_cp,
                    logical_ids,
                )
                remeshed.append(TemporalPayload(transitioned, logical_ids))
        else:
            target_batch = self._expected_shape(destination_cp, destination_width)[1]
            for payload in payloads:
                if payload.width % destination_width:
                    raise RuntimeError(
                        f"Payload width {payload.width} cannot split into destination "
                        f"width {destination_width}."
                    )
                with self._profile_range(
                    "tmf_transition_forward",
                    cohort=cohort_id,
                    source_cp=source_cp,
                    destination_cp=destination_cp,
                    logical_ids=list(payload.logical_ids),
                ):
                    transitioned = transmition_cp_size(
                        payload.tensor,
                        source_cp,
                        destination_cp,
                        self.cp_cache,
                        strategy="redistribute",
                        use_triton=self.transition_use_triton,
                    )
                self._register_transition_backward(
                    [(payload.tensor, payload.logical_ids)],
                    transitioned,
                    cohort_id,
                    source_cp,
                    destination_cp,
                    payload.logical_ids,
                )
                chunks = transitioned.split(target_batch, dim=1)
                id_chunks = tuple(
                    payload.logical_ids[offset : offset + destination_width]
                    for offset in range(0, payload.width, destination_width)
                )
                if len(chunks) != len(id_chunks):
                    raise RuntimeError(
                        f"Transition CP{source_cp}->CP{destination_cp} produced "
                        f"{len(chunks)} chunks for {len(id_chunks)} identity groups."
                    )
                remeshed.extend(
                    TemporalPayload(chunk.contiguous(), tuple(logical_ids))
                    for chunk, logical_ids in zip(chunks, id_chunks)
                )

        for payload in remeshed:
            self._validate_payload(payload, destination_cp, "transition output")
        self.trace.emit(
            "tmf_remesh_end",
            cohort_id,
            source_cp=source_cp,
            destination_cp=destination_cp,
            output_widths=[payload.width for payload in remeshed],
            output_ids=[list(payload.logical_ids) for payload in remeshed],
        )
        return remeshed

    def forward_cohort(self, cohort_id, inputs):
        logical_ids = self.logical_ids(cohort_id)
        if len(inputs) != self.cohort_width:
            raise RuntimeError(
                f"Cohort {cohort_id} has {len(inputs)} inputs; expected "
                f"K={self.cohort_width}."
            )
        payloads = [
            TemporalPayload(tensor, (logical_id,))
            for tensor, logical_id in zip(inputs, logical_ids)
        ]
        for payload in payloads:
            self._validate_payload(payload, self.input_cp, "cohort input")
        self.trace.emit(
            "cohort_forward_begin",
            cohort_id,
            logical_ids=list(logical_ids),
            cohort_width=self.cohort_width,
            input_cp=self.input_cp,
        )

        current_cp = self.input_cp
        blocks = self.model_chunk.blocks
        for block_index, (cp_size, block_layers) in enumerate(blocks):
            if cp_size != current_cp:
                raise RuntimeError(
                    f"Block {block_index} expects CP{cp_size}, but payloads are "
                    f"in CP{current_cp}."
                )

            next_cp = blocks[block_index + 1][0] if block_index + 1 < len(blocks) else None
            async_contract = bool(
                self.transition_overlap
                and next_cp is not None
                and cp_size > next_cp
            )
            async_groups = []
            async_routes = []
            transition_profile_id = None
            if async_contract:
                destination_width = temporal_width(
                    self.logical_batch,
                    self.mesh_size,
                    next_cp,
                )
                groups = self._pack_groups(payloads, destination_width)
                multiple = cp_size // next_cp
                for group_index, group in enumerate(groups):
                    if len(group) != multiple:
                        raise RuntimeError(
                            f"Async CP{cp_size}->CP{next_cp} transition needs "
                            f"{multiple} equal-width lanes, got {len(group)}."
                        )
                    if len({payload.width for payload in group}) != 1:
                        raise RuntimeError(
                            "Async transition lanes must have equal temporal widths."
                        )
                    transition = AsyncLaneGatherTransition(
                        cp_size,
                        next_cp,
                        self.cp_cache,
                        self.transition_stream,
                        use_triton=self.transition_use_triton,
                        lane_group_size=self.transition_lane_group_size,
                    )
                    logical_group_ids = tuple(
                        logical_id
                        for payload in group
                        for logical_id in payload.logical_ids
                    )
                    async_groups.append(
                        {
                            "transition": transition,
                            "logical_ids": logical_group_ids,
                            "inputs": [],
                        }
                    )
                    async_routes.extend(
                        (group_index, lane_index)
                        for lane_index in range(len(group))
                    )
                if len(async_routes) != len(payloads):
                    raise RuntimeError("Async transition routing lost a payload lane.")
                self.trace.emit(
                    "tmf_remesh_begin",
                    cohort_id,
                    source_cp=cp_size,
                    destination_cp=next_cp,
                    input_widths=[payload.width for payload in payloads],
                    destination_width=destination_width,
                    overlap_mode="lane_async",
                    lane_group_size=self.transition_lane_group_size,
                    kernel_backend=(
                        "triton" if self.transition_use_triton else "torch"
                    ),
                )

            block_outputs = []
            for payload_index, payload in enumerate(payloads):
                tensor = payload.tensor
                for layer_index, layer, layer_fn in block_layers:
                    backward_operation = {"id": None}
                    profile_fields = {
                        "cohort": cohort_id,
                        "block": block_index,
                        "layer": layer_index,
                        "layer_type": layer.layer_type,
                        "cp_size": layer.cp_size,
                        "logical_ids": list(payload.logical_ids),
                    }

                    def backward_end(_gradient, operation=backward_operation):
                        if self.profiler is not None:
                            self.profiler.end(operation["id"])

                    _register_backward_trace(
                        tensor,
                        self.trace,
                        "layer_backward_end",
                        cohort_id,
                        block=block_index,
                        layer=layer_index,
                        layer_type=layer.layer_type,
                        cp_size=layer.cp_size,
                        logical_ids=list(payload.logical_ids),
                        callback=backward_end,
                    )
                    self.trace.emit(
                        "layer_forward_begin",
                        cohort_id,
                        block=block_index,
                        layer=layer_index,
                        layer_type=layer.layer_type,
                        cp_size=layer.cp_size,
                        logical_ids=list(payload.logical_ids),
                        shape=tuple(tensor.shape),
                    )
                    with self._profile_range("layer_forward", **profile_fields):
                        tensor = layer_fn(tensor)
                    self.trace.emit(
                        "layer_forward_end",
                        cohort_id,
                        block=block_index,
                        layer=layer_index,
                        layer_type=layer.layer_type,
                        cp_size=layer.cp_size,
                        logical_ids=list(payload.logical_ids),
                        shape=tuple(tensor.shape),
                    )
                    def backward_begin(
                        _gradient,
                        operation=backward_operation,
                        fields=profile_fields,
                    ):
                        if self.profiler is not None:
                            operation["id"] = self.profiler.begin(
                                "layer_backward", **fields
                            )

                    _register_backward_trace(
                        tensor,
                        self.trace,
                        "layer_backward_begin",
                        cohort_id,
                        block=block_index,
                        layer=layer_index,
                        layer_type=layer.layer_type,
                        cp_size=layer.cp_size,
                        logical_ids=list(payload.logical_ids),
                        callback=backward_begin,
                    )
                output_payload = TemporalPayload(tensor, payload.logical_ids)
                block_outputs.append(output_payload)
                if async_contract:
                    if transition_profile_id is None and self.profiler is not None:
                        transition_profile_id = self.profiler.begin(
                            "tmf_transition_forward",
                            cohort=cohort_id,
                            source_cp=cp_size,
                            destination_cp=next_cp,
                            overlap_mode="lane_async",
                            lane_group_size=self.transition_lane_group_size,
                            kernel_backend=(
                                "triton" if self.transition_use_triton else "torch"
                            ),
                        )
                    group_index, expected_lane = async_routes[payload_index]
                    group_state = async_groups[group_index]
                    communication_group = (
                        expected_lane // self.transition_lane_group_size
                    )
                    self.trace.emit(
                        "tmf_transition_lane_enqueue_begin",
                        cohort_id,
                        source_cp=cp_size,
                        destination_cp=next_cp,
                        transition_group=group_index,
                        lane=expected_lane,
                        lane_group_size=self.transition_lane_group_size,
                        communication_group=communication_group,
                        logical_ids=list(payload.logical_ids),
                    )
                    actual_lane = group_state["transition"].enqueue(tensor)
                    if actual_lane != expected_lane:
                        raise RuntimeError(
                            f"Async transition enqueued lane {actual_lane}, "
                            f"expected {expected_lane}."
                        )
                    group_state["inputs"].append(
                        (tensor, payload.logical_ids)
                    )
                    self.trace.emit(
                        "tmf_transition_lane_enqueue_end",
                        cohort_id,
                        source_cp=cp_size,
                        destination_cp=next_cp,
                        transition_group=group_index,
                        lane=actual_lane,
                        lane_group_size=self.transition_lane_group_size,
                        communication_group=communication_group,
                        communication_enqueued=(
                            (actual_lane + 1)
                            % self.transition_lane_group_size
                            == 0
                        ),
                        logical_ids=list(payload.logical_ids),
                    )

            if async_contract:
                remeshed = []
                for group_index, group_state in enumerate(async_groups):
                    logical_group_ids = group_state["logical_ids"]
                    with self._profile_range(
                        "tmf_transition_consumer_wait",
                        cohort=cohort_id,
                        source_cp=cp_size,
                        destination_cp=next_cp,
                        transition_group=group_index,
                        lane_group_size=self.transition_lane_group_size,
                        logical_ids=list(logical_group_ids),
                    ):
                        transitioned = group_state["transition"].finish()
                    self._register_transition_backward(
                        group_state["inputs"],
                        transitioned,
                        cohort_id,
                        cp_size,
                        next_cp,
                        logical_group_ids,
                    )
                    remeshed.append(
                        TemporalPayload(transitioned, logical_group_ids)
                    )
                    self.trace.emit(
                        "tmf_transition_lane_group_ready",
                        cohort_id,
                        source_cp=cp_size,
                        destination_cp=next_cp,
                        transition_group=group_index,
                        logical_ids=list(logical_group_ids),
                    )
                if self.profiler is not None:
                    self.profiler.end(transition_profile_id)
                for payload in remeshed:
                    self._validate_payload(payload, next_cp, "transition output")
                self.trace.emit(
                    "tmf_remesh_end",
                    cohort_id,
                    source_cp=cp_size,
                    destination_cp=next_cp,
                    output_widths=[payload.width for payload in remeshed],
                    output_ids=[list(payload.logical_ids) for payload in remeshed],
                    overlap_mode="lane_async",
                    lane_group_size=self.transition_lane_group_size,
                )
                payloads = remeshed
                current_cp = next_cp
            else:
                payloads = block_outputs
                if next_cp is not None and next_cp != cp_size:
                    payloads = self._remesh_payloads(
                        payloads,
                        cp_size,
                        next_cp,
                        cohort_id,
                    )
                    current_cp = next_cp

        flattened_ids = tuple(
            logical_id for payload in payloads for logical_id in payload.logical_ids
        )
        if flattened_ids != logical_ids or any(payload.width != 1 for payload in payloads):
            raise RuntimeError(
                f"Stage {self.stage} must return width-1 payloads in identity order; "
                f"got widths={[payload.width for payload in payloads]} and "
                f"ids={flattened_ids}."
            )
        outputs = tuple(payload.tensor for payload in payloads)
        self.trace.emit(
            "cohort_forward_end",
            cohort_id,
            logical_ids=list(logical_ids),
            output_cp=self.output_cp,
            output_shapes=[list(tensor.shape) for tensor in outputs],
        )
        return CohortState(cohort_id, logical_ids, tuple(inputs), outputs)

    def backward_cohort(
        self,
        state,
        output_gradients,
        *,
        last_stage,
        loss_denominator,
    ):
        if not last_stage and len(output_gradients) != self.cohort_width:
            raise RuntimeError(
                f"Cohort {state.cohort_id} has {len(output_gradients)} output "
                f"gradients; expected K={self.cohort_width}."
            )
        for tensor in state.inputs:
            if tensor.grad is not None:
                tensor.grad = None
        self.trace.emit(
            "cohort_backward_begin",
            state.cohort_id,
            logical_ids=list(state.logical_ids),
            last_stage=last_stage,
        )
        with self._profile_range(
            "cohort_backward",
            cohort=state.cohort_id,
            logical_ids=list(state.logical_ids),
            last_stage=last_stage,
        ):
            if last_stage:
                loss = sum(output.float().sum() for output in state.outputs)
                loss = loss / loss_denominator
                loss.backward()
                loss_value = loss.detach()
            else:
                torch.autograd.backward(state.outputs, tuple(output_gradients))
                loss_value = None
        input_gradients = tuple(tensor.grad for tensor in state.inputs)
        if any(gradient is None for gradient in input_gradients):
            missing = [
                logical_id
                for logical_id, gradient in zip(state.logical_ids, input_gradients)
                if gradient is None
            ]
            raise RuntimeError(
                f"Cohort {state.cohort_id} did not produce input gradients for "
                f"logical microbatches {missing}."
            )
        if getattr(self.args, "check_input_grad_finite", True):
            for logical_id, gradient in zip(state.logical_ids, input_gradients):
                if not torch.isfinite(gradient).all():
                    raise RuntimeError(
                        f"Cohort {state.cohort_id} input gradient for logical "
                        f"microbatch {logical_id} is non-finite."
                    )
        self.trace.emit(
            "cohort_backward_end",
            state.cohort_id,
            logical_ids=list(state.logical_ids),
            input_gradient_shapes=[list(gradient.shape) for gradient in input_gradients],
        )
        return input_gradients, loss_value

    def make_stage0_cohort(self, cohort_id):
        inputs = []
        stage_root = self.stage_ranks[0]
        first_cp = self.input_cp
        dp_size = self.mesh_size // first_cp
        local_batch = self.logical_batch // dp_size
        stage_local_rank = self.stage_ranks.index(dist.get_rank())
        dp_rank = stage_local_rank // first_cp
        for logical_id in self.logical_ids(cohort_id):
            full = torch.empty(
                self.args.seq_length,
                self.logical_batch,
                self.args.hidden_size,
                device="cuda",
                dtype=torch.bfloat16,
            )
            if dist.get_rank() == stage_root:
                generator = torch.Generator(device="cuda")
                generator.manual_seed(1234 + logical_id)
                full.normal_(generator=generator)
            self.trace.emit(
                "stage0_input_broadcast_begin",
                cohort_id,
                logical_id=logical_id,
                source=stage_root,
                shape=tuple(full.shape),
            )
            dist.broadcast(full, src=stage_root, group=self.stage_group)
            self.trace.emit(
                "stage0_input_broadcast_end",
                cohort_id,
                logical_id=logical_id,
                source=stage_root,
            )
            batch_start = dp_rank * local_batch
            local_full = full[
                :, batch_start : batch_start + local_batch
            ].contiguous()
            cp_group = self.cp_cache[first_cp][0]
            local = (
                partition_for_cp(local_full, cp_group)
                if first_cp > 1
                else local_full
            )
            expected = self._expected_shape(first_cp, 1)
            if tuple(local.shape) != expected:
                raise RuntimeError(
                    f"Stage-0 logical microbatch {logical_id} shape "
                    f"{tuple(local.shape)} does not match {expected}."
                )
            inputs.append(local.detach().requires_grad_(True))
        self.trace.emit(
            "stage0_cohort_ready",
            cohort_id,
            logical_ids=list(self.logical_ids(cohort_id)),
        )
        return tuple(inputs)

    def _validate_wire_tensors(self, tensors, boundary, context):
        if len(tensors) != self.cohort_width:
            raise RuntimeError(
                f"{context} has {len(tensors)} tensors; expected K={self.cohort_width}."
            )
        for index, tensor in enumerate(tensors):
            if tuple(tensor.shape) != boundary.wire_item_shape:
                raise RuntimeError(
                    f"boundary={boundary.boundary_id} {context}[{index}] shape "
                    f"{tuple(tensor.shape)}; expected {boundary.wire_item_shape}."
                )

    def _wait_ops(self, ops, event, cohort_id, **fields):
        self.trace.emit(event + "_begin", cohort_id, **fields)
        with self._profile_range(
            "pp_p2p",
            cohort=cohort_id,
            p2p_kind=event,
            **fields,
        ):
            for work in dist.batch_isend_irecv(ops):
                work.wait()
        self.trace.emit(event + "_end", cohort_id, **fields)

    def send_forward(self, state, next_rank, boundary):
        self._validate_wire_tensors(state.outputs, boundary, "forward send")
        ops = [
            dist.P2POp(dist.isend, tensor.detach().contiguous(), next_rank)
            for tensor in state.outputs
        ]
        self._wait_ops(
            ops,
            "cohort_p2p_forward_send",
            state.cohort_id,
            peer=next_rank,
            boundary=boundary.boundary_id,
            logical_ids=list(state.logical_ids),
        )

    def recv_forward(self, cohort_id, previous_rank, boundary):
        tensors = tuple(
            torch.empty(
                boundary.wire_item_shape,
                device="cuda",
                dtype=torch.bfloat16,
            )
            for _ in range(self.cohort_width)
        )
        ops = [dist.P2POp(dist.irecv, tensor, previous_rank) for tensor in tensors]
        self._wait_ops(
            ops,
            "cohort_p2p_forward_recv",
            cohort_id,
            peer=previous_rank,
            boundary=boundary.boundary_id,
            logical_ids=list(self.logical_ids(cohort_id)),
        )
        return tuple(tensor.detach().requires_grad_(True) for tensor in tensors)

    def send_backward(self, state, input_gradients, previous_rank, boundary):
        self._validate_wire_tensors(input_gradients, boundary, "backward send")
        ops = [
            dist.P2POp(dist.isend, gradient.detach().contiguous(), previous_rank)
            for gradient in input_gradients
        ]
        self._wait_ops(
            ops,
            "cohort_p2p_backward_send",
            state.cohort_id,
            peer=previous_rank,
            boundary=boundary.boundary_id,
            logical_ids=list(state.logical_ids),
        )

    def recv_backward(self, state, next_rank, boundary):
        gradients = tuple(torch.empty_like(output) for output in state.outputs)
        ops = [dist.P2POp(dist.irecv, gradient, next_rank) for gradient in gradients]
        self._wait_ops(
            ops,
            "cohort_p2p_backward_recv",
            state.cohort_id,
            peer=next_rank,
            boundary=boundary.boundary_id,
            logical_ids=list(state.logical_ids),
        )
        return gradients

    def send_forward_recv_backward(
        self,
        forward_state,
        backward_state,
        next_rank,
        boundary,
    ):
        self._validate_wire_tensors(
            forward_state.outputs, boundary, "duplex forward send"
        )
        gradients = tuple(torch.empty_like(output) for output in backward_state.outputs)
        ops = []
        for output, gradient in zip(forward_state.outputs, gradients):
            ops.append(
                dist.P2POp(dist.isend, output.detach().contiguous(), next_rank)
            )
            ops.append(dist.P2POp(dist.irecv, gradient, next_rank))
        self._wait_ops(
            ops,
            "cohort_p2p_forward_send_backward_recv",
            forward_state.cohort_id,
            backward_cohort=backward_state.cohort_id,
            peer=next_rank,
            boundary=boundary.boundary_id,
            forward_ids=list(forward_state.logical_ids),
            backward_ids=list(backward_state.logical_ids),
        )
        return gradients

    def send_backward_recv_forward(
        self,
        backward_state,
        input_gradients,
        next_cohort_id,
        previous_rank,
        boundary,
    ):
        self._validate_wire_tensors(
            input_gradients, boundary, "duplex backward send"
        )
        next_inputs = tuple(
            torch.empty(
                boundary.wire_item_shape,
                device="cuda",
                dtype=torch.bfloat16,
            )
            for _ in range(self.cohort_width)
        )
        ops = []
        for gradient, next_input in zip(input_gradients, next_inputs):
            ops.append(
                dist.P2POp(
                    dist.isend,
                    gradient.detach().contiguous(),
                    previous_rank,
                )
            )
            ops.append(dist.P2POp(dist.irecv, next_input, previous_rank))
        self._wait_ops(
            ops,
            "cohort_p2p_backward_send_forward_recv",
            backward_state.cohort_id,
            forward_cohort=next_cohort_id,
            peer=previous_rank,
            boundary=boundary.boundary_id,
            backward_ids=list(backward_state.logical_ids),
            forward_ids=list(self.logical_ids(next_cohort_id)),
        )
        return tuple(tensor.detach().requires_grad_(True) for tensor in next_inputs)
