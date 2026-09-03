"""Fused local layout transforms used by TMF CP transitions.

The Triton kernels deliberately contain no communication.  They materialize
contiguous NCCL send buffers and write received shards into the final layout,
so the communication layer does not need chains of ``cat``/``index_copy_``
kernels.  CPU execution and installations without Triton use exact PyTorch
fallbacks, which also serve as the correctness oracle.
"""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised on remote GPU installations.
    triton = None
    tl = None


def triton_transition_available():
    return triton is not None and torch.cuda.is_available()


def _should_use_triton(tensor, requested):
    disabled = os.environ.get("TMF_TRANSITION_TRITON", "1").lower() in {
        "0",
        "false",
        "off",
        "no",
    }
    return bool(requested and not disabled and tensor.is_cuda and triton is not None)


if triton is not None:

    @triton.jit
    def _pack_scatter_kernel(
        source,
        packed,
        total_elements: tl.constexpr,
        sequence: tl.constexpr,
        batch: tl.constexpr,
        hidden: tl.constexpr,
        multiple: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = (
            tl.program_id(0).to(tl.int64) * BLOCK
            + tl.arange(0, BLOCK).to(tl.int64)
        )
        mask = offsets < total_elements
        feature = offsets % hidden
        quotient = offsets // hidden
        batch_index = quotient % batch
        quotient = quotient // batch
        send_sequence = quotient % (sequence // multiple)
        piece = quotient // (sequence // multiple)
        half = sequence // (2 * multiple)
        source_block = tl.where(
            send_sequence < half,
            piece,
            2 * multiple - piece - 1,
        )
        source_in_block = send_sequence % half
        source_sequence = source_block * half + source_in_block
        source_offset = (source_sequence * batch + batch_index) * hidden + feature
        value = tl.load(source + source_offset, mask=mask)
        tl.store(packed + offsets, value, mask=mask)


    @triton.jit
    def _join_scatter_kernel(
        received,
        output,
        total_elements: tl.constexpr,
        sequence: tl.constexpr,
        batch: tl.constexpr,
        hidden: tl.constexpr,
        multiple: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = (
            tl.program_id(0).to(tl.int64) * BLOCK
            + tl.arange(0, BLOCK).to(tl.int64)
        )
        mask = offsets < total_elements
        feature = offsets % hidden
        quotient = offsets // hidden
        output_batch = quotient % (batch * multiple)
        sequence_index = quotient // (batch * multiple)
        piece = output_batch // batch
        batch_index = output_batch % batch
        source_offset = (
            ((piece * sequence + sequence_index) * batch + batch_index) * hidden
            + feature
        )
        value = tl.load(received + source_offset, mask=mask)
        tl.store(output + offsets, value, mask=mask)


    @triton.jit
    def _pack_gather_kernel(
        source,
        packed,
        total_elements: tl.constexpr,
        sequence: tl.constexpr,
        local_batch: tl.constexpr,
        hidden: tl.constexpr,
        multiple: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = (
            tl.program_id(0).to(tl.int64) * BLOCK
            + tl.arange(0, BLOCK).to(tl.int64)
        )
        mask = offsets < total_elements
        feature = offsets % hidden
        quotient = offsets // hidden
        batch_index = quotient % local_batch
        quotient = quotient // local_batch
        sequence_index = quotient % sequence
        piece = quotient // sequence
        source_batch = piece * local_batch + batch_index
        source_offset = (
            (sequence_index * (local_batch * multiple) + source_batch) * hidden
            + feature
        )
        value = tl.load(source + source_offset, mask=mask)
        tl.store(packed + offsets, value, mask=mask)


    @triton.jit
    def _assemble_gather_kernel(
        received,
        output,
        total_elements: tl.constexpr,
        source_sequence: tl.constexpr,
        local_batch: tl.constexpr,
        hidden: tl.constexpr,
        multiple: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = (
            tl.program_id(0).to(tl.int64) * BLOCK
            + tl.arange(0, BLOCK).to(tl.int64)
        )
        mask = offsets < total_elements
        feature = offsets % hidden
        quotient = offsets // hidden
        batch_index = quotient % local_batch
        output_sequence = quotient // local_batch
        half = source_sequence // 2
        output_block = output_sequence // half
        sequence_in_block = output_sequence % half
        is_back_block = output_block >= multiple
        piece = tl.where(is_back_block, 2 * multiple - output_block - 1, output_block)
        source_sequence_index = sequence_in_block + tl.where(is_back_block, half, 0)
        source_offset = (
            ((piece * source_sequence + source_sequence_index) * local_batch + batch_index)
            * hidden
            + feature
        )
        value = tl.load(received + source_offset, mask=mask)
        tl.store(output + offsets, value, mask=mask)


def _launch(kernel, source, output, **constants):
    total_elements = output.numel()
    block = 256
    kernel[(triton.cdiv(total_elements, block),)](
        source,
        output,
        total_elements=total_elements,
        BLOCK=block,
        **constants,
    )
    return output


def pack_scatter_send(source, multiple, *, use_triton=True):
    """Pack zigzag sequence pieces into one contiguous buffer per destination."""
    source = source.contiguous()
    sequence, batch, hidden = source.shape
    if sequence % (2 * multiple):
        raise ValueError(
            f"Sequence length {sequence} must divide 2*multiple={2 * multiple}."
        )
    output = source.new_empty(multiple, sequence // multiple, batch, hidden)
    if _should_use_triton(source, use_triton):
        return _launch(
            _pack_scatter_kernel,
            source,
            output,
            sequence=sequence,
            batch=batch,
            hidden=hidden,
            multiple=multiple,
        )
    view = source.view(2 * multiple, sequence // (2 * multiple), batch, hidden)
    for piece in range(multiple):
        output[piece].copy_(
            torch.cat([view[piece], view[2 * multiple - piece - 1]], dim=0)
        )
    return output


def join_scatter_recv(received, *, use_triton=True):
    """Join per-source receive buffers along the destination batch dimension."""
    if received.dim() != 4:
        raise ValueError(f"Expected [piece, sequence, batch, hidden], got {received.shape}.")
    multiple, sequence, batch, hidden = received.shape
    output = received.new_empty(sequence, batch * multiple, hidden)
    if _should_use_triton(received, use_triton):
        return _launch(
            _join_scatter_kernel,
            received,
            output,
            sequence=sequence,
            batch=batch,
            hidden=hidden,
            multiple=multiple,
        )
    return torch.cat(tuple(received.unbind(0)), dim=1)


def pack_gather_send(source, multiple, *, use_triton=True):
    """Pack interleaved batch lanes into contiguous buffers for CP contraction."""
    source = source.contiguous()
    sequence, batch, hidden = source.shape
    if batch % multiple:
        raise ValueError(f"Batch {batch} must be divisible by multiple={multiple}.")
    local_batch = batch // multiple
    output = source.new_empty(multiple, sequence, local_batch, hidden)
    if _should_use_triton(source, use_triton):
        return _launch(
            _pack_gather_kernel,
            source,
            output,
            sequence=sequence,
            local_batch=local_batch,
            hidden=hidden,
            multiple=multiple,
        )
    for piece in range(multiple):
        output[piece].copy_(
            source.narrow(1, piece * local_batch, local_batch)
        )
    return output


def assemble_gather_recv(received, *, use_triton=True):
    """Write received CP shards directly into the destination zigzag layout."""
    if received.dim() != 4:
        raise ValueError(f"Expected [piece, sequence, batch, hidden], got {received.shape}.")
    multiple, source_sequence, local_batch, hidden = received.shape
    if source_sequence % 2:
        raise ValueError(f"Source sequence {source_sequence} must be even.")
    output = received.new_empty(
        source_sequence * multiple,
        local_batch,
        hidden,
    )
    if _should_use_triton(received, use_triton):
        return _launch(
            _assemble_gather_kernel,
            received,
            output,
            source_sequence=source_sequence,
            local_batch=local_batch,
            hidden=hidden,
            multiple=multiple,
        )
    half = source_sequence // 2
    view = output.view(2 * multiple, half, local_batch, hidden)
    for piece in range(multiple):
        shard = received[piece].view(2, half, local_batch, hidden)
        view[piece].copy_(shard[0])
        view[2 * multiple - piece - 1].copy_(shard[1])
    return output
