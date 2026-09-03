"""CUDA-aware component profiling for the TMF 1F1B benchmark."""

from __future__ import annotations

import json
import statistics
import time
from contextlib import nullcontext
from pathlib import Path

import torch


def _json_value(value):
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, torch.Size):
        return list(value)
    return value


def _nvtx_message(name, fields):
    """Build a compact, machine-readable NVTX label."""
    parts = [f"tmf::{name}"]
    for key, value in fields.items():
        encoded = json.dumps(_json_value(value), separators=(",", ":"))
        parts.append(f"{key}={encoded}")
    return "|".join(parts)


class _NsightStepRange:
    def __init__(self, capture, step, **metadata):
        self.capture = capture
        self.step_index = step
        self.metadata = metadata
        self.pushed = False

    def __enter__(self):
        if self.capture.enabled:
            fields = {
                "rank": self.capture.rank,
                "stage": self.capture.stage,
                "step": self.step_index,
                **self.metadata,
            }
            torch.cuda.nvtx.range_push(_nvtx_message("measured_step", fields))
            self.pushed = True
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.pushed:
            torch.cuda.nvtx.range_pop()
            self.capture.step(self.step_index, failed=exc_type is not None)
        return False


class NsightProfilerCapture:
    """Control an external Nsight Systems capture through CUDA Profiler API.

    CUDA Profiler API has start/stop primitives but no step primitive. ``step``
    therefore emits an NVTX boundary marker, while ``step_range`` provides the
    enclosing NVTX range for a measured optimizer step.
    """

    def __init__(self, rank, stage, enabled=False):
        self.rank = rank
        self.stage = stage
        self.enabled = bool(enabled)
        self.running = False

    def start(self):
        if not self.enabled:
            return
        if self.running:
            raise RuntimeError("Nsight capture is already running.")
        torch.cuda.profiler.start()
        self.running = True
        torch.cuda.nvtx.mark(
            _nvtx_message(
                "capture_start",
                {"rank": self.rank, "stage": self.stage},
            )
        )

    def step_range(self, step, **metadata):
        return _NsightStepRange(self, step, **metadata)

    def step(self, step, **metadata):
        if not self.running:
            return
        torch.cuda.nvtx.mark(
            _nvtx_message(
                "step_end",
                {
                    "rank": self.rank,
                    "stage": self.stage,
                    "step": step,
                    **metadata,
                },
            )
        )

    def stop(self):
        if not self.enabled or not self.running:
            return
        torch.cuda.nvtx.mark(
            _nvtx_message(
                "capture_stop",
                {"rank": self.rank, "stage": self.stage},
            )
        )
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
        self.running = False


class _ProfileRange:
    def __init__(self, profiler, name, metadata):
        self.profiler = profiler
        self.name = name
        self.metadata = metadata
        self.operation_id = None
        self.nvtx_pushed = False

    def __enter__(self):
        if self.profiler.nvtx_enabled:
            fields = {
                "rank": self.profiler.rank,
                "stage": self.profiler.stage,
                **self.profiler.context,
                **self.metadata,
            }
            torch.cuda.nvtx.range_push(_nvtx_message(self.name, fields))
            self.nvtx_pushed = True
        try:
            self.operation_id = self.profiler.begin(
                self.name,
                _nvtx_mark=False,
                **self.metadata,
            )
        except BaseException:
            if self.nvtx_pushed:
                torch.cuda.nvtx.range_pop()
                self.nvtx_pushed = False
            raise
        return self.operation_id

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.profiler.end(self.operation_id, failed=exc_type is not None)
        finally:
            if self.nvtx_pushed:
                torch.cuda.nvtx.range_pop()
        return False


class StageProfiler:
    """Record paired CUDA-event and host-wall timings without forced syncs.

    CUDA events measure work on the current CUDA stream. ``wall_ms`` records
    host-side launch/wait time and is especially useful for NCCL P2P waits.
    Nested ranges are intentional; consumers must not add parent and child
    categories together. In grouped backward, sibling hook ranges can also
    overlap while autograd traverses several graph branches, so their totals
    are inclusive diagnostics rather than an additive phase decomposition.
    """

    def __init__(self, rank, stage, enabled=False, nvtx_enabled=False):
        self.rank = rank
        self.stage = stage
        self.enabled = enabled
        self.nvtx_enabled = bool(enabled and nvtx_enabled)
        self.active = False
        self.context = {}
        self.counter = 0
        self.open_operations = {}
        self.samples = []

    def set_active(self, active, **context):
        if self.open_operations:
            raise RuntimeError("Cannot change profiler state with open operations.")
        self.active = bool(self.enabled and active)
        self.context = {key: _json_value(value) for key, value in context.items()}

    def record(self, name, **metadata):
        if not self.active:
            return nullcontext()
        return _ProfileRange(self, name, metadata)

    def begin(self, name, _nvtx_mark=True, **metadata):
        if not self.active:
            return None
        operation_id = self.counter
        self.counter += 1
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        record = {
            "operation_id": operation_id,
            "name": name,
            "rank": self.rank,
            "stage": self.stage,
            **self.context,
            **{key: _json_value(value) for key, value in metadata.items()},
            "wall_start_ns": time.perf_counter_ns(),
            "cuda_start": start_event,
            "_nvtx_marked": bool(self.nvtx_enabled and _nvtx_mark),
        }
        self.open_operations[operation_id] = record
        if record["_nvtx_marked"]:
            torch.cuda.nvtx.mark(
                _nvtx_message(
                    f"{name}_begin",
                    {
                        "rank": self.rank,
                        "stage": self.stage,
                        **self.context,
                        "operation_id": operation_id,
                        **metadata,
                    },
                )
            )
        return operation_id

    def end(self, operation_id, **metadata):
        if operation_id is None:
            return
        try:
            record = self.open_operations.pop(operation_id)
        except KeyError as exc:
            raise RuntimeError(f"Unknown profiling operation {operation_id}.") from exc
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        record.update({key: _json_value(value) for key, value in metadata.items()})
        record["wall_end_ns"] = time.perf_counter_ns()
        record["cuda_end"] = end_event
        if record.pop("_nvtx_marked"):
            mark_fields = {
                key: value
                for key, value in record.items()
                if key
                not in {
                    "name",
                    "wall_start_ns",
                    "wall_end_ns",
                    "cuda_start",
                    "cuda_end",
                }
            }
            torch.cuda.nvtx.mark(
                _nvtx_message(f"{record['name']}_end", mark_fields)
            )
        self.samples.append(record)

    def finalize(self, output_path, metadata=None):
        if self.open_operations:
            names = [value["name"] for value in self.open_operations.values()]
            raise RuntimeError(f"Unclosed profiling operations: {names}.")
        torch.cuda.synchronize()
        serialized = []
        for sample in self.samples:
            start_event = sample.pop("cuda_start")
            end_event = sample.pop("cuda_end")
            sample["cuda_ms"] = float(start_event.elapsed_time(end_event))
            sample["wall_ms"] = (
                sample.pop("wall_end_ns") - sample.pop("wall_start_ns")
            ) / 1.0e6
            serialized.append(sample)

        grouped = {}
        for sample in serialized:
            group_fields = {
                key: value
                for key, value in sample.items()
                if key
                not in {
                    "operation_id",
                    "rank",
                    "step",
                    "phase",
                    "cuda_ms",
                    "wall_ms",
                    "failed",
                    "logical_ids",
                    "cohort",
                }
            }
            key = json.dumps(group_fields, sort_keys=True)
            grouped.setdefault(key, []).append(sample)

        summary = []
        for key, values in grouped.items():
            fields = json.loads(key)
            cuda_values = [value["cuda_ms"] for value in values]
            wall_values = [value["wall_ms"] for value in values]
            summary.append(
                {
                    **fields,
                    "count": len(values),
                    "cuda_total_ms": sum(cuda_values),
                    "cuda_mean_ms": statistics.fmean(cuda_values),
                    "cuda_max_ms": max(cuda_values),
                    "wall_total_ms": sum(wall_values),
                    "wall_mean_ms": statistics.fmean(wall_values),
                    "wall_max_ms": max(wall_values),
                }
            )

        payload = {
            "schema_version": 1,
            "rank": self.rank,
            "stage": self.stage,
            "metadata": metadata or {},
            "samples": serialized,
            "summary": sorted(
                summary,
                key=lambda value: value["cuda_total_ms"],
                reverse=True,
            ),
        }
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return payload
