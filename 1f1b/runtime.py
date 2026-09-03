"""Distributed runtime helpers shared by the TMF adapter and scheduler."""

import json
import os
import threading
import time

import torch
import torch.distributed as dist


def reset_model_rng(seed=1234):
    """Reset torch and Megatron/TE RNG streams before a comparable step."""
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


class TraceLogger:
    """Line-buffered per-rank JSONL trace for distributed stall diagnosis."""

    def __init__(self, rank, stage, stdout=False, trace_dir=None):
        self.rank = rank
        self.stage = stage
        self.stdout = stdout
        self.sequence = 0
        self.lock = threading.Lock()
        self.context = {}
        self.file = None
        if trace_dir:
            os.makedirs(trace_dir, exist_ok=True)
            path = os.path.join(trace_dir, f"rank_{rank:04d}.jsonl")
            self.file = open(path, "w", encoding="utf-8", buffering=1)

    @property
    def enabled(self):
        return self.stdout or self.file is not None

    def set_context(self, **fields):
        self.context = dict(fields)

    def emit(self, event, cohort=None, **fields):
        if not self.enabled:
            return
        with self.lock:
            record = {
                "sequence": self.sequence,
                "time_ns": time.time_ns(),
                "monotonic_ns": time.monotonic_ns(),
                "rank": self.rank,
                "stage": self.stage,
                "cohort": cohort,
                "event": event,
                **self.context,
            }
            self.sequence += 1
            for key, value in fields.items():
                if isinstance(value, (tuple, torch.Size)):
                    value = list(value)
                record[key] = value
            line = json.dumps(record, sort_keys=True, ensure_ascii=False)
            if self.stdout:
                print(f"[tmf-1f1b-trace] {line}", flush=True)
            if self.file is not None:
                self.file.write(line + "\n")

    def close(self):
        if self.file is not None:
            self.file.flush()
            self.file.close()
            self.file = None


def make_stage_groups(stage_plans, stage_mesh_size):
    """Create every stage-local CP subgroup in globally identical order."""
    rank = dist.get_rank()
    pp_size = len(stage_plans)
    cp_sizes = sorted(
        {stage_mesh_size}
        | {layer.cp_size for stage_plan in stage_plans for layer in stage_plan}
    )
    local_cp_cache = {}
    local_stage_ranks = None

    for stage in range(pp_size):
        stage_start = stage * stage_mesh_size
        stage_ranks = list(range(stage_start, stage_start + stage_mesh_size))
        for cp_size in cp_sizes:
            for offset in range(0, stage_mesh_size, cp_size):
                ranks = stage_ranks[offset : offset + cp_size]
                group = dist.new_group(ranks)
                if rank in ranks:
                    local_cp_cache[cp_size] = (group, ranks)
                    local_stage_ranks = stage_ranks

    tp_group = None
    for global_rank in range(pp_size * stage_mesh_size):
        group = dist.new_group([global_rank])
        if rank == global_rank:
            tp_group = group

    if local_stage_ranks is None or stage_mesh_size not in local_cp_cache:
        raise RuntimeError("Failed to create this rank's stage-local process groups.")
    stage_group = local_cp_cache[stage_mesh_size][0]
    return local_cp_cache, stage_group, local_stage_ranks, tp_group


def synchronize_stage_parameters(params, stage_group, stage_root, trace):
    for param_index, param in enumerate(params):
        trace.emit(
            "parameter_broadcast_begin",
            parameter=param_index,
            source=stage_root,
            shape=tuple(param.shape),
        )
        dist.broadcast(param.data, src=stage_root, group=stage_group)
        trace.emit(
            "parameter_broadcast_end",
            parameter=param_index,
            source=stage_root,
        )


def reduce_stage_gradients(
    params,
    stage_group,
    trace,
    *,
    check_finite=True,
    profiler=None,
):
    gradients = []
    for param_index, param in enumerate(params):
        if param.grad is None:
            raise RuntimeError(
                f"Stage parameter {param_index} gradient was not produced."
            )
        trace.emit(
            "gradient_all_reduce_begin",
            parameter=param_index,
            shape=tuple(param.shape),
        )
        profile_range = (
            profiler.record(
                "gradient_all_reduce",
                parameter=param_index,
                shape=tuple(param.shape),
                dtype=str(param.grad.dtype),
            )
            if profiler is not None
            else None
        )
        if profile_range is None:
            dist.all_reduce(param.grad, group=stage_group)
        else:
            with profile_range:
                dist.all_reduce(param.grad, group=stage_group)
        trace.emit("gradient_all_reduce_end", parameter=param_index)
        if check_finite and not torch.isfinite(param.grad).all():
            raise RuntimeError(f"Non-finite gradient for stage parameter {param_index}.")
        gradients.append(param.grad)
    return gradients
