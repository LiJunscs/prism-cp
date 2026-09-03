"""Cohort-aware TMF non-interleaved 1F1B runtime/profiling harness."""

import argparse
import os
import statistics
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for search_path in (str(PROJECT_ROOT), str(SCRIPT_DIR)):
    if search_path not in sys.path:
        sys.path.insert(0, search_path)

import torch
import torch.distributed as dist

from analyze_profile import summarize_profile_dir
from plan import build_execution_plan, format_stage_plan, parse_stage_layer_patterns
from profiling import NsightProfilerCapture, StageProfiler
from runtime import (TraceLogger, make_stage_groups, reset_model_rng,
                     synchronize_stage_parameters)
from scheduler import run_cohort_1f1b
from tmf_adapter import StageModelChunk, TMFAdapter
from utils import _ATTN_BACKEND_ENV, parse_benchmark_args, setup_dist


def add_tmf_1f1b_args(parser):
    parser.add_argument("--pipeline-parallel-size", type=int, default=2)
    parser.add_argument("--context-parallel-size", type=int, default=1)
    parser.add_argument("--micro-batches", type=int, default=4)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--layers-per-stage", type=int, default=2)
    parser.add_argument("--stage-layer-patterns", default=None)
    parser.add_argument("--cross-stage-cp-policy", choices=("forbid", "source", "destination"), default="forbid")
    parser.add_argument("--trace-schedule", action="store_true")
    parser.add_argument("--trace-dir", default=None)
    parser.add_argument("--check-input-grad-finite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--profile-stages", action="store_true")
    parser.add_argument("--profile-check-finite", action="store_true")
    parser.add_argument(
        "--tmf-transition-kernel",
        choices=("triton", "torch"),
        default="triton",
        help="Local CP-transition layout kernel backend.",
    )
    parser.add_argument(
        "--tmf-transition-overlap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Overlap CP-contraction lane communication with later producer "
            "lanes on a dedicated CUDA stream."
        ),
    )
    parser.add_argument(
        "--tmf-transition-lane-group-size",
        type=int,
        default=1,
        help=(
            "Number of consecutive temporal lanes combined into one NCCL "
            "enqueue group for an overlapped CP contraction. Must be positive "
            "and divide source_cp/destination_cp."
        ),
    )
    parser.add_argument(
        "--nsys-capture",
        action="store_true",
        help=(
            "Emit NVTX ranges and delimit measured iterations with CUDA "
            "Profiler API for nsys --capture-range=cudaProfilerApi."
        ),
    )
    return parser


def parse_args(argv=None):
    args = parse_benchmark_args(add_tmf_1f1b_args, argv=argv, description=__doc__)
    if args.profile_stages and not args.trace_dir:
        raise ValueError("--trace-dir is required for profiling artifacts.")
    if args.nsys_capture and not args.profile_stages:
        raise ValueError("--nsys-capture requires --profile-stages.")
    if args.warmup < 0 or args.iters < 1:
        raise ValueError("--warmup must be nonnegative and --iters positive.")
    if args.tmf_transition_lane_group_size < 1:
        raise ValueError("--tmf-transition-lane-group-size must be positive.")
    return args


def execution_plan(args, plans):
    return build_execution_plan(
        plans, pp_size=args.pipeline_parallel_size,
        stage_mesh_size=args.context_parallel_size,
        logical_microbatches=args.micro_batches,
        logical_microbatch_size=args.micro_batch_size,
        seq_length=args.seq_length, hidden_size=args.hidden_size,
        cross_stage_policy=args.cross_stage_cp_policy)


def print_plan(label, plans, plan):
    if dist.get_rank() != 0:
        return
    print(f"{label} execution plan:", flush=True)
    for stage, layers in enumerate(plans):
        print(f"  stage {stage}: {format_stage_plan(layers)} | K_stage={plan.stage_cohort_widths[stage]}", flush=True)
    print(f"  logical_microbatches={plan.logical_microbatches} | K={plan.cohort_width} | pipeline_cohorts={plan.pipeline_cohorts}", flush=True)


def build_adapter(args, label, stage, plans, plan, groups, trace, profiler=None):
    cp_cache, stage_group, stage_ranks, tp_group = groups
    trace.emit("model_chunk_build_begin", model_label=label, layer_plan=format_stage_plan(plans[stage]))
    model = StageModelChunk(args, stage, plans[stage], cp_cache, tp_group, trace)
    trace.emit("model_chunk_build_end", model_label=label, parameter_count=len(model.params), block_cps=[cp for cp, _ in model.blocks])
    adapter = TMFAdapter(args, plan, stage, model, cp_cache, stage_group,
                         stage_ranks, trace, profiler=profiler)
    return model, adapter


def run_once(args, stage, adapter, stage_group, plan, trace, *, phase, step,
             profiler=None, profile_active=False, check_finite=True):
    trace.set_context(phase=phase, step=step)
    if profiler is not None:
        profiler.set_active(profile_active, phase=phase, step=step)
    trace.emit("global_barrier_begin")
    dist.barrier()
    trace.emit("global_barrier_end")
    reset_model_rng(1234)
    torch.cuda.synchronize()
    operation = (
        profiler.record(
            "optimizer_step",
            pipeline_cohorts=plan.pipeline_cohorts,
            cohort_width=plan.cohort_width,
        )
        if profiler
        else nullcontext()
    )
    with operation:
        start = time.perf_counter()
        events, local_loss = run_cohort_1f1b(
            args, stage, adapter, stage_group, plan.boundaries, trace,
            profiler=profiler, check_parameter_finite=check_finite)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
    elapsed = torch.tensor(elapsed_ms, device="cuda")
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    loss = local_loss.clone()
    dist.all_reduce(loss)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite pipeline loss.")
    trace.emit("step_end", loss=loss.item(), local_elapsed_ms=elapsed_ms,
               worst_rank_ms=elapsed.item())
    return events, loss.item(), elapsed.item()


def run_profile(args, rank, stage, plans, plan, adapter, stage_group, trace, profiler):
    for step in range(args.warmup):
        run_once(args, stage, adapter, stage_group, plan, trace,
                 phase="profile_warmup", step=step, profiler=profiler,
                 check_finite=args.profile_check_finite)
    capture = NsightProfilerCapture(rank, stage, enabled=args.nsys_capture)
    worst_times = []
    dist.barrier()
    capture.start()
    try:
        # Ensure every rank has enabled its CUDA Profiler context before any
        # measured optimizer step starts.
        dist.barrier()
        for step in range(args.iters):
            with capture.step_range(step, phase="profile_measured"):
                _, _, worst = run_once(
                    args, stage, adapter, stage_group, plan, trace,
                    phase="profile_measured", step=step,
                    profiler=profiler, profile_active=True,
                    check_finite=args.profile_check_finite)
            worst_times.append(worst)
        # All GPU work is already synchronized by run_once. This barrier keeps
        # the first rank reaching cudaProfilerStop from truncating another
        # rank's final measured step in a multi-process nsys capture.
        dist.barrier()
    finally:
        capture.stop()
    profiler.finalize(Path(args.trace_dir) / f"profile_rank_{rank:04d}.json",
                      metadata={"stage_plan": format_stage_plan(plans[stage]),
                                "cohort_width": plan.cohort_width,
                                "pipeline_cohorts": plan.pipeline_cohorts,
                                "warmup": args.warmup, "iters": args.iters,
                                "nsys_capture": args.nsys_capture,
                                "tmf_transition_kernel": args.tmf_transition_kernel,
                                "tmf_transition_overlap": args.tmf_transition_overlap,
                                "tmf_transition_lane_group_size": args.tmf_transition_lane_group_size})
    dist.barrier()
    if rank == 0:
        report = summarize_profile_dir(args.trace_dir)
        print(f"TMF PROFILE PASS | measured={args.iters} median_worst_step={statistics.median(worst_times):.3f} ms | report={Path(args.trace_dir, 'profile_summary.json')}", flush=True)
        for item in report["top_components"][:5]:
            print(f"  diagnostic_range stage={item['stage']} rank={item['rank']} name={item['name']} wall={item['wall_total_ms']:.3f} ms cuda={item['cuda_total_ms']:.3f} ms", flush=True)


def main(argv=None):
    args = parse_args(argv)
    plans = parse_stage_layer_patterns(args.stage_layer_patterns,
                                       args.pipeline_parallel_size,
                                       args.layers_per_stage,
                                       args.context_parallel_size)
    plan = execution_plan(args, plans)
    for key, value in _ATTN_BACKEND_ENV[args.attention_backend].items():
        os.environ[key] = value
    if args.profile_stages and not args.profile_check_finite:
        args.check_input_grad_finite = False
    trace = None
    setup_dist()
    try:
        rank = dist.get_rank()
        expected = args.pipeline_parallel_size * args.context_parallel_size
        if dist.get_world_size() != expected:
            raise ValueError(f"world_size={dist.get_world_size()} must equal PP*stage_mesh={expected}.")
        stage = rank // args.context_parallel_size
        trace = TraceLogger(rank, stage, stdout=args.trace_schedule, trace_dir=args.trace_dir)
        trace.emit("process_group_build_begin", layer_plan=format_stage_plan(plans[stage]), cohort_width=plan.cohort_width)
        groups = make_stage_groups(plans, args.context_parallel_size)
        trace.emit("process_group_build_end")
        profiler = StageProfiler(
            rank,
            stage,
            enabled=args.profile_stages,
            nvtx_enabled=args.nsys_capture,
        )
        model, adapter = build_adapter(args, "runtime", stage, plans, plan,
                                       groups, trace, profiler)
        synchronize_stage_parameters(model.params, groups[1], groups[2][0], trace)
        print_plan("TMF cohort-aware 1F1B", plans, plan)
        trace.emit("rank_plan_ready", layer_plan=format_stage_plan(plans[stage]),
                   stage_cohort_width=plan.stage_cohort_widths[stage],
                   global_cohort_width=plan.cohort_width,
                   pipeline_cohorts=plan.pipeline_cohorts,
                   tmf_transition_kernel=args.tmf_transition_kernel,
                   tmf_transition_overlap=args.tmf_transition_overlap,
                   tmf_transition_lane_group_size=args.tmf_transition_lane_group_size,
                   boundaries=[asdict(boundary) for boundary in plan.boundaries])
        if args.profile_stages:
            run_profile(args, rank, stage, plans, plan, adapter, groups[1], trace, profiler)
        else:
            events, loss, worst = run_once(args, stage, adapter, groups[1], plan,
                                           trace, phase="smoke", step=0)
            if args.trace_schedule:
                print(f"[rank {rank} stage {stage}] cohort_events={events}", flush=True)
            if rank == 0:
                utilization = plan.pipeline_cohorts / (plan.pipeline_cohorts + args.pipeline_parallel_size - 1)
                print(f"TMF cohort 1F1B PASS | PP={args.pipeline_parallel_size} stage_mesh={args.context_parallel_size} M={args.micro_batches} K={plan.cohort_width} pipeline_units={plan.pipeline_cohorts} | worst_rank_step={worst:.3f} ms | cohort_pipeline_utilization={utilization:.4f} | loss={loss:.6e}", flush=True)
    except BaseException as exc:
        if trace is not None:
            trace.emit("fatal_error", error=repr(exc))
        raise
    finally:
        if trace is not None:
            trace.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
