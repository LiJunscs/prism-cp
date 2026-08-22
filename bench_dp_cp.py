# bench_dp_cp_scan_attn_gdn.py
"""
Benchmark Attention/GDN under one DP + CP hybrid parallel configuration.

Each launch benchmarks a single DP-CP combination; use run_bench_dp_cp.sh to
sweep the combinations. --global-batch-size is the fixed global batch: every
DP group computes its own microbatch = global_batch_size / dp_size and
constructs its local sequence directly (no full-sequence materialization).
MFU formula: global_flops / time / (world_size * single_gpu_peak_tflops)
Parallel groups follow Megatron-LM DP × CP 2D parallel layout.

NOTE: this benchmark does NOT perform DP parameter-gradient all-reduce, so the
backward latency excludes real DP gradient synchronization. MFU and activation
peak memory are still valid, but the numbers are NOT full DP training
throughput (see the dp_grad_sync=false column in the CSV).
"""

import csv
import os

import torch
import torch.distributed as dist

from utils import (
    _ATTN_BACKEND_ENV,
    attn_flops_global,
    build_attn_te,
    build_gdn_te,
    gdn_flops_global,
    get_pretrain_batch_on_this_cp_rank,
    parse_benchmark_args,
    setup_dist,
    timed_bwd,
    timed_fwd,
    timed_fwd_bwd,
)


def make_dp_cp_groups(cp_size: int):
    """
    Construct DP group & CP group for DP × CP hybrid parallel (Megatron style).
    Layout:
        dp_rank = global_rank // cp_size
        cp_rank = global_rank % cp_size
    - CP group: all ranks sharing same dp_rank
    - DP group: all ranks sharing same cp_rank, stride = cp_size
    Returns: dp_group, cp_group, dp_size, dp_rank, cp_rank
    """
    world_size = dist.get_world_size()
    dp_size = world_size // cp_size
    assert world_size == dp_size * cp_size, f"world_size({world_size}) must be divisible by cp_size({cp_size})"

    global_rank = dist.get_rank()
    for dp_rank in range(dp_size):
        cp_group_ranks = list(range(dp_rank * cp_size, (dp_rank + 1) * cp_size))
        cp_group = dist.new_group(ranks=cp_group_ranks)
        if global_rank in cp_group_ranks:
            local_cp_group = cp_group
    for cp_rank in range(cp_size):
        dp_group_ranks = list(range(cp_rank, world_size, cp_size))
        dp_group = dist.new_group(ranks=dp_group_ranks)
        if global_rank in dp_group_ranks:
            local_dp_group = dp_group
    dp_rank = local_dp_group.rank()
    cp_rank = local_cp_group.rank()

    return local_dp_group, local_cp_group, dp_size, dp_rank, cp_rank


def add_dp_cp_args(parser):
    """Add fixed-global-load DP-CP profiling arguments."""
    parser.add_argument("--module", required=True, choices=["attention", "gdn"])
    parser.add_argument("--cp", required=True, type=int)
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=8,
        help="Fixed global batch; per-DP-group microbatch = global / dp_size.",
    )
    parser.add_argument(
        "--gpu-peak-tflops",
        type=float,
        default=989.0,
        help="Single GPU peak TFLOPS (BF16).",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="bench_result.csv",
        help="Output CSV file path.",
    )
    return parser


def parse_args(extra_args_provider=None, argv=None):
    """Parse common arguments plus DP-CP and caller-provided arguments."""

    def provider(parser):
        add_dp_cp_args(parser)
        return extra_args_provider(parser) if extra_args_provider is not None else parser

    return parse_benchmark_args(
        provider,
        argv=argv,
        description="Profile one layer under a fixed-global-load DP-CP layout.",
    )



def run_single_cp_config(args, cp_size: int, world_size: int, rank: int, result_rows: list, tp_group):
    dp_group, cp_group, dp_size, dp_rank, cp_rank = make_dp_cp_groups(cp_size)
    local_seq = args.seq_length // cp_size

    if args.global_batch_size % dp_size != 0:
        if rank == 0:
            print(
                f"[SKIP] cp_size={cp_size}, global_batch_size={args.global_batch_size} "
                f"not divisible by dp_size={dp_size}, skip"
            )
        dist.destroy_process_group(dp_group)
        dist.destroy_process_group(cp_group)
        dist.barrier()
        return
    microbatch = args.global_batch_size // dp_size

    if local_seq < 128:
        if rank == 0:
            print(f"[SKIP] cp_size={cp_size}, local_seq={local_seq} < 128, skip")
        dist.destroy_process_group(dp_group)
        dist.destroy_process_group(cp_group)
        dist.barrier()
        return

    # build module
    if args.module == "attention":
        fn = build_attn_te(
            args.hidden_size, args.num_attention_heads,
            args.num_query_groups, cp_size,
            tp_group=tp_group, cp_group=cp_group
        )
        global_flops = attn_flops_global(
            microbatch, args.seq_length,
            args.hidden_size, args.num_attention_heads,
            args.num_query_groups, dp_size
        )
    else:
        fn = build_gdn_te(
            args.hidden_size, args.num_attention_heads,
            args.num_query_groups, cp_size,
            tp_group=tp_group, cp_group=cp_group, args=args
        )
        global_flops = gdn_flops_global(
            microbatch, args.seq_length,
            args.hidden_size, args.gdn_key_head_dim,
            args.gdn_value_head_dim, args.gdn_num_key_heads,
            args.gdn_num_value_heads, args.gdn_conv_kernel_dim,
            dp_size
        )

    # local tensor [local_seq, microbatch, hidden]: construct the partitioned
    # sequence directly, no full-sequence materialization
    x = get_pretrain_batch_on_this_cp_rank(
        args.seq_length, microbatch, args.hidden_size, cp_group,
        device="cuda", dtype=torch.bfloat16,
    )
    x = x.detach().requires_grad_(True)

    # --------------------------
    # Warmup strengthen (no_grad, flush cache before measurement)
    # --------------------------
    torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(args.warmup):
            fn(x)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # --------------------------
    # Forward benchmark & memory
    # --------------------------
    torch.cuda.reset_peak_memory_stats()
    t_fwd = timed_fwd(fn, x, warmup=0, iters=args.iters)
    torch.cuda.synchronize()
    fwd_peaks = torch.tensor(
        [torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()],
        dtype=torch.float64,
        device="cuda",
    )
    dist.all_reduce(fwd_peaks, op=dist.ReduceOp.MAX)
    fwd_max_alloc_gb, fwd_max_resv_gb = (fwd_peaks / 1024**3).tolist()
    avg_fwd_ms = sum(t_fwd) / len(t_fwd)
    fwd_sec = avg_fwd_ms / 1000.0
    global_tflops_fwd = global_flops / fwd_sec / 1e12
    mfu_fwd = global_tflops_fwd / (world_size * args.gpu_peak_tflops)

    # --------------------------
    # Backward benchmark & memory
    # --------------------------
    torch.cuda.reset_peak_memory_stats()
    t_bwd = timed_bwd(fn, x, warmup=0, iters=args.iters)
    torch.cuda.synchronize()
    bwd_peaks = torch.tensor(
        [torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()],
        dtype=torch.float64,
        device="cuda",
    )
    dist.all_reduce(bwd_peaks, op=dist.ReduceOp.MAX)
    bwd_max_alloc_gb, bwd_max_resv_gb = (bwd_peaks / 1024**3).tolist()
    avg_bwd_ms = sum(t_bwd) / len(t_bwd)
    bwd_sec = avg_bwd_ms / 1000.0
    global_tflops_bwd = (global_flops * 2) / bwd_sec / 1e12
    mfu_bwd = global_tflops_bwd / (world_size * args.gpu_peak_tflops)

    # --------------------------
    # Fwd+Bwd benchmark & memory
    # --------------------------
    torch.cuda.reset_peak_memory_stats()
    t_both = timed_fwd_bwd(fn, x, warmup=0, iters=args.iters)
    torch.cuda.synchronize()
    both_peaks = torch.tensor(
        [torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()],
        dtype=torch.float64,
        device="cuda",
    )
    dist.all_reduce(both_peaks, op=dist.ReduceOp.MAX)
    both_max_alloc_gb, both_max_resv_gb = (both_peaks / 1024**3).tolist()
    avg_both_ms = sum(t_both) / len(t_both)
    both_sec = avg_both_ms / 1000.0
    global_tflops_both = (global_flops * 3) / both_sec / 1e12
    mfu_both = global_tflops_both / (world_size * args.gpu_peak_tflops)

    # collect csv row
    row = {
        "module": args.module,
        "world_size": world_size,
        "cp_size": cp_size,
        "dp_size": dp_size,
        "seq_len": args.seq_length,
        "local_seq": local_seq,
        "global_batch_size": args.global_batch_size,
        "microbatch": microbatch,
        "hidden_size": args.hidden_size,
        "global_flops": global_flops,
        "avg_fwd_ms": avg_fwd_ms,
        "global_tflops_fwd": global_tflops_fwd,
        "mfu_fwd": mfu_fwd,
        "fwd_max_alloc_gb": fwd_max_alloc_gb,
        "fwd_max_resv_gb": fwd_max_resv_gb,
        "avg_bwd_ms": avg_bwd_ms,
        "global_tflops_bwd": global_tflops_bwd,
        "mfu_bwd": mfu_bwd,
        "bwd_max_alloc_gb": bwd_max_alloc_gb,
        "bwd_max_resv_gb": bwd_max_resv_gb,
        "avg_both_ms": avg_both_ms,
        "global_tflops_both": global_tflops_both,
        "mfu_both": mfu_both,
        "both_max_alloc_gb": both_max_alloc_gb,
        "both_max_resv_gb": both_max_resv_gb,
        "gpu_peak_tflops": args.gpu_peak_tflops,
        "dp_grad_sync": "false",
    }
    result_rows.append(row)

    # console log
    if rank == 0:
        print("-" * 90)
        print(f"CONFIG: module={args.module} | world_size={world_size} | cp_size={cp_size} | dp_size={dp_size}")
        print(f"seq_len={args.seq_length}, local_seq={local_seq}, "
              f"global_batch={args.global_batch_size}, microbatch={microbatch}")
        print(f"Global FLOPS={global_flops:.3e}")
        print(f"  fwd:  {avg_fwd_ms:.2f} ms | TF={global_tflops_fwd:.2f} | MFU={mfu_fwd:.4f} | PeakAlloc={fwd_max_alloc_gb:.2f}GB")
        print(f"  bwd:  {avg_bwd_ms:.2f} ms | TF={global_tflops_bwd:.2f} | MFU={mfu_bwd:.4f} | PeakAlloc={bwd_max_alloc_gb:.2f}GB")
        print(f"  both: {avg_both_ms:.2f} ms | TF={global_tflops_both:.2f} | MFU={mfu_both:.4f} | PeakAlloc={both_max_alloc_gb:.2f}GB")
        print("-" * 90)

    # clean communication groups
    dist.barrier()
    dist.destroy_process_group(dp_group)
    dist.destroy_process_group(cp_group)
    dist.destroy_process_group(tp_group)
    dist.barrier()


def main():
    args = parse_args()
    for k, v in _ATTN_BACKEND_ENV[args.attention_backend].items():
        os.environ[k] = v

    setup_dist()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    result_rows = []
    for i in range(world_size):
        tp_group = dist.new_group(ranks=[i])
        if rank == i:
            local_tp_group = tp_group
    dist.barrier()

    if rank == 0:
        print("=" * 100)
        print(f"DP-CP Hybrid Benchmark Start: ")
        print(f"CP={args.cp}, DP={world_size // args.cp}, world_size={world_size}, single GPU peak TFLOPS={args.gpu_peak_tflops}")
        print(f"Module: {args.module}, seq_len={args.seq_length}, global_batch={args.global_batch_size}")
        print(f"NOTE: dp_grad_sync=false - no DP parameter-gradient all-reduce; "
              f"bwd latency excludes real DP grad sync")
        print(f"CSV output path: {args.csv_path}")
        print("=" * 100)

    dist.barrier()
    run_single_cp_config(args, args.cp, world_size, rank, result_rows, local_tp_group)

    # rank0 write csv
    if rank == 0 and len(result_rows) > 0:
        fieldnames = list(result_rows[0].keys())
        with open(args.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(result_rows)
        print(f"\n[Result Saved] CSV written to {args.csv_path}, total entries: {len(result_rows)}")

    dist.destroy_process_group()
    if rank == 0:
        print(f"CP size {args.cp} scan finished!", flush=True)

    # Bypass interpreter teardown: the CUDA/NCCL atexit cleanup can keep the
    # worker process alive for a long time, which delays torchrun's exit and
    # stalls the sweep between tasks. The benchmark has no state to persist
    # after this point (CSV is already written), so force-exit immediately.
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    os._exit(0)


if __name__ == "__main__":
    main()
