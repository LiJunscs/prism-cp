"""Profile one Attention or GDN layer at a selected CP size."""

import csv
import datetime
import os

import numpy as np
import torch
import torch.distributed as dist

from utils import (
    _ATTN_BACKEND_ENV,
    attn_flops_global,
    build_attn_te,
    build_gdn_te,
    gdn_flops_global,
    get_pretrain_batch_on_this_cp_rank,
    make_cp_tp_groups,
    parse_benchmark_args,
    setup_dist,
    timed_bwd,
    timed_fwd,
    timed_fwd_bwd,
)


def add_single_layer_args(parser):
    """Add arguments specific to unconstrained single-layer profiling."""
    parser.add_argument("--module", required=True, choices=["attention", "gdn"])
    parser.add_argument("--cp", required=True, type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gpu-peak-tflops", type=float, default=989.0)
    parser.add_argument(
        "--log-csv",
        type=str,
        default="bench_mfu_mem_log.csv",
        help="Save MFU and memory log.",
    )
    return parser


def parse_args(extra_args_provider=None, argv=None):
    """Parse common arguments plus single-layer and caller-provided arguments."""

    def provider(parser):
        add_single_layer_args(parser)
        return extra_args_provider(parser) if extra_args_provider is not None else parser

    return parse_benchmark_args(
        provider,
        argv=argv,
        description="Profile one Attention or GDN layer.",
    )


def gather_all_gpu_data(local_val: float) -> list[float]:
    """Allgather single float across WORLD group"""
    local_tensor = torch.tensor([local_val], dtype=torch.float32, device="cuda")
    world_size = dist.get_world_size()
    gathered = [torch.zeros_like(local_tensor) for _ in range(world_size)]
    dist.all_gather(gathered, local_tensor)
    return [t.item() for t in gathered]


def calc_stats(data: list[float]) -> dict:
    """Compute mean/min/max/std"""
    arr = np.array(data)
    return {
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
    }


def reset_gpu_mem_stats():
    torch.cuda.reset_peak_memory_stats()


def get_peak_gpu_mem_gb() -> float:
    """Return peak allocated memory in GB"""
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def write_csv_log(
    args,
    mfu_stats,
    mem_stats,
    total_flops,
    world_size,
    dp_size,
    avg_fwd,
    avg_bwd,
    avg_both,
    global_tokens_per_second,
):
    # Keep raw timing/throughput alongside derived MFU so the measurements can
    # be consumed by a performance model without reconstructing hidden inputs.
    header = [
        "timestamp", "module", "cp_size", "dp_size", "world_size",
        "seq_len", "bs", "hidden", "attn_backend",
        "warmup", "iters", "gpu_peak_tflops",
        "fwd_ms_mean", "bwd_ms_mean", "both_ms_mean",
        "global_tokens_per_second",
        # MFU mean/std/min/max
        "fwd_mfu_mean", "fwd_mfu_std", "fwd_mfu_min", "fwd_mfu_max",
        "bwd_mfu_mean", "bwd_mfu_std", "bwd_mfu_min", "bwd_mfu_max",
        "both_mfu_mean", "both_mfu_std", "both_mfu_min", "both_mfu_max",
        # Memory peak GB stats
        "mem_gb_mean", "mem_gb_std", "mem_gb_min", "mem_gb_max",
        # GDN extra
        "gdn_kdim", "gdn_vdim", "gdn_kheads", "gdn_vheads", "gdn_conv_k"
    ]
    row = [
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        args.module, args.cp, dp_size, world_size,
        args.seq_length, args.batch_size, args.hidden_size, args.attention_backend,
        args.warmup, args.iters, args.gpu_peak_tflops,
        avg_fwd, avg_bwd, avg_both, global_tokens_per_second,
        # FWD MFU
        mfu_stats["fwd"]["mean"], mfu_stats["fwd"]["std"], mfu_stats["fwd"]["min"], mfu_stats["fwd"]["max"],
        # BWD MFU
        mfu_stats["bwd"]["mean"], mfu_stats["bwd"]["std"], mfu_stats["bwd"]["min"], mfu_stats["bwd"]["max"],
        # FWD+BWD MFU
        mfu_stats["both"]["mean"], mfu_stats["both"]["std"], mfu_stats["both"]["min"], mfu_stats["both"]["max"],
        # Memory
        mem_stats["mean"], mem_stats["std"], mem_stats["min"], mem_stats["max"],
        # GDN args
        args.gdn_key_head_dim, args.gdn_value_head_dim,
        args.gdn_num_key_heads, args.gdn_num_value_heads, args.gdn_conv_kernel_dim
    ]

    file_exists = os.path.exists(args.log_csv)
    with open(args.log_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


def main():
    args = parse_args()
    for k, v in _ATTN_BACKEND_ENV[args.attention_backend].items():
        os.environ[k] = v

    setup_dist()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    cp_size = args.cp
    dp_size = world_size // cp_size
    local_seq = args.seq_length // cp_size
    if local_seq < 128:
        if rank == 0:
            print(f"ERROR: local_seq {local_seq} < 128, skip benchmark")
        dist.destroy_process_group()
        return

    tp_group, cp_group = make_cp_tp_groups(cp_size)

    # Build module & compute global flops
    if args.module == "attention":
        fn = build_attn_te(args.hidden_size, args.num_attention_heads,
                           args.num_query_groups, cp_size, tp_group, cp_group)
        total_flops = attn_flops_global(args.batch_size, args.seq_length,
                                        args.hidden_size, args.num_attention_heads,
                                        args.num_query_groups, dp_size)
    else:
        fn = build_gdn_te(args.hidden_size, args.num_attention_heads,
                          args.num_query_groups, cp_size, tp_group, cp_group, args)
        total_flops = gdn_flops_global(args.batch_size, args.seq_length,
                                       args.hidden_size, args.gdn_key_head_dim,
                                       args.gdn_value_head_dim, args.gdn_num_key_heads,
                                       args.gdn_num_value_heads, args.gdn_conv_kernel_dim, dp_size)

    per_gpu_flops = total_flops / world_size

    # Input tensor
    x_split = get_pretrain_batch_on_this_cp_rank(args.seq_length, args.batch_size, args.hidden_size, cp_group, device="cuda", dtype=torch.bfloat16)
    x = x_split.detach().requires_grad_(True)

    # -------------------------- FWD Benchmark + MFU --------------------------
    reset_gpu_mem_stats()
    t_fwd = timed_fwd(fn, x, args.warmup, args.iters)
    avg_fwd = sum(t_fwd) / len(t_fwd)
    mfu_fwd = (per_gpu_flops / (avg_fwd / 1000) / 1e12) / args.gpu_peak_tflops

    # -------------------------- BWD Benchmark + MFU --------------------------
    reset_gpu_mem_stats()
    t_bwd = timed_bwd(fn, x, args.warmup, args.iters)
    avg_bwd = sum(t_bwd) / len(t_bwd)
    mfu_bwd = (per_gpu_flops * 2 / (avg_bwd / 1000) / 1e12) / args.gpu_peak_tflops

    # -------------------------- FWD+BWD Benchmark + Memory Peak --------------------------
    reset_gpu_mem_stats()
    t_both = timed_fwd_bwd(fn, x, args.warmup, args.iters)
    avg_both = sum(t_both) / len(t_both)
    mfu_both = (per_gpu_flops * 3 / (avg_both / 1000) / 1e12) / args.gpu_peak_tflops
    global_tokens = args.batch_size * args.seq_length * dp_size
    global_tokens_per_second = global_tokens / (avg_both / 1000)
    local_peak_mem_gb = get_peak_gpu_mem_gb()

    # -------------------------- AllGather all ranks MFU & Memory --------------------------
    all_fwd_mfu = gather_all_gpu_data(mfu_fwd)
    all_bwd_mfu = gather_all_gpu_data(mfu_bwd)
    all_both_mfu = gather_all_gpu_data(mfu_both)
    all_mem_gb = gather_all_gpu_data(local_peak_mem_gb)

    # Calculate aggregated stats
    mfu_stats = {
        "fwd": calc_stats(all_fwd_mfu),
        "bwd": calc_stats(all_bwd_mfu),
        "both": calc_stats(all_both_mfu),
    }
    mem_stats = calc_stats(all_mem_gb)

    # -------------------------- Rank 0 Print Only MFU + Memory --------------------------
    if rank == 0:
        print("="*70)
        print(f"Module: {args.module} | CP={cp_size}, DP={dp_size}, World={world_size}")
        print(f"SeqLen={args.seq_length}, Hidden={args.hidden_size}, BS={args.batch_size}")
        if args.module == "gdn":
            print(f"GDN linear_cp_mode={args.gdn_linear_cp_mode or 'MEGATRON_DEFAULT'}")
        print("="*70)
        print(f"[Single GPU Rank0 MFU]")
        print(f"  FWD MFU:  {mfu_fwd:.4f}")
        print(f"  BWD MFU:  {mfu_bwd:.4f}")
        print(f"  FWD+BWD MFU: {mfu_both:.4f}")
        print(f"  FWD time: {avg_fwd:.3f} ms")
        print(f"  BWD time: {avg_bwd:.3f} ms")
        print(f"  FWD+BWD time: {avg_both:.3f} ms")
        print(f"  Global tokens/s: {global_tokens_per_second:.0f}")
        print(f"  Peak Mem: {local_peak_mem_gb:.2f} GB")
        print("-"*70)
        print(f"[All GPU Aggregated Statistics]")
        print(f"FWD MFU: mean={mfu_stats['fwd']['mean']:.4f} ± {mfu_stats['fwd']['std']:.4f} | min={mfu_stats['fwd']['min']:.4f}, max={mfu_stats['fwd']['max']:.4f}")
        print(f"BWD MFU: mean={mfu_stats['bwd']['mean']:.4f} ± {mfu_stats['bwd']['std']:.4f} | min={mfu_stats['bwd']['min']:.4f}, max={mfu_stats['bwd']['max']:.4f}")
        print(f"FWD+BWD MFU: mean={mfu_stats['both']['mean']:.4f} ± {mfu_stats['both']['std']:.4f} | min={mfu_stats['both']['min']:.4f}, max={mfu_stats['both']['max']:.4f}")
        print(f"Peak Memory (GB): mean={mem_stats['mean']:.2f} ± {mem_stats['std']:.2f} | min={mem_stats['min']:.2f}, max={mem_stats['max']:.2f}")
        print("="*70)
        print(f"Result log saved to: {args.log_csv}")
        print("="*70)

        # Save MFU & memory to CSV
        write_csv_log(
            args,
            mfu_stats,
            mem_stats,
            total_flops,
            world_size,
            dp_size,
            avg_fwd,
            avg_bwd,
            avg_both,
            global_tokens_per_second,
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
