"""No-PP performance model for module-level heterogeneous CP patterns.

Module latency comes from the fixed-global-load DP/CP frontier CSV. Attention
ring communication is assumed fully hidden. GDN latency is used as measured;
its CP1 latency is also reported as the no-All-to-All baseline and the CP>1
increment is checked against the exact headwise All-to-All payload.

CP-transition latency is an alpha-beta estimate over the exact training
payload returned by :mod:`model_cp_transition`. It is kept as a bandwidth
sensitivity because no direct CP8<->CP1 transition was measured.
"""

import argparse
import csv
from pathlib import Path

from model_cp_transition import activation_bytes, model_training_path


DEFAULT_PROFILE = Path(
    "agent_run/20260816T070154Z_36b8f3172_post_refactor/"
    "dp_cp_frontier/merged.csv"
)


def parse_pattern(text):
    pattern = []
    for token in text.split(","):
        module, cp_text = token.strip().split(":", 1)
        module = module.lower()
        if module not in ("attention", "gdn"):
            raise argparse.ArgumentTypeError(f"unsupported module {module!r}")
        cp = int(cp_text)
        if cp < 1:
            raise argparse.ArgumentTypeError("CP sizes must be positive")
        pattern.append((module, cp))
    if not pattern:
        raise argparse.ArgumentTypeError("pattern cannot be empty")
    return tuple(pattern)


def parse_bandwidths(text):
    values = tuple(float(item) for item in text.split(","))
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("bandwidths must be positive GiB/s values")
    return values


def load_profile(path, seq_length):
    rows = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["seq_len"]) != seq_length:
                continue
            key = (row["module"].lower(), int(row["cp_size"]))
            rows[key] = float(row["avg_both_ms"])
    if not rows:
        raise ValueError(f"no seq_length={seq_length} rows in {path}")
    return rows


def gdn_a2a_training_bytes_per_rank(
    seq_length,
    global_batch_size,
    world_size,
    cp_size,
    bytes_per_element,
    qk_dim,
    value_dim,
    num_value_heads,
):
    if cp_size == 1:
        return 0.0
    # Forward CP->HP moves q, k, v, gate, beta and alpha. HP->CP moves
    # norm_out. Autograd executes an inverse A2A for both forward collectives.
    in_proj_dim = 2 * qk_dim + 2 * value_dim + 2 * num_value_heads
    communicated_width = in_proj_dim + value_dim
    tokens_per_rank = seq_length * global_batch_size / world_size
    forward_local_bytes = tokens_per_rank * communicated_width * bytes_per_element
    off_rank_fraction = 1.0 - 1.0 / cp_size
    return 2.0 * forward_local_bytes * off_rank_fraction


def transition_training_stats(
    cp_path,
    strategy,
    unique_bytes,
    world_size,
    alpha_us,
):
    result = model_training_path(
        path=list(cp_path),
        strategy=strategy,
        unique_bytes=unique_bytes,
        world_size=world_size,
        alpha_us=alpha_us,
        bandwidth_gbps=1.0,
    )
    total = result[-1]
    return (
        total["aggregate_network_bytes"] / world_size,
        total["messages_per_rank"],
    )


def pattern_label(pattern):
    return ",".join(f"{module}:{cp}" for module, cp in pattern)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-csv", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--seq-length", type=int, default=65536)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--bytes-per-element", type=float, default=2.0)
    parser.add_argument("--qk-dim", type=int, default=4096)
    parser.add_argument("--value-dim", type=int, default=4096)
    parser.add_argument("--num-value-heads", type=int, default=32)
    parser.add_argument(
        "--pattern",
        action="append",
        type=parse_pattern,
        default=None,
        help="repeatable module:CP path, e.g. attention:8,gdn:1,gdn:1",
    )
    parser.add_argument(
        "--transition-strategy",
        choices=("redistribute", "repeat"),
        default="redistribute",
    )
    parser.add_argument(
        "--transition-bandwidth-gib-s",
        type=parse_bandwidths,
        default=(50.0, 70.0, 80.0),
        help="comma-separated per-rank effective bandwidth sensitivity",
    )
    parser.add_argument("--alpha-us", type=float, default=0.0)
    args = parser.parse_args(argv)

    patterns = args.pattern or (
        parse_pattern("attention:8,gdn:1,gdn:1"),
        parse_pattern("attention:8,gdn:4,gdn:1"),
    )
    if args.world_size < 1 or args.global_batch_size < 1 or args.seq_length < 1:
        parser.error("world size, global batch size and sequence length must be positive")
    for pattern in patterns:
        for _, cp in pattern:
            if args.world_size % cp:
                parser.error(f"CP{cp} does not divide world_size={args.world_size}")

    profile = load_profile(args.profile_csv, args.seq_length)
    unique_bytes = activation_bytes(
        args.seq_length,
        args.global_batch_size,
        args.hidden_size,
        args.bytes_per_element,
    )
    gib = 2**30

    print(
        f"profile={args.profile_csv} seq={args.seq_length} "
        f"GBS={args.global_batch_size} W={args.world_size}"
    )
    print("GDN headwise A2A calibration (training fwd+bwd, per rank):")
    gdn_cp1_ms = profile.get(("gdn", 1))
    if gdn_cp1_ms is None:
        raise ValueError("profile must contain a GDN CP1 baseline")
    for (module, cp), measured_ms in sorted(profile.items()):
        if module != "gdn":
            continue
        network_bytes = gdn_a2a_training_bytes_per_rank(
            args.seq_length,
            args.global_batch_size,
            args.world_size,
            cp,
            args.bytes_per_element,
            args.qk_dim,
            args.value_dim,
            args.num_value_heads,
        )
        delta_ms = measured_ms - gdn_cp1_ms
        effective_bw = (
            network_bytes / gib / (delta_ms / 1000.0)
            if cp > 1 and delta_ms > 0
            else float("nan")
        )
        print(
            f"  CP{cp}: measured={measured_ms:.3f} ms, "
            f"CP1_delta={delta_ms:.3f} ms, a2a={network_bytes / gib:.3f} GiB, "
            f"effective_bw={effective_bw:.2f} GiB/s"
        )

    modeled = []
    print(f"Patterns (transition strategy={args.transition_strategy}):")
    for pattern in patterns:
        missing = [item for item in pattern if item not in profile]
        if missing:
            raise ValueError(f"missing measured module rows for {missing}")
        module_ms = sum(profile[item] for item in pattern)
        cp_path = tuple(cp for _, cp in pattern)
        transition_bytes, messages = transition_training_stats(
            cp_path,
            args.transition_strategy,
            unique_bytes,
            args.world_size,
            args.alpha_us,
        )
        estimates = []
        for bandwidth in args.transition_bandwidth_gib_s:
            transition_ms = transition_bytes / gib / bandwidth * 1000.0
            transition_ms += messages * args.alpha_us / 1000.0
            estimates.append((bandwidth, module_ms + transition_ms, transition_ms))
        modeled.append((pattern, module_ms, transition_bytes, messages, estimates))
        print(
            f"  {pattern_label(pattern)}: modules={module_ms:.3f} ms, "
            f"transition={transition_bytes / gib:.3f} GiB/rank, "
            f"messages={messages}/rank"
        )
        for bandwidth, total_ms, transition_ms in estimates:
            print(
                f"    Btr={bandwidth:.1f} GiB/s: transition={transition_ms:.3f} ms, "
                f"total={total_ms:.3f} ms"
            )

    if len(modeled) == 2:
        print("Comparison (pattern 2 relative to pattern 1):")
        for index, bandwidth in enumerate(args.transition_bandwidth_gib_s):
            first = modeled[0][-1][index][1]
            second = modeled[1][-1][index][1]
            print(
                f"  Btr={bandwidth:.1f} GiB/s: delta={second - first:.3f} ms, "
                f"ratio={second / first:.6f}x"
            )


if __name__ == "__main__":
    main()
