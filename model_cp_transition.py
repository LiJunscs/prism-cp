"""Closed-form communication model for complete nested-CP layout paths.

The reported payload counts bytes once at the sender (aggregate network
injection), rather than adding send and receive counters.  A training pass is
modelled as the forward path plus its reverse autograd path.
"""

import argparse
import csv
import sys


def parse_path(value):
    path = [int(item) for item in value.split(",")]
    if len(path) < 2 or any(cp < 1 for cp in path):
        raise argparse.ArgumentTypeError("path must contain at least two positive CP sizes")
    for src, dst in zip(path, path[1:]):
        large, small = max(src, dst), min(src, dst)
        if large % small:
            raise argparse.ArgumentTypeError(
                f"transition {src}->{dst} is not nested"
            )
    return path


def activation_bytes(seq_length, global_batch_size, hidden_size, bytes_per_element):
    return seq_length * global_batch_size * hidden_size * bytes_per_element


def model_path(
    path,
    strategy,
    unique_bytes,
    world_size,
    alpha_us,
    bandwidth_gbps,
    initial_physical_multiplier=None,
):
    if world_size % max(path):
        raise ValueError(f"max CP={max(path)} must divide world_size={world_size}")
    if strategy not in ("redistribute", "repeat"):
        raise ValueError(f"unsupported strategy {strategy!r}")

    rows = []
    physical_multiplier = (
        float(initial_physical_multiplier)
        if initial_physical_multiplier is not None
        else (float(max(path)) / path[0] if strategy == "repeat" else 1.0)
    )
    total_bytes = 0.0
    total_messages_per_rank = 0
    total_time_us = 0.0
    for index, (src, dst) in enumerate(zip(path, path[1:])):
        ratio = max(src, dst) // min(src, dst)
        messages_per_rank = ratio - 1
        if strategy == "redistribute":
            network_bytes = unique_bytes * (1.0 - 1.0 / ratio)
            per_rank_bytes = unique_bytes / world_size * (1.0 - 1.0 / ratio)
        else:
            # A contraction creates replicas; an expansion merges them. Model both
            # directions from the physical multiplicity at the larger-replica side.
            if dst < src:
                network_bytes = unique_bytes * physical_multiplier * (ratio - 1)
                per_rank_bytes = network_bytes / world_size
                physical_after = physical_multiplier * ratio
            else:
                physical_after = physical_multiplier / ratio
                if physical_after < 1.0 - 1e-12:
                    raise ValueError(
                        "repeat expansion cannot merge below one logical copy; "
                        "the path must start from a compatible replicated layout"
                    )
                network_bytes = unique_bytes * physical_multiplier * (1.0 - 1.0 / ratio)
                per_rank_bytes = network_bytes / world_size
            physical_multiplier = physical_after

        bandwidth_bytes_per_us = bandwidth_gbps * 1e9 / 8.0 / 1e6
        beta_time_us = per_rank_bytes / bandwidth_bytes_per_us
        alpha_time_us = messages_per_rank * alpha_us
        stage_time_us = alpha_time_us + beta_time_us
        total_bytes += network_bytes
        total_messages_per_rank += messages_per_rank
        total_time_us += stage_time_us
        rows.append(
            {
                "stage": index,
                "src_cp": src,
                "dst_cp": dst,
                "ratio": ratio,
                "strategy": strategy,
                "physical_multiplier_after": physical_multiplier,
                "aggregate_network_bytes": network_bytes,
                "per_rank_network_bytes": per_rank_bytes,
                "messages_per_rank": messages_per_rank,
                "alpha_time_us": alpha_time_us,
                "beta_time_us": beta_time_us,
                "estimated_time_us": stage_time_us,
            }
        )
    return rows, {
        "aggregate_network_bytes": total_bytes,
        "messages_per_rank": total_messages_per_rank,
        "estimated_time_us": total_time_us,
        "final_physical_multiplier": physical_multiplier,
    }


def model_training_path(*args, **kwargs):
    """Model forward boundaries and the reverse boundaries created by autograd."""
    forward_rows, forward_total = model_path(*args, **kwargs)
    path = args[0] if args else kwargs["path"]
    strategy = args[1] if len(args) > 1 else kwargs["strategy"]

    reverse_kwargs = dict(kwargs)
    reverse_kwargs["initial_physical_multiplier"] = forward_total[
        "final_physical_multiplier"
    ]
    if args:
        reverse_args = list(args)
        reverse_args[0] = list(reversed(path))
        if len(reverse_args) >= 7:
            reverse_args[6] = forward_total["final_physical_multiplier"]
        reverse_rows, reverse_total = model_path(*reverse_args, **reverse_kwargs)
    else:
        reverse_kwargs["path"] = list(reversed(path))
        reverse_kwargs["strategy"] = strategy
        reverse_rows, reverse_total = model_path(**reverse_kwargs)

    total = {
        "aggregate_network_bytes": (
            forward_total["aggregate_network_bytes"]
            + reverse_total["aggregate_network_bytes"]
        ),
        "messages_per_rank": (
            forward_total["messages_per_rank"]
            + reverse_total["messages_per_rank"]
        ),
        "estimated_time_us": (
            forward_total["estimated_time_us"]
            + reverse_total["estimated_time_us"]
        ),
        "final_physical_multiplier": reverse_total["final_physical_multiplier"],
    }
    return forward_rows, forward_total, reverse_rows, reverse_total, total


def path_label(path):
    return "->".join(str(cp) for cp in path)


def print_path_report(path, strategy, unique_bytes, result):
    forward_rows, forward_total, _, reverse_total, training_total = result
    print(f"path={path_label(path)} strategy={strategy} unique={unique_bytes / 2**30:.3f} GiB")
    for row in forward_rows:
        print(
            f"  CP{row['src_cp']}->CP{row['dst_cp']}: "
            f"payload={row['aggregate_network_bytes'] / 2**30:.3f} GiB "
            f"({row['aggregate_network_bytes'] / unique_bytes:.6g} U), "
            f"messages/rank={row['messages_per_rank']}, "
            f"estimate={row['estimated_time_us']:.3f} us"
        )
    print(
        f"  forward total: payload={forward_total['aggregate_network_bytes'] / 2**30:.3f} GiB "
        f"({forward_total['aggregate_network_bytes'] / unique_bytes:.6g} U), "
        f"messages/rank={forward_total['messages_per_rank']}, "
        f"estimate={forward_total['estimated_time_us']:.3f} us"
    )
    print(
        f"  backward total: payload={reverse_total['aggregate_network_bytes'] / 2**30:.3f} GiB "
        f"({reverse_total['aggregate_network_bytes'] / unique_bytes:.6g} U), "
        f"messages/rank={reverse_total['messages_per_rank']}, "
        f"estimate={reverse_total['estimated_time_us']:.3f} us"
    )
    print(
        f"  training total: payload={training_total['aggregate_network_bytes'] / 2**30:.3f} GiB "
        f"({training_total['aggregate_network_bytes'] / unique_bytes:.6g} U), "
        f"messages/rank={training_total['messages_per_rank']}, "
        f"estimate={training_total['estimated_time_us']:.3f} us"
    )


def print_comparison(base_path, compare_path, unique_bytes, base, compare):
    base_total = base[-1]
    compare_total = compare[-1]
    base_bytes = base_total["aggregate_network_bytes"]
    compare_bytes = compare_total["aggregate_network_bytes"]
    print(
        f"comparison ({path_label(compare_path)} vs {path_label(base_path)}): "
        f"training payload ratio={compare_bytes / base_bytes:.6f}x, "
        f"reduction={(1.0 - compare_bytes / base_bytes) * 100.0:.3f}%, "
        f"saved={(base_bytes - compare_bytes) / unique_bytes:.6g} U"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=parse_path, required=True, help="e.g. 8,4,1")
    parser.add_argument(
        "--compare-path",
        type=parse_path,
        default=None,
        help="optional second complete CP path, e.g. 8,4,1,4,8",
    )
    parser.add_argument("--strategy", choices=["redistribute", "repeat"], required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--seq-length", type=int, default=65536)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--bytes-per-element", type=float, default=2.0)
    parser.add_argument("--alpha-us", type=float, default=3.0)
    parser.add_argument("--bandwidth-gbps", type=float, default=400.0)
    parser.add_argument(
        "--initial-physical-multiplier",
        type=float,
        default=None,
        help=(
            "initial repeat replication; default is max(path)/path[0], which "
            "models both contraction and reverse merge paths"
        ),
    )
    parser.add_argument("--csv", action="store_true")
    args = parser.parse_args(argv)
    if args.world_size < 1 or args.bandwidth_gbps <= 0 or args.alpha_us < 0:
        parser.error("world size and bandwidth must be positive; alpha must be non-negative")

    unique_bytes = activation_bytes(
        args.seq_length,
        args.global_batch_size,
        args.hidden_size,
        args.bytes_per_element,
    )
    model_kwargs = dict(
        strategy=args.strategy,
        unique_bytes=unique_bytes,
        world_size=args.world_size,
        alpha_us=args.alpha_us,
        bandwidth_gbps=args.bandwidth_gbps,
        initial_physical_multiplier=args.initial_physical_multiplier,
    )
    result = model_training_path(path=args.path, **model_kwargs)
    if args.csv:
        all_rows = []
        paths_and_results = [(args.path, result)]
        if args.compare_path is not None:
            paths_and_results.append(
                (
                    args.compare_path,
                    model_training_path(path=args.compare_path, **model_kwargs),
                )
            )
        for path, path_result in paths_and_results:
            forward_rows, _, backward_rows, _, _ = path_result
            for direction, rows in (("forward", forward_rows), ("backward", backward_rows)):
                for row in rows:
                    all_rows.append(
                        {"path": path_label(path), "direction": direction, **row}
                    )
        writer = csv.DictWriter(sys.stdout, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
        return

    print_path_report(args.path, args.strategy, unique_bytes, result)
    if args.compare_path is not None:
        compare_result = model_training_path(path=args.compare_path, **model_kwargs)
        print_path_report(
            args.compare_path, args.strategy, unique_bytes, compare_result
        )
        print_comparison(
            args.path, args.compare_path, unique_bytes, result, compare_result
        )


if __name__ == "__main__":
    main()
