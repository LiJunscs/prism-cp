"""Aggregate per-rank TMF 1F1B component profiles."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


LEAF_COMPONENTS = {
    "layer_forward",
    "layer_backward",
    "tmf_transition_forward",
    "tmf_transition_backward",
    "pp_p2p",
    "gradient_all_reduce",
}


def summarize_profile_dir(profile_dir):
    profile_dir = Path(profile_dir)
    files = sorted(profile_dir.glob("profile_rank_*.json"))
    if not files:
        raise FileNotFoundError(f"No profile_rank_*.json files in {profile_dir}.")

    per_rank = []
    grouped = defaultdict(lambda: {"cuda_ms": 0.0, "wall_ms": 0.0, "count": 0})
    step_times = defaultdict(list)
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rank = payload["rank"]
        stage = payload["stage"]
        per_rank.append({"rank": rank, "stage": stage, "sample_count": len(payload["samples"])})
        for sample in payload["samples"]:
            name = sample["name"]
            if name == "optimizer_step":
                step_times[(stage, rank)].append(sample["wall_ms"])
            if name not in LEAF_COMPONENTS:
                continue
            key = (
                stage,
                rank,
                name,
                sample.get("layer_type"),
                sample.get("cp_size"),
                sample.get("source_cp"),
                sample.get("destination_cp"),
                sample.get("p2p_kind"),
            )
            grouped[key]["cuda_ms"] += sample["cuda_ms"]
            grouped[key]["wall_ms"] += sample["wall_ms"]
            grouped[key]["count"] += 1

    components = []
    for key, values in grouped.items():
        (
            stage,
            rank,
            name,
            layer_type,
            cp_size,
            source_cp,
            destination_cp,
            p2p_kind,
        ) = key
        components.append(
            {
                "stage": stage,
                "rank": rank,
                "name": name,
                "layer_type": layer_type,
                "cp_size": cp_size,
                "source_cp": source_cp,
                "destination_cp": destination_cp,
                "p2p_kind": p2p_kind,
                "timing_semantics": (
                    "inclusive_autograd_hook"
                    if name in {"layer_backward", "tmf_transition_backward"}
                    else "scoped_range"
                ),
                "count": values["count"],
                "cuda_total_ms": values["cuda_ms"],
                "wall_total_ms": values["wall_ms"],
            }
        )
    components.sort(key=lambda value: value["wall_total_ms"], reverse=True)

    stage_steps = []
    for (stage, rank), values in sorted(step_times.items()):
        stage_steps.append(
            {
                "stage": stage,
                "rank": rank,
                "count": len(values),
                "wall_mean_ms": sum(values) / len(values),
                "wall_max_ms": max(values),
            }
        )

    report = {
        "schema_version": 1,
        "profile_dir": str(profile_dir),
        "rank_files": per_rank,
        "step_times": stage_steps,
        "components": components,
        "top_components": components[:20],
        "interpretation": [
            "wall time includes host/NCCL waits; CUDA time measures current-stream work.",
            "Parent ranges are omitted from component totals to avoid double counting.",
            "Grouped backward can overlap sibling layer/transition hook ranges; their totals are inclusive diagnostics and must not be added together.",
            "Use optimizer_step and cohort_backward parent ranges for non-overlapping backward phase time.",
            "Compare repeated measured steps after warmup; do not use the first cold step.",
        ],
    }
    output = profile_dir / "profile_summary.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile_dir")
    args = parser.parse_args(argv)
    report = summarize_profile_dir(args.profile_dir)
    print(f"profile ranks={len(report['rank_files'])}")
    for component in report["top_components"][:10]:
        label = component["name"]
        if component.get("layer_type"):
            label += f":{component['layer_type']}@CP{component['cp_size']}"
        if component.get("p2p_kind"):
            label += f":{component['p2p_kind']}"
        print(
            f"stage={component['stage']} rank={component['rank']} "
            f"{label} wall={component['wall_total_ms']:.3f} ms "
            f"cuda={component['cuda_total_ms']:.3f} ms n={component['count']}"
        )


if __name__ == "__main__":
    main()
