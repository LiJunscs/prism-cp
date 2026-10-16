"""Build a hardware-specific provisional model from the four H100 module points.

No fitting to scheduler outcomes, GPU execution, or modification of H800 files.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics

from joint_cp import Model
from joint_cp.response import H100_SCHEMA

ROOT = Path(__file__).resolve().parent


def build(analysis_path, cp32_analysis_path=None):
    measured = json.loads(analysis_path.read_text())
    if (measured["status"] != "all_raw_records_complete_and_valid"
            or measured["protocol"]["length"] != 262144
            or not all(r["passed"] for r in measured["cp_consistency"].values())):
        raise ValueError("Expected validated 256Ki H100 module measurements")
    anchors = {}
    for kind in ("gqa", "gdn"):
        groups = measured["points"][f"{kind}_cp8"]["groups"]
        node_ms = [groups[name]["fwd_bwd_ms"]["median"] for name in ("ranks_0_7", "ranks_8_15")]
        t8 = statistics.median(node_ms)
        t16 = measured["points"][f"{kind}_cp16"]["groups"]["ranks_0_15"]["fwd_bwd_ms"]["median"]
        anchors[kind.upper()] = dict(milliseconds={"8": t8, "16": t16}, cp8_node_medians_ms=node_ms,
            cp8_to_cp16_per_gpu_efficiency_ratio=t8/(2*t16))
    raw = Path(measured["source"]) / "raw_rank0.jsonl"
    record = next(json.loads(line) for line in raw.read_text().splitlines()
                  if json.loads(line)["event"] == "iteration")
    model = asdict(Model())
    model.update(device_bytes=record["total"], external_bytes=7*1024**3,
        provenance="H100 measured 256Ki module anchors only; node-median aggregation; all other lengths/packing and CP1/2/4 extrapolated; MLP/head/routes/update/memory remain explicit engineering assumptions, not H800 measurements")
    model["cp_response"] = dict(schema=H100_SCHEMA, name="h100_256ki_operator_anchors",
        hardware="H100", anchor_tokens=262144, anchors=anchors,
        source=str(analysis_path.resolve().relative_to(ROOT)),
        aggregation="CP8: median of the two node-wise rankmax iteration medians; CP16: rankmax iteration median",
        shape_policy="useful_work_scaling_unvalidated",
        unmeasured_cp_policy="constant_cp8_per_gpu_efficiency",
        formula="T_operator(S,Q,p) = measured_T_operator(L0,p) * useful_work(S,Q)/useful_work(L0,L0^2); CP1/2/4 use T8*8/p",
        measured_domain="B1, 256Ki, CP8 node-local or CP16 two-node; H4096 Q32/KV8/D128 GDN16x128/conv4; BF16 THD; GQA TE p2p zigzag, GDN chunkwise contiguous",
        accounting="Full module F+B anchors already include operator communication; no second communication charge; MLP and layout traffic remain separate",
        limitations=["Single-length calibration cannot identify separate linear, quadratic and fixed overhead coefficients",
                     "Useful-work shape scaling and CP1/2/4 are unvalidated extrapolations; CP>16 unsupported",
                     "MLP/head throughput and intra/inter-link bandwidths are uncalibrated analytic assumptions",
                     "Memory envelope is structural; external 7GiB and margin 2GiB are assumptions, not admission measurements",
                     "Specific FLA subkernel dispatch was not directly logged"])
    if cp32_analysis_path is not None:
        extra = json.loads(cp32_analysis_path.read_text())
        if (extra["status"] != "all_raw_records_complete_and_valid"
                or extra["protocol"]["world"] != 32 or extra["protocol"]["length"] != 262144
                or not all(r["passed"] for r in extra["cp_consistency"].values())):
            raise ValueError("Expected validated 256Ki H100 CP32 measurements")
        response = model["cp_response"]
        for kind in ("gqa", "gdn"):
            t32 = extra["points"][f"{kind}_cp32"]["groups"]["ranks_0_31"]["fwd_bwd_ms"]["median"]
            anchor = response["anchors"][kind.upper()]
            anchor["milliseconds"]["32"] = t32
            anchor["cp16_to_cp32_per_gpu_efficiency_ratio"] = anchor["milliseconds"]["16"]/(2*t32)
        response["cp32_source"] = str(cp32_analysis_path.resolve().relative_to(ROOT))
        response["cp32_measurement_summary"] = {
            kind: extra["points"][f"{kind.lower()}_cp32"]["groups"]["ranks_0_31"]["fwd_bwd_ms"]
            for kind in ("GQA", "GDN")}
        response["name"] = "h100_256ki_operator_anchors_cp32"
        response["measured_domain"] += "; CP32 four-node module F+B; CP8/16 and CP32 from separate launches"
        response["limitations"][1] = "Useful-work shape scaling and CP1/2/4 are unvalidated extrapolations; CP>32 unsupported"
        response["limitations"].append("CP32 baseline ratios use earlier CP8/16 launch; four-node packed training groups and cross-layout traffic are unmeasured")
        if "gdn_recheck" in extra:
            response["aggregation"] += "; GDN CP32 pools all initial 3 and supplemental 7 rankmax observations; no outlier removal"
            response["limitations"].append("GDN CP32 initial timings fluctuate; pooled sample range is retained, not a confidence interval; supplemental CP16 references are in the analysis, not substituted")
        model["provenance"] += "; adds independent four-node CP32 module anchors, no full-training validation"
    Model(**model)  # Validate before emitting a solver-loadable profile.
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=ROOT/"results/h100_module_cp_analysis_20261016_r1/analysis.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cp32-analysis", type=Path, help="Optional independently validated CP32 module analysis")
    args = parser.parse_args()
    model = build(args.analysis, args.cp32_analysis)
    with args.output.open("x") as handle:
        json.dump(model, handle, indent=2)
        handle.write("\n")
    print(json.dumps(model["cp_response"]["anchors"], indent=2))


if __name__ == "__main__":
    main()
