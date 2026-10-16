"""CPU-only checks of the retained CP evidence, model, solver and replay.

Reads named handoff artifacts only; no GPU imports, workspace walks or hashes.
Run with an existing Python environment and the sibling Megatron-LM checkout
(or set MEGATRON_ROOT). Does not write results or change either Git index.
"""

import csv
from dataclasses import replace
import json
import math
from pathlib import Path

from joint_cp import Model, Plan, SearchOptions, solve
from joint_cp.analysis import measure
from joint_cp.inputs import build_manifest
from joint_cp.replay import compile_replay, validate_replay
from joint_cp.response import load_response
from joint_advantage_search import audit_frozen
from native import MEGATRON_ROOT


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RESULTS = HERE / "results"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(line for line in stream if not line.startswith("#")))


def check_observations():
    observations = read_csv(ROOT / "bench_data/cp_response_observations.csv")
    by_source = {name: read_csv(ROOT / "bench_data" / name)
                 for name in {row["source"] for row in observations}}
    audit = {}
    for row in read_csv(RESULTS / "historical_cp_response_20261012/fit_validation.csv"):
        key = row["source"], row["data_row"]
        fields = {k: row[k] for k in observations[0]}
        if key in audit:
            assert audit[key] == fields, f"Inconsistent observation across profiles: {key}"
        audit[key] = fields
    assert len(observations) == len(audit) == 99
    assert {(r["source"], r["data_row"]) for r in observations} == set(audit)
    for row in observations:
        key = row["source"], row["data_row"]
        assert row == audit[key], key
        raw = by_source[row["source"]][int(row["data_row"]) - 1]
        fixed = row["source"] == "fixed_overhead.csv"
        observed = (int(raw["cp_size"]), int(raw["seq_len" if fixed else "seq_length"]),
                    int(raw["microbatch"]) if fixed else 1,
                    float(raw["avg_both_ms" if fixed else "time_ms"]))
        expected = (int(row["cp"]), int(float(row["sequence_ki"]) * 1024),
                    int(row["batch"]), float(row["time_ms"]))
        assert observed == expected, key
        operator = raw["module" if fixed else "model"]
        assert row["kind"] == {"attention": "GQA", "gated_delta_net": "GDN", "gdn": "GDN"}[operator]
    for name, count, operators in (
        ("gdn_csa_hca_scaling.csv", 12, {"gdn_chunkwise", "csa", "hca"}),
        ("csa_hca(csa128)_scaling.csv", 8, {"csa", "hca"}),
    ):
        rows = read_csv(ROOT / "analysis" / name)
        assert len(rows) == count and {r["operator"] for r in rows} == operators
        assert {(r["operator"], int(r["cp"])) for r in rows} == {
            (op, cp) for op in operators for cp in (1, 2, 4, 8)}
        assert all(float(r["median_ms"]) > 0 and float(r["peak_alloc_gib"]) > 0 for r in rows)
    return dict(fit_observations_matched=99, early_scaling_rows=20)


def check_files():
    protected = [ROOT / p for p in (
        "AGENTS.md", "README.md", "progress/CODEX_HANDOFF.md", "progress/H100_MIGRATION.md",
        "progress/ARCHIVE-20261016.md", "progress/HETEROGENEOUS_CP_MASTER_ROADMAP_20260918.md",
        "analysis/README.md", "analysis/gdn_csa_hca_scaling.csv", "analysis/csa_hca(csa128)_scaling.csv",
        "bench_data/README.md", "bench_data/cp_response_observations.csv",
        "bench_data/bench_mfu_mem_log.csv", "bench_data/fixed_overhead.csv",
        "scripts/run_manual_gdn_cp_case.sh", "scripts/torchrun_launcher.py",
    )]
    protected += [ROOT / f"bench_data/bench_merged_cp{cp}.csv" for cp in (1, 2, 4, 8)]
    protected += [HERE / p for p in (
        "handoff_check.py", "JOINT_CP.md", "configs/joint_16gpu_measured.json", "configs/joint_executor.json",
        "joint_cp/ir.py", "joint_cp/model.py", "joint_cp/response.py", "joint_cp/analysis.py",
        "joint_cp/solver.py", "joint_cp/memory_selection.py", "joint_cp/replay.py",
        "joint_cp/inputs.py", "joint_cp/adapter.py", "joint_cp/transport.py", "joint_cp/__init__.py",
        "joint_cp_search.py", "joint_cp_scale_search.py", "joint_cp_scale_audit.py",
        "native.py", "native_n_generalization_theory.py", "joint_advantage_search.py",
        "random_batch_capacity_analysis.py", "random_batch_capacity_gpu.py", "runtime.py", "gpu_smoke.py",
        "results_io.py", "two_dimensional_inputs.py", "two_dimensional_packet.py",
    )]
    for directory, names in (
        ("historical_cp_response_20261012", ("fit_statistics.csv", "fit_validation.csv", "profiles/joint.json",
                                           "profiles/single_b1.json", "profiles/fixed_g8.json")),
        ("cross_node_cp_evidence_20261012", ("paired_response.csv", "gqa_anchors.csv", "raw_rank_measurements.csv", "evidence.json")),
        ("nsys_gqa_cross_node_overlap_20261012", ("summary.csv", "layer_summary.csv", "unit_coverage.csv", "unit_coverage_audit.json")),
        ("joint_advantage_search_20261016_r1", ("analysis.json", "audit.json", "L224_S4_plans.json", "irregular_S8_plans.json", "L192_S4_plans.json")),
    ):
        protected += [RESULTS / directory / name for name in names]
    protected += [HERE.parent / "performance_model" / p for p in (
        "module_bench.py", "fit_costs.py", "cost_model.py",
        "results/cp1_20261009_p1_v2/stack_coefficients.json", "results/cp1_20261009_p1_v2/measurements.csv",
        "results/cp1_20261009_p1_v2/fit_residuals.csv", "results/cp1_20261009_p1_v2/costs.json",
        "results/cp1_20261009_p1_v2/inputs.json",
    )]
    selected = set((ROOT / "progress/PRISM_GIT_PATHS.txt").read_text().splitlines())
    for path in protected:
        assert path.is_file() and path.stat().st_size, f"Missing handoff file: {path}"
        assert path.relative_to(ROOT).as_posix() in selected, f"Not in Git selection: {path}"
    megatron_paths = (ROOT / "progress/MEGATRON_GIT_PATHS.txt").read_text().splitlines()
    assert len(megatron_paths) == 10
    for name in megatron_paths:
        assert (MEGATRON_ROOT / name).is_file(), f"Missing Megatron dependency: {name}"
    return dict(protected_prism_files=len(protected), megatron_files=len(megatron_paths))


def check_model_and_solver():
    model = Model(**json.loads((HERE / "configs/joint_16gpu_measured.json").read_text()))
    for regime in ("joint", "single_b1", "fixed_g8"):
        replace(model, cp_response=load_response(regime))
    bundle = RESULTS / "joint_advantage_search_20261016_r1"
    cases = json.loads((bundle / "analysis.json").read_text())["cases"]
    case = next(c for c in cases if c["case"] == "L224_S4")
    candidate = next(c for c in case["joint"] if c["id"] == "J_p1_packed_redistribute")
    plan = Plan.from_dict(json.loads((bundle / "L224_S4_plans.json").read_text())[candidate["id"]])
    time_ms, evidence = measure(plan, dict(enumerate(case["lengths"])), model)
    peak_gib = max(max(e["peak_bytes"]) for e in evidence) / 1024**3
    cached = candidate["scores"]["paired_linear"]
    assert math.isclose(time_ms, cached["predicted_update_ms"], rel_tol=1e-12)
    assert math.isclose(peak_gib, cached["peak_gib"], rel_tol=1e-12)
    result = solve({0: 1024, 1: 2048}, 8, "GGG*GGG*", model,
                   SearchOptions(beam=2, layouts=2, max_states=2000))
    assert result["plan"] is not None and not result["state_budget_exhausted"]
    manifest = build_manifest([1024, 2048], "handoff", 8)
    row = compile_replay(result, manifest, updates=1)[0]
    validate_replay(row, 8, "GGG*GGG*")
    return dict(response_profiles=3, cached_plan_score_matches=True,
                solver_states=result["states_examined"], replay_valid=True)


def main():
    summary = dict(files=check_files(), data=check_observations(), model=check_model_and_solver())
    frozen = audit_frozen()
    summary["frozen_plans"] = dict(native=len(frozen["native"]), joint=len(frozen["joint"]))
    print(json.dumps(dict(status="PASS", gpu_started=False, **summary), indent=2))


if __name__ == "__main__":
    main()
