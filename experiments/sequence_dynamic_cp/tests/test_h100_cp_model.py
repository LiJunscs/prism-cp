"""Measured H100 response, resource accounting and independent search integration."""
from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_h100_cp_profile import build
from joint_cp import Group, Layout, Model, Plan, SearchOptions, solve, validate_plan
from joint_cp.analysis import measure
from joint_cp.model import Evaluator, resource_metrics
from joint_cp.response import response_coverage


def h100():
    return Model(**json.loads((ROOT / "configs/joint_16gpu_h100.json").read_text()))


def cost(model, symbol, cp, lengths=(262144,)):
    values = dict(enumerate(lengths))
    layout = Layout((Group(tuple(range(cp)), tuple(values)),),
                    "contiguous" if symbol == "G" else "zigzag")
    return Evaluator(model, values, cp, "GGG*GGG*").operator(symbol, layout)


def test_profile_rebuild_and_four_independent_measured_anchors():
    profile = json.loads((ROOT / "configs/joint_16gpu_h100.json").read_text())
    assert build(ROOT / profile["cp_response"]["source"]) == profile
    model = Model(**profile)
    # These are independent measured values, not outputs of the model formula.
    assert cost(model, "*", 8)[0] == pytest.approx(463.3460388183594)
    assert cost(model, "*", 16)[0] == pytest.approx(305.55267333984375)
    assert cost(model, "G", 8)[0] == pytest.approx(26.383840560913086)
    assert cost(model, "G", 16)[0] == pytest.approx(14.369376182556152)
    # Operator anchors contain communication; changing an analytic link or GEMM
    # assumption must not charge communication twice or rescale the observation.
    altered = replace(model, inter_gbps=1, intra_gbps=1, compute_tflops=600)
    for symbol in ("*", "G"):
        for cp in (8, 16):
            assert cost(altered, symbol, cp)[0] == pytest.approx(cost(model, symbol, cp)[0])


def test_relative_efficiency_and_same_world_concurrent_work():
    model = h100()
    efficiencies = {s: cost(model, s, 8)[0] / (2*cost(model, s, 16)[0]) for s in ("*", "G")}
    assert 0 < efficiencies["*"] < efficiencies["G"] < 1
    lengths = {0: 262144, 1: 262144}
    groups = (Group(tuple(range(8)), (0,)), Group(tuple(range(8, 16)), (1,)))
    for symbol in ("*", "G"):
        layout = Layout(groups, "contiguous" if symbol == "G" else "zigzag")
        concurrent = Evaluator(model, lengths, 16, "GGG*GGG*").operator(symbol, layout)[0]
        serial = 2 * cost(model, symbol, 16)[0]
        assert concurrent < serial  # Two CP8 groups use the same total 16 GPUs.
        assert concurrent / serial == pytest.approx(efficiencies[symbol])
    base = resource_metrics({0:262144}, 8, "GGG*GGG*", model, 1000)
    more = resource_metrics({0:262144}, 16, "GGG*GGG*", model, 750)
    assert more["predicted_gpu_ms"] > base["predicted_gpu_ms"]
    assert more["predicted_useful_tflops_per_gpu"] < base["predicted_useful_tflops_per_gpu"]
    assert more["mfu"] is None


def test_shape_extrapolation_uses_sum_of_squares_and_keeps_memory_separate():
    model = h100()
    assert cost(model, "G", 8, (131072, 131072))[0] == pytest.approx(cost(model, "G", 8)[0])
    assert cost(model, "*", 8, (131072, 131072))[0] < cost(model, "*", 8)[0]
    for symbol in ("*", "G"):
        assert cost(model, symbol, 4)[0] == pytest.approx(2*cost(model, symbol, 8)[0])
        assert cost(model, symbol, 8)[1:] == cost(replace(model, cp_response=None), symbol, 8)[1:]
    assert cost(model, "M", 8) == cost(replace(model, cp_response=None), "M", 8)


def test_hardware_shape_topology_and_unmeasured_cp_guards():
    model = h100()
    with pytest.raises(ValueError, match="require"):
        replace(model, gdn_heads=32)
    with pytest.raises(ValueError, match="H100 CP>16"):
        cost(model, "G", 32)
    crossing = Layout((Group((0,1,2,3,8,9,10,11), (0,)),), "zigzag")
    with pytest.raises(ValueError, match="node-local"):
        Evaluator(model, {0:262144}, 16, "GGG*GGG*").operator("*", crossing)
    profile = json.loads(json.dumps(model.cp_response))
    profile["anchors"]["GDN"]["milliseconds"]["16"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        replace(model, cp_response=profile)


def test_solver_uses_h100_and_reports_full_resource_cost_and_extrapolation():
    model, lengths = h100(), {0:131072, 1:131072}
    result = solve(lengths, 16, "GGG*GGG*", model,
                   SearchOptions(beam=4, layouts=5, max_states=100000, node_affinity=True))
    assert result["plan"] and not result["state_budget_exhausted"]
    plan = Plan.from_dict(result["plan"])
    validate_plan(plan, lengths)
    value, evidence = measure(plan, lengths, model)
    assert all(row["fits"] for row in evidence)
    assert result["predicted_update_ms"] == pytest.approx(value)
    assert result["resource_metrics"]["predicted_gpu_ms"] == pytest.approx(16*value)
    coverage = response_coverage(plan, lengths, model)
    assert coverage["hardware"] == "H100"
    assert coverage["counts"]["anchor_domain_group_calls"] == 0
    assert coverage["counts"]["shape_extrapolated_group_calls"] > 0
    assert coverage["gpu_measured"] is False


def test_measured_cp32_keeps_existing_anchors_and_requires_evidence():
    profile = json.loads((ROOT / "configs/joint_32gpu_h100.json").read_text())
    response = profile["cp_response"]
    assert build(ROOT / response["source"], ROOT / response["cp32_source"]) == profile
    model = Model(**profile)
    for symbol in ("*", "G"):
        for cp in (8, 16):
            assert cost(model, symbol, cp)[0] == cost(h100(), symbol, cp)[0]
        altered = replace(model, inter_gbps=1, compute_tflops=100)
        assert cost(altered, symbol, 32)[0] == cost(model, symbol, 32)[0]
    assert cost(model, "*", 32)[0] == pytest.approx(221.34371948242188)
    with pytest.raises(ValueError, match="CP>32"):
        cost(model, "G", 64)
    missing = json.loads(json.dumps(response))
    del missing["cp32_source"]
    with pytest.raises(ValueError, match="independent measured source"):
        replace(model, cp_response=missing)


def test_cp32_solver_accepts_measured_domain_and_marks_shape_extrapolation():
    model = Model(**json.loads((ROOT / "configs/joint_32gpu_h100.json").read_text()))
    lengths = {0:524288}
    result = solve(lengths,32,"GGG*GGG*",model,SearchOptions(beam=4,layouts=8,max_states=10000))
    assert result["plan"] and all(e["fits"] for e in result["microbatch_evidence"])
    counts = result["cp_response_evidence"]["counts"]
    assert counts["cp32_gdn_calls"] == 6 and counts["cp32_gqa_calls"] == 2
    assert counts["shape_extrapolated_group_calls"] == 8
    assert counts["anchor_domain_group_calls"] == 0
    assert result["resource_metrics"]["predicted_gpu_ms"] == pytest.approx(32*result["predicted_update_ms"])
