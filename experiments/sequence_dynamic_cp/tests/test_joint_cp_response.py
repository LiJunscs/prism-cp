"""Profiling accounting, topology transfer, and balanced placement coverage."""

from dataclasses import asdict
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from joint_cp import Group, Layout, Microbatch, Model, Plan, SearchOptions, solve, validate_plan
from joint_cp.model import Evaluator
from joint_cp.response import load_response


def group_time(model, symbol, cp, values):
    layout = Layout((Group(tuple(range(cp)), tuple(range(len(values)))),), "zigzag")
    return Evaluator(model, dict(enumerate(values)), cp, "G*").operator(symbol, layout)


def test_node_response_replaces_communication_once_per_packed_call_and_preserves_memory():
    m = Model(cp_response=load_response(cp16="carry8"))
    p = m.cp_response["fits"]["GDN"]["parameters"]["8"]
    for values in ([65536], [4096]*16):
        # Same linear work and one invocation: intercept is NOT multiplied by N.
        cp1 = group_time(Model(), "G", 1, values)[0]
        row = group_time(m, "G", 8, values)
        assert row[0] == pytest.approx(cp1/8*(1+p["token_fraction"])+p["fixed_ms"])
        assert row[1:] == group_time(Model(), "G", 8, values)[1:]
    assert group_time(m, "G", 1, [65536]) == group_time(Model(), "G", 1, [65536])
    assert group_time(m, "M", 8, [65536]) == group_time(Model(), "M", 8, [65536])


@pytest.mark.parametrize("policy", ["paired_linear", "paired_quadratic"])
def test_cp16_reproduces_paired_ratios_without_extra_node_or_link_charge(policy):
    m = Model(cp_response=load_response(cp16=policy))
    for pair in m.cp_response["paired_anchors"]:
        n = pair["length"]
        assert group_time(m, "*", 16, [n])[0]/group_time(m, "*", 8, [n])[0] == pytest.approx(pair["ratio"])
    # GDN does not inherit the GQA ratio or its packed interpolation policy.
    carry = Model(cp_response=load_response(cp16="carry8"))
    assert group_time(m, "G", 16, [65536,4096]) == group_time(carry, "G", 16, [65536,4096])
    p = m.cp_response["fits"]["GDN"]["parameters"]["8"]
    ideal = group_time(Model(), "G", 1, [65536,4096])[0]/16
    state_bytes = 2*15*2*(4*m.gdn_heads*m.gdn_dim**2+18*m.gdn_heads*m.gdn_dim)
    delta = state_bytes*(1/m.inter_gbps-1/m.intra_gbps)/1e6
    assert group_time(m, "G", 16, [65536,4096])[0] == pytest.approx(ideal*(1+p["token_fraction"])+p["fixed_ms"]+delta)


def test_profile_roundtrip_gqa_control_and_domain_guard():
    m = Model(cp_response=load_response(gdn=False))
    restored = Model(**json.loads(json.dumps(asdict(m))))
    assert group_time(m, "*", 8, [65536]) == group_time(restored, "*", 8, [65536])
    assert group_time(m, "G", 8, [65536]) == group_time(Model(), "G", 8, [65536])
    with pytest.raises(ValueError, match="requires"):
        Model(hidden=2048, cp_response=load_response())


def test_balanced_candidates_cover_heterogeneous_cp_groups():
    lengths = dict(enumerate([32768]+[512]*448))
    result = solve(lengths, 16, "GGG*GGG*", Model(), SearchOptions(
        beam=16, layouts=12, node_affinity=True,
        homogeneous_packing=True, balanced_packing=True, max_states=2000000))
    validate_plan(Plan.from_dict(result["plan"]), lengths)
    # The final memory tie-break may spend its declared time tolerance.
    assert result["predicted_update_ms"] <= 729.9358388224 * 1.001 + 1e-8
    assert not result["state_budget_exhausted"]
