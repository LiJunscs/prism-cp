"""CPU regression for large CP1 packs with negligible time benefit."""

from dataclasses import replace
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from joint_cp import Group, Layout, Microbatch, Model, Plan, validate_plan
from joint_cp.memory_selection import prefer_lower_memory
from joint_cp.model import Evaluator


def fixture():
    pattern, world = "GGG*GGG*", 16
    lengths = {0: 262144, **{i: 4096 for i in range(1, 194)}}
    ingress = Layout((Group(tuple(range(world)), (0, 1)),), "zigzag")
    contiguous = ingress.with_mode("contiguous")
    gqa = Layout((Group(tuple(range(8)), (0,)), Group(tuple(range(8, 16)), (1,))), "zigzag")
    heavy = Microbatch(ingress, tuple((contiguous if c == "G" else gqa, contiguous) for c in pattern))
    batches = [heavy]
    for start in (2, 98):
        layout = Layout(tuple(Group((r,), tuple(range(start + 6*r, start + 6*r + 6))) for r in range(world)), "zigzag")
        batches.append(Microbatch(layout, tuple(
            (layout.with_mode("contiguous") if c == "G" else layout, layout.with_mode("contiguous")) for c in pattern)))
    return Plan(world, pattern, tuple(batches)), lengths, Model()


def owners(plan):
    return [[s for b in plan.microbatches[1:] for s in b.ingress.rank_ids(plan.world)[r]] for r in range(plan.world)]


def test_memory_tie_break_preserves_distributed_work_and_rank_residency():
    original, lengths, model = fixture()
    plan, evidence = prefer_lower_memory(original, lengths, model)
    validate_plan(plan, lengths)
    assert plan.microbatches[0] == original.microbatches[0]
    assert owners(plan) == owners(original)
    assert len(plan.microbatches) == 4  # 2 x 96 shorts -> 3 x 64 shorts.
    assert evidence["repacked_families"][0]["tokens_per_rank"] == 16384
    assert evidence["selected_peak_bytes"] < evidence["original_peak_bytes"]
    heavy_peak = max(Evaluator(model, lengths, 16, plan.pattern).evaluate(plan.microbatches[0])["peak_bytes"])
    assert evidence["selected_peak_bytes"] == heavy_peak
    assert evidence["selected_predicted_update_ms"] - evidence["original_predicted_update_ms"] == pytest.approx(.1)
    assert evidence["selected_predicted_update_ms"] <= evidence["original_predicted_update_ms"] * 1.001
    for batch in plan.microbatches:
        for symbol, (attention, mlp) in zip(plan.pattern, batch.layers):
            if symbol == "G":
                assert attention == mlp


def test_time_budget_rejects_slow_repacking_and_small_packs_stay_unchanged():
    original, lengths, model = fixture()
    assert prefer_lower_memory(original, lengths, model, 0)[0] == original
    assert prefer_lower_memory(original, lengths, replace(model, microbatch_ms=1000))[0] == original
    selected, _ = prefer_lower_memory(original, lengths, model)
    assert prefer_lower_memory(selected, lengths, model)[0] == selected
    with pytest.raises(ValueError, match="tolerance"):
        prefer_lower_memory(original, lengths, model, float("nan"))


def test_catalogue_audit_reports_final_memory_selected_plan(monkeypatch):
    import joint_cp_scale_audit as audit
    from joint_cp.response import load_response
    original, lengths, model = fixture()
    model = replace(model, cp_response=load_response())
    monkeypatch.setitem(audit.CASES, "memory_regression", (1, 262144, 193, 4096))
    monkeypatch.setitem(audit.SCENARIOS, "memory_regression", (16, None, 1., None))
    catalogue = {}
    for batch in original.microbatches:
        audit.add_template(catalogue, batch, lengths, 262144, 4096, 16, "regression")
    result, _, _ = audit.audit(list(catalogue.values()), "memory_regression", "memory_regression", model)
    plan = Plan.from_dict(result["plan"])
    validate_plan(plan, lengths)
    assert len(plan.microbatches) == 4
    assert result["memory_selection"]["changed"]
    assert "optimal_packing" not in result["status"]
    evaluator = Evaluator(model, lengths, 16, plan.pattern)
    actual = [evaluator.evaluate(b) for b in plan.microbatches]
    assert result["microbatch_evidence"] == actual
    assert result["predicted_update_ms"] == pytest.approx(sum(r["predicted_ms"] for r in actual) + model.update_ms)
