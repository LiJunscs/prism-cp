"""Targeted CPU correctness gates; these are not GPU kernel admission."""

from dataclasses import replace
from itertools import product
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from joint_cp import Group, Layout, Microbatch, Model, Plan, SearchOptions, solve, validate_plan
from joint_cp.ir import route, spans, validate_layout
from joint_cp.model import Evaluator
from joint_cp.replay import compile_replay, validate_replay
from joint_cp.solver import SearchLimit, layout_candidates


def tiny_model(**kwargs):
    return Model(hidden=4, ffn=8, vocab=16, q_heads=2, kv_heads=1, head_dim=2,
                 gdn_heads=1, gdn_dim=2, static_bytes=0, external_bytes=0, margin_bytes=0,
                 **kwargs)


def labels(layout, lengths, world):
    return [[(sample, start + k) for sample, start, _, count in row for k in range(count)]
            for row in spans(layout, lengths, world)]


def test_routes_cover_positions_across_group_changes_and_modes():
    lengths = {0: 16, 1: 8, 2: 8}
    layouts = layout_candidates(lengths, lengths, 4, True, 100)
    for source in layouts:
        for target in layouts:
            target = target.with_mode("contiguous")
            original, expected = labels(source, lengths, 4), labels(target, lengths, 4)
            routed = [[None] * len(row) for row in expected]
            for src, dst, begin, end, count in route(source, target, lengths, 4):
                assert all(v is None for v in routed[dst][end:end + count])
                routed[dst][end:end + count] = original[src][begin:begin + count]
            assert routed == expected


def test_exact_joint_matches_independent_bruteforce_over_packings_and_layer_paths():
    lengths, world, pattern, model = {0: 8, 1: 4}, 2, "G*", tiny_model()
    optimum = float("inf")
    # All two-sample microbatch set partitions, both rank assignments and every
    # attention/MLP path. No solver DP or beam machinery used by the oracle.
    for packing in (((0, 1),), ((0,), (1,))):
        total = model.update_ms
        for ids in packing:
            layouts = layout_candidates(ids, lengths, world, True, 100)
            mlps = tuple(dict.fromkeys(x.with_mode(m) for x in layouts for m in ("zigzag", "contiguous")))
            best = float("inf")
            for ingress, first, second, mlp_a, mlp_b in product(layouts, layouts, layouts, mlps, mlps):
                batch = Microbatch(ingress, ((first.with_mode("contiguous"), mlp_a), (second, mlp_b)))
                row = Evaluator(model, lengths, world, pattern).evaluate(batch)
                if row["fits"]:
                    best = min(best, row["predicted_ms"])
            total += best
        optimum = min(optimum, total)
    result = solve(lengths, world, pattern, model, SearchOptions(mode="exact"))
    assert result["status"] == "optimal_in_declared_domain"
    assert result["predicted_update_ms"] == pytest.approx(optimum)
    plan = Plan.from_dict(json.loads(json.dumps(result["plan"])))
    validate_plan(plan, lengths)


def test_capacity_is_derived_not_24ki_and_whole_backward_memory_rejects():
    lengths = {0: 65536}
    group = (Group((0, 1), (0,)),)
    ingress, attention = Layout(group, "zigzag"), Layout(group, "contiguous")
    batch = Microbatch(ingress, ((attention, attention),) * 8)
    large = tiny_model()
    result = solve(lengths, 2, "G*", large, SearchOptions(mode="exact"))
    assert result["plan"] and max(Plan.from_dict(result["plan"]).microbatches[0].ingress.tokens(lengths, 2)) > 24576
    ev = Evaluator(large, lengths, 2, "G" * 8)
    row = ev.evaluate(batch)
    # Peak contains accumulated activations from earlier layers, not max of
    # independent layer footprints, and includes corresponding backward workspace.
    assert row["points"][7]["saved_live_bytes"][0] > 7 * row["points"][0]["saved_live_bytes"][0]
    peak = max(row["peak_bytes"])
    assert peak > max(row["points"][0]["peak_bytes"])
    # Device reservation is 8GiB, so this creates a precise just-too-small budget.
    low = replace(large, device_bytes=8 * 1024**3 + peak - 1)
    assert not Evaluator(low, lengths, 2, "G" * 8).evaluate(batch)["fits"]


def test_joint_does_not_pin_gqa_or_gdn_to_ingress(monkeypatch):
    import joint_cp.solver as solver
    class Synthetic(Evaluator):
        """A controlled global tradeoff: GQA and GDN require different groups."""
        def operator(self, symbol, layout):
            cp = len(layout.groups[0].ranks)
            # GDN favors CP1; GQA favors CP2; transfer is cheap but nonzero.
            cost = {"G": {1: 1., 2: 9.}, "*": {1: 9., 2: 1.}, "M": {1: 1., 2: 1.}}[symbol][cp]
            return cost, (1, 1), (0, 0)
        def transfer(self, source, target):
            return (0. if source == target else .01), (0, 0)
    monkeypatch.setattr(solver, "Evaluator", Synthetic)
    result = solve({0: 8, 1: 8}, 2, "G*", tiny_model(), SearchOptions(mode="exact"))
    assert result["predicted_update_ms"] < 6.0
    batch = Plan.from_dict(result["plan"]).microbatches[0]
    assert len(batch.layers[0][0].groups[0].ranks) == 1
    assert len(batch.layers[1][0].groups[0].ranks) == 2


def test_replay_preserves_joint_microbatches_and_rejects_tampering():
    lengths = {0: 8, 1: 4}
    result = solve(lengths, 2, "G*", tiny_model(), SearchOptions(mode="exact"))
    manifest = dict(samples=[dict(sample_id=17, length=8), dict(sample_id=23, length=4)],
                    input_tokens=12, loss_tokens=6, case_id="test", seed=3)
    rows = compile_replay(result, manifest)
    for row in rows:
        validate_replay(row, 2, "G*")
        assert row["logical_sample_ids"] == [17, 23]
        assert row["capacity_semantics"] == "derived_ingress_allocation_bound"
    rows[0]["capacity_tokens"] += 1
    with pytest.raises(ValueError, match="changed"):
        validate_replay(rows[0], 2)


def test_bounded_plan_is_complete_and_never_claims_optimal():
    lengths, model = {i: 32 for i in range(8)}, tiny_model()
    result = solve(lengths, 2, "G*", model)
    validate_plan(Plan.from_dict(result["plan"]), lengths)
    assert result["status"] == "best_found_no_optimality_certificate"
    exhausted = solve(lengths, 2, "G*", model, SearchOptions(max_states=1))
    assert exhausted["status"] == "search_budget_exhausted"
    assert exhausted["plan"] is None
    with pytest.raises(SearchLimit):
        solve({0: 8, 1: 8}, 2, "G*", model, SearchOptions(mode="exact", max_states=1))


def test_invalid_or_incomplete_ownership_is_rejected():
    with pytest.raises(ValueError, match="alignment"):
        validate_layout(Layout((Group((0, 1), (0,)),), "zigzag"), {0: 7}, 2)
    with pytest.raises(ValueError, match="exactly once"):
        validate_plan(Plan(2, "G", ()), {0: 8})


def test_bounded_budget_keeps_only_completed_incumbent(monkeypatch):
    from joint_cp.solver import _Search
    lengths, model = {0: 8, 1: 4}, tiny_model()
    batch = _Search.batch
    completed = {}
    def stop_after_complete_packing(search, ids):
        if search.incumbent is not None:
            completed["score"] = search.incumbent[0] + model.update_ms
            raise SearchLimit("Controlled interruption after a complete packing")
        return batch(search, ids)
    monkeypatch.setattr(_Search, "batch", stop_after_complete_packing)
    result = solve(lengths, 2, "G*", model, SearchOptions(homogeneous_packing=True))
    assert result["state_budget_exhausted"]
    assert result["status"] == "best_found_no_optimality_certificate"
    assert result["predicted_update_ms"] == completed["score"]
    validate_plan(Plan.from_dict(result["plan"]), lengths)


def test_node_affine_cp_changes_preserve_sample_node_and_remove_remote_routes():
    lengths = dict(enumerate([32768] + [512] * 448))
    def uniform(affine):
        layouts = layout_candidates(lengths, lengths, 16, False, 6, 8 if affine else None)
        return {len(x.groups[0].ranks): x for x in layouts
                if len({len(g.ranks) for g in x.groups}) == 1}
    original, local = uniform(False), uniform(True)
    def remote_tokens(layouts):
        return sum(count for src, dst, _, _, count in
                   route(layouts[2].with_mode("contiguous"), layouts[8], lengths, 16)
                   if src // 8 != dst // 8)
    assert remote_tokens(original) > 0
    assert remote_tokens(local) == 0
    residency = [{s: g.ranks[0] // 8 for g in local[cp].groups for s in g.samples}
                 for cp in (1, 2, 4, 8)]
    assert all(nodes == residency[0] for nodes in residency)
    for cp in (2, 4, 8):
        assert max(local[cp].tokens(lengths, 16)) == 16384


def test_homogeneous_packing_remains_a_joint_search_candidate():
    # The complete pool may separate lengths; mixed packs are not mandatory.
    from joint_cp.solver import _Search
    lengths = dict(enumerate([196608] + [2048] * 64))
    model = Model()
    options = SearchOptions(beam=16, layouts=6, homogeneous_packing=True)
    separate = _Search(lengths, 16, "GGG*GGG*", model, options)
    reference = separate.batch((0,))[0] + separate.batch(tuple(range(1, 65)))[0] + model.update_ms
    result = solve(lengths, 16, "GGG*GGG*", model, options)
    assert result["predicted_update_ms"] <= reference * (1 + options.memory_time_tolerance) + 1e-8
    validate_plan(Plan.from_dict(result["plan"]), lengths)


def test_default_model_can_select_different_cp_with_topology_aware_candidates():
    lengths = dict(enumerate([32768] + [512] * 448))
    result = solve(lengths, 16, "GGG*GGG*", Model(), SearchOptions(
        beam=16, layouts=6, node_affinity=True, homogeneous_packing=True))
    assert not result["state_budget_exhausted"]
    plan = Plan.from_dict(result["plan"])
    validate_plan(plan, lengths)
    assert any(len({a.sample_cp() for a, _ in batch.layers}) > 1 for batch in plan.microbatches)
