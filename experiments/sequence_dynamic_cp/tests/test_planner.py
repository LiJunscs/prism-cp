"""CPU invariants and independent checks for the experimental scheduler."""

import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import COEFFICIENTS
from native import replay
from planner import (
    Constraints,
    ExactCosts,
    Group,
    ProxyCost,
    export_plan,
    fingerprint,
    from_rank_ids,
    models,
    native_plan,
    pack_layout,
    search,
    select,
    to_rank_ids,
    validate,
)


def test_native_regressions():
    assert replay([(0, 6144), (1, 2048)], 2, 4096) == [[[0, 1], [0, 1]]]
    assert replay([(0, 50), (1, 50)], 14, 100) == [[[0, 1]] * 14]


def test_square_preserves_roadmap_counterexample():
    lengths = {0: 16384, **{i: 2048 for i in range(1, 129)}}
    limits = Constraints(16, 32768, alignment=32)
    costs = models(COEFFICIENTS)
    square = native_plan(lengths, limits, costs["square"])
    weighted = native_plan(lengths, limits, costs["hybrid"])
    assert len(square) == len(weighted) == 1
    assert len(next(g for g in square[0] if 0 in g.ids).ranks) == 2
    assert len(next(g for g in weighted[0] if 0 in g.ids).ranks) == 1
    assert max(sum(lengths[i] for i in g.ids) / len(g.ranks) for g in square[0]) == 32768
    assert max(sum(lengths[i] for i in g.ids) / len(g.ranks) for g in weighted[0]) == 18432


@pytest.mark.parametrize(
    "plan,reason",
    [
        (((Group((0,), (0,)), Group((1,), (0,))),), "exactly once"),
        (((Group((0,), (0,)),),), "All ranks"),
        (((Group((1, 2), (0, 1)), Group((0, 3), (2,))),), "aligned"),
        (((Group((0, 1), (0, 1, 2)),),), "capacity"),
    ],
)
def test_illegal_plans_rejected(plan, reason):
    lengths = {0: 4096, 1: 4096, 2: 4096}
    limits = Constraints(4 if reason == "aligned" else 2, 4096)
    with pytest.raises(ValueError, match=reason):
        validate(plan, lengths, limits)


@pytest.mark.parametrize("world,min_cp", [(2, 1), (4, 1), (8, 2), (16, 4)])
def test_random_complete_steps_and_search_monotonicity(world, min_cp):
    rng = random.Random(20261009 + world)
    costs = models(COEFFICIENTS)
    for _ in range(10):
        lengths = {i: rng.choice([2048, 4096, 8192, 16384, 32768]) for i in range(rng.randint(1, 40))}
        limits = Constraints(world, 16384, min_cp, alignment=2 * world)
        result = search(lengths, limits, costs["hybrid"], local_evaluations=24)
        for candidate in result["candidates"]:
            validate(candidate["plan"], lengths, limits)
            assert from_rank_ids(to_rank_ids(candidate["plan"], world)) == candidate["plan"]
        native = [c for c in result["candidates"] if c["name"] == "native_square"]
        if native:
            assert result["selected"]["proxy_cost"] <= costs["hybrid"].evaluate(native[0]["plan"], lengths) + 1e-8


def test_fallback_requires_explicit_proxy_opt_in():
    lengths = dict(enumerate([16384] + [2048] * 32))
    result = search(lengths, Constraints(8, 32768), models(COEFFICIENTS)["hybrid"], 16)
    choice, reason = select(result, lengths)
    assert choice["name"] == "native_square"
    assert reason == "native_fallback_unknown_distributed_cost"
    assert select(result, lengths, allow_proxy=True)[1] == "experimental_compute_proxy"


def test_exact_table_reverses_proxy_order_and_excludes_failed_numerics():
    lengths = dict(enumerate([16384] + [2048] * 32))
    result = search(lengths, Constraints(8, 32768), models(COEFFICIENTS)["hybrid"], 16)
    native = next(c for c in result["candidates"] if c["name"] == "native_square")
    other = next(c for c in result["candidates"] if c["fingerprint"] != native["fingerprint"])
    contract = dict(model="frozen4", backend="p2p", topology="1x8")
    data = dict(
        contract=contract,
        scope="complete_optimizer_step",
        records=[
            dict(
                fingerprint=native["fingerprint"],
                milliseconds=10,
                status="ok",
                numerical_pass=True,
            ),
            dict(
                fingerprint=other["fingerprint"],
                milliseconds=1,
                status="ok",
                numerical_pass=False,
            ),
        ],
    )
    table = ExactCosts(data, contract)
    assert select(result, lengths, table)[0] == native
    data["records"][1]["numerical_pass"] = True
    assert select(result, lengths, ExactCosts(data, contract))[0] == other
    with pytest.raises(ValueError, match="contract"):
        ExactCosts(data, dict(model="other"))
    data["records"][1]["numerical_pass"] = False
    data["records"].append(dict(data["records"][1]))
    with pytest.raises(ValueError, match="Duplicate"):
        ExactCosts(data, contract)


def test_complete_step_objective_and_geometry():
    plan = ((Group((0,), (0,)), Group((1,), (1,))), (Group((0, 1), (2,)),))
    lengths = {0: 4096, 1: 8192, 2: 16384}
    model = ProxyCost("independent", 2, 3)
    # sum_m max_g, not a single maximum over the entire step.
    expected = max(2 * 4**2 + 3 * 4, 2 * 8**2 + 3 * 8) + (2 * 16**2 + 3 * 16) / 2
    assert model.evaluate(plan, lengths) == expected
    exported = export_plan(plan, lengths, model, node_size=1)
    assert exported["compute_proxy_cost"] == expected
    assert exported["microbatches"][1]["groups"][0]["placement"] == "cross_node"


def test_sequence_cap_and_multimicrobatch_tail():
    lengths = {i: 2048 for i in range(19)}
    limits = Constraints(4, 8192, max_sequences=2)
    plan = pack_layout(lengths, limits, (1, 1, 1, 1), ProxyCost("square", 1, 0))
    validate(plan, lengths, limits)
    assert len(plan) > 1
    assert all(len(g.ids) <= 2 for batch in plan for g in batch)




def test_full_group_reaches_additive_work_lower_bound():
    lengths = dict(enumerate([65536, 32768, 8192] + [2048] * 20))
    limits = Constraints(8, 8192)
    for model in models(COEFFICIENTS).values():
        full = pack_layout(lengths, limits, (8,), model)
        lower_bound = sum(model.a * (n / 1024) ** 2 + model.b * n / 1024 for n in lengths.values()) / 8
        assert len(full) > 1
        assert model.evaluate(full, lengths) == pytest.approx(lower_bound)
        result = search(lengths, limits, model, 16)
        assert result["selected"]["proxy_cost"] == pytest.approx(lower_bound)


def test_numerical_gate_checks_complete_tensors_and_rejects_changes(tmp_path, monkeypatch):
    import torch
    from numerics import TrainingNumerics

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("SEQUENCE_CP_REFERENCE", str(tmp_path / "reference"))
    monkeypatch.setenv("SEQUENCE_CP_NUMERICS", "write")
    write = TrainingNumerics(tmp_path / "write")
    write.step = 1
    write.check("optimizer0_gradients", {"0": torch.ones(17)})
    monkeypatch.setenv("SEQUENCE_CP_NUMERICS", "read")
    read = TrainingNumerics(tmp_path / "read")
    read.step = 1
    read.check("optimizer0_gradients", {"0": torch.ones(17)})
    with pytest.raises(RuntimeError, match="tensor gate failed"):
        read.check("optimizer0_gradients", {"0": torch.full((17,), 1.1)})
    rows = [json.loads(line) for line in (tmp_path / "read/numerics_rank0.jsonl").read_text().splitlines()]
    assert [r["passed"] for r in rows] == [True, False]
    assert all(r["elements"] == 17 for r in rows)


@pytest.mark.parametrize("rank,positions", [(0, [0, 1, 6, 7]), (1, [2, 3, 4, 5])])
def test_logical_hooks_accept_embedding_keywords_and_map_native_zigzag(tmp_path, monkeypatch, rank, positions):
    import torch
    from logical_numerics import LogicalNumerics

    class Embedding(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Embedding(8, 3)

        def forward(self, input_ids, position_ids):
            return self.weight(input_ids)

    class Core(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = Embedding()
            self.decoder = torch.nn.Identity()

        def compute_language_model_loss(self, labels, logits):
            return logits.square().sum(-1)

    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setenv("SEQUENCE_CP_LOGICAL", "write")
    monkeypatch.setattr(sys.modules["__main__"], "loss_func", lambda mask, loss: (mask * loss).sum(), raising=False)
    check = LogicalNumerics(dict(plan=((Group((0, 1), (7,)),),), lengths={7: 8}), tmp_path)
    core = Core()
    check.begin([core], 1)
    tokens = torch.tensor([positions])
    y = core.embedding(input_ids=tokens, position_ids=tokens)
    loss = core.compute_language_model_loss(tokens, core.decoder(y))
    sys.modules["__main__"].loss_func(torch.ones_like(loss), loss).backward()
    assert set(check.parts) == {
        "tokens",
        "positions",
        "labels",
        "loss_mask",
        "embedding_Y",
        "decoder_Y",
        "embedding_dX",
        "token_loss",
    }
    for pieces in check.parts.values():
        assert len(pieces) == 1
        assert pieces[0]["sample_id"] == 7
        assert pieces[0]["positions"].tolist() == positions
        assert pieces[0]["value"].shape[0] == 4


def test_relaxed_cp_contract_accepts_rounding_and_rejects_data_or_large_errors(tmp_path, monkeypatch):
    import torch
    from numerical_contracts import contract, load_contract
    from numerics import TrainingNumerics

    frozen = tmp_path / "contract.json"
    frozen.write_text(json.dumps(contract("cp_bf16_v2")))
    monkeypatch.setenv("SEQUENCE_CP_NUMERICAL_CONTRACT", str(frozen))
    monkeypatch.setenv("SEQUENCE_CP_REFERENCE", str(tmp_path / "reference"))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("SEQUENCE_CP_NUMERICS", "write")
    write = TrainingNumerics(tmp_path / "write")
    write.step = 1
    for phase in ("gradients", "weights", "initial_weights"):
        write.check("optimizer0_" + phase, {"0": torch.ones(17)})
    monkeypatch.setenv("SEQUENCE_CP_NUMERICS", "read")
    read = TrainingNumerics(tmp_path / "read")
    read.step = 1
    read.check("optimizer0_gradients", {"0": torch.full((17,), 1.02)})
    read.check("optimizer0_weights", {"0": torch.full((17,), 1.001)})
    with pytest.raises(RuntimeError, match="tensor gate failed"):
        read.check("optimizer0_initial_weights", {"0": torch.full((17,), 1.000001)})
    with pytest.raises(RuntimeError, match="tensor gate failed"):
        read.check("optimizer0_gradients", {"0": torch.full((17,), 1.2)})
    with pytest.raises(RuntimeError, match="tensor gate failed"):
        read.check("optimizer0_gradients", {"0": torch.full((17,), float("nan"))})
    altered = contract("cp_bf16_v2")
    altered["optimizer"]["gradients"]["relative"] = 1.0
    frozen.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="frozen named version"):
        load_contract()
