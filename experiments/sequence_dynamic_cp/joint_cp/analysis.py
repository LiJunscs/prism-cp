"""Plan cost accounting shared by current joint search tools; no GPU runs."""

from collections import Counter
import math

from .ir import Plan, validate_plan
from .model import Evaluator, resource_metrics


def describe(result, lengths, model):
    """Compact auditable stage accounting, with whole-plan scores kept separate."""
    if result["plan"] is None:
        return dict(status=result["status"])
    plan = Plan.from_dict(result["plan"])
    ev = Evaluator(model, lengths, plan.world, plan.pattern)
    costs = dict(gdn_ms=0., gqa_ms=0., mlp_ms=0., routes_ms=0., head_and_mb_ms=0.,
                 update_ms=model.update_ms)
    batches = []
    for batch in plan.microbatches:
        previous = batch.ingress
        layers = []
        for symbol, (attention, mlp) in zip(plan.pattern, batch.layers):
            costs["gdn_ms" if symbol == "G" else "gqa_ms"] += ev.operator(symbol, attention)[0]
            costs["mlp_ms"] += ev.operator("M", mlp)[0]
            costs["routes_ms"] += ev.transfer(previous, attention)[0] + 2 * ev.transfer(attention, mlp)[0]
            counts = Counter((lengths[s], cp) for s, cp in attention.sample_cp())
            layers.append(dict(operator="GDN" if symbol == "G" else "GQA",
                cp_by_length=[dict(length=n, cp=cp, samples=count) for (n, cp), count in sorted(counts.items())],
                max_attention_tokens=max(attention.tokens(lengths, plan.world)),
                max_mlp_tokens=max(mlp.tokens(lengths, plan.world))))
            previous = attention
        head, _ = ev.finish(ev.initial(batch.ingress), previous, batch.ingress)
        final_route = ev.transfer(previous, batch.ingress)[0]
        costs["routes_ms"] += final_route
        costs["head_and_mb_ms"] += head - final_route
        ids = [s for g in batch.ingress.groups for s in g.samples]
        batches.append(dict(samples=len(ids), input_tokens=sum(lengths[s] for s in ids), layers=layers,
            changes_cp=len({a.sample_cp() for a, _ in batch.layers}) > 1))
    predicted = result["predicted_update_ms"]
    if not math.isclose(sum(costs.values()), predicted, rel_tol=1e-10, abs_tol=1e-8):
        raise AssertionError("Stage breakdown does not match the complete plan score")
    return dict(status=result["status"], predicted_update_ms=predicted,
        resource_metrics=resource_metrics(lengths, plan.world, plan.pattern, model, predicted),
        microbatches=len(batches), changed_cp_microbatches=sum(b["changes_cp"] for b in batches),
        peak_gib=max(max(e["peak_bytes"]) for e in result["microbatch_evidence"]) / 1024**3,
        stage_costs=costs, batches=batches, state_budget_exhausted=result["state_budget_exhausted"],
        states_examined=result["states_examined"])


def measure(plan, lengths, model):
    validate_plan(plan, lengths)
    ev = Evaluator(model, lengths, plan.world, plan.pattern)
    evidence = [ev.evaluate(b) for b in plan.microbatches]
    return sum(e["predicted_ms"] for e in evidence)+model.update_ms, evidence
