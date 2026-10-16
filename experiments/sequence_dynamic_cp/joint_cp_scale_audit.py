"""Repack a shared catalogue and re-score it under explicit CP32 hypotheses.

The count DP is exact only over the saved microbatch catalogue. Layouts and
layer paths remain bounded candidates; this is not a global optimality proof.
Every catalogue entry is re-evaluated, including complete F/B memory, and
the joint planner may freely repartition the entire input pool.
"""

import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

from joint_cp import Group, Layout, Microbatch, Model, Plan, SearchOptions, validate_plan
from joint_cp.model import Evaluator
from joint_cp.memory_selection import prefer_lower_memory
from joint_cp.response import load_response, response_coverage
from joint_cp.solver import _Search, layout_candidates
from joint_cp_scale_search import CASES, SCENARIOS
from joint_cp.analysis import describe

PATTERN = "GGG*GGG*"


def remap_batch(batch, mapping):
    def remap(layout):
        return Layout(tuple(Group(g.ranks, tuple(mapping[s] for s in g.samples))
                            for g in layout.groups), layout.mode)
    return Microbatch(remap(batch.ingress), tuple((remap(a), remap(m)) for a, m in batch.layers))


def add_template(catalogue, batch, lengths, long, short, world, source):
    ids = sorted(s for g in batch.ingress.groups for s in g.samples)
    if any(lengths[s] not in (long, short) for s in ids):
        return
    # All inputs in this experiment use ascending long IDs then short IDs.
    # Preserve order within each length class when replacing sample identities.
    ids = sorted(ids, key=lambda s: (-lengths[s], s))
    batch = remap_batch(batch, {s: i for i, s in enumerate(ids)})
    values = [lengths[s] for s in ids]
    plan = Plan(world, PATTERN, (batch,))
    validate_plan(plan, dict(enumerate(values)))
    # Equal ownership IR with different sample lengths is a different shape.
    key = json.dumps([values, plan.to_dict()], separators=(",", ":"))
    if key not in catalogue:
        catalogue[key] = dict(plan=plan.to_dict(), long_samples=values.count(long),
                              short_samples=values.count(short), lengths=values, source=source)


def count_dp(rows, long_count, short_count):
    """Min-cost complete packing; rows are (long_count, short_count, cost, id)."""
    costs = [[math.inf]*(short_count+1) for _ in range(long_count+1)]
    previous = [[None]*(short_count+1) for _ in range(long_count+1)]
    costs[0][0] = 0.
    for a in range(long_count+1):
        for b in range(short_count+1):
            if not math.isfinite(costs[a][b]):
                continue
            for h, s, cost, index in rows:
                if h+s <= 0:
                    raise ValueError("Empty template")
                x, y = a+h, b+s
                if x <= long_count and y <= short_count and costs[a][b]+cost < costs[x][y]:
                    costs[x][y] = costs[a][b]+cost
                    previous[x][y] = (h, s, index)
    if not math.isfinite(costs[-1][-1]):
        raise ValueError("Catalogue cannot cover the complete input")
    chosen, a, b = [], long_count, short_count
    while a or b:
        h, s, index = previous[a][b]
        chosen.append(index)
        a, b = a-h, b-s
    return costs[-1][-1], list(reversed(chosen))


def audit(catalogue, case, scenario, model_override=None, memory_time_tolerance=0.001):
    hcount, long, scount, short = CASES[case]
    world, policy, multiplier, cap = SCENARIOS[scenario]
    model = model_override or Model(cp_response=load_response(cp32=policy, gdn32_multiplier=multiplier))
    candidates = {}
    for index, row in enumerate(catalogue):
        plan = Plan.from_dict(row["plan"])
        if plan.world != world:
            raise ValueError("Catalogue world mismatch")
        batch = plan.microbatches[0]
        if cap and any(len(g.ranks)>cap for a, _ in batch.layers for g in a.groups):
            continue
        h, s = row["long_samples"], row["short_samples"]
        lengths = dict(enumerate([long]*h+[short]*s))
        ev = Evaluator(model, lengths, world, PATTERN).evaluate(batch)
        if not ev["fits"]:
            continue
        old = candidates.get((h, s))
        if old is None or ev["predicted_ms"] < old[2]:
            candidates[h, s] = (h, s, ev["predicted_ms"], index)
    values = [long]*hcount+[short]*scount
    lengths = dict(enumerate(values))
    score, indices = count_dp(list(candidates.values()), hcount, scount)
    batches, next_h, next_s = [], 0, hcount
    for index in indices:
        row = catalogue[index]
        h, s = row["long_samples"], row["short_samples"]
        ids = list(range(next_h, next_h+h))+list(range(next_s, next_s+s))
        batches.append(remap_batch(Plan.from_dict(row["plan"]).microbatches[0], dict(enumerate(ids))))
        next_h, next_s = next_h+h, next_s+s
    plan = Plan(world, PATTERN, tuple(batches))
    validate_plan(plan, lengths)
    ev = Evaluator(model, lengths, world, PATTERN)
    checks = [ev.evaluate(b) for b in batches]
    assert all(e["fits"] for e in checks)
    assert math.isclose(sum(e["predicted_ms"] for e in checks), score, rel_tol=1e-12)
    plan, memory_selection = prefer_lower_memory(plan, lengths, model, memory_time_tolerance)
    validate_plan(plan, lengths)
    checks = [ev.evaluate(b) for b in plan.microbatches]
    score = sum(e["predicted_ms"] for e in checks)+model.update_ms
    histogram = [dict(template=i, microbatches=n) for i, n in sorted(Counter(indices).items())]
    result = dict(case=case, scenario=scenario, world=world, input_tokens=sum(values), input_samples=len(values),
        model=asdict(model), plan=plan.to_dict(), predicted_update_ms=score,
        microbatch_evidence=checks, status="catalogue_incumbent_with_bounded_memory_selection_no_global_certificate",
        memory_selection=memory_selection,
        state_budget_exhausted=False, states_examined=None, gpu_admitted=False,
        grouping_constraint=False, hypothetical_hardware=world==32,
        catalogue_templates=len(catalogue), two_templates=len(candidates),
        selected_templates=histogram,
        selected_templates_scope="catalogue DP before memory selection; final packing is in plan",
        cp_response_evidence=response_coverage(plan, lengths, model))
    analysis = dict(two=describe(result,lengths,model))
    summary = {k:result[k] for k in ("case","scenario","world","input_tokens","input_samples","catalogue_templates","two_templates")}
    summary.update({k:v for k,v in analysis["two"].items() if k not in ("batches","stage_costs")})
    summary["cp_calls"] = result["cp_response_evidence"]["counts"]
    return result, analysis, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument("--catalogues", type=Path, nargs="*", default=[], help="Previously saved template catalogues")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world", type=int, choices=(16,32), required=True)
    parser.add_argument("--cases", nargs="+", choices=CASES, required=True)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, required=True)
    parser.add_argument("--expand", action="store_true", help="Add bounded shape searches before the count DP")
    parser.add_argument("--memory-time-tolerance", type=float, default=0.001,
                        help="Relative modeled update-time allowance for bounded CP1 memory selection")
    parser.add_argument("--constant-cp-controls", action="store_true",
                        help="Include fixed-attention-CP controls, including full-world CP32")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    long, short = CASES[args.cases[0]][1::2]
    assert all(CASES[c][1::2]==(long,short) for c in args.cases)
    assert all(SCENARIOS[s][0]==args.world for s in args.scenarios)
    catalogue = {}
    for path in args.catalogues:
        for row in json.loads(path.read_text()):
            plan = Plan.from_dict(row["plan"])
            if plan.world != args.world or plan.pattern != PATTERN:
                raise ValueError("Imported catalogue domain mismatch")
            lengths = dict(enumerate([long]*row["long_samples"]+[short]*row["short_samples"]))
            if "lengths" in row and row["lengths"] != list(lengths.values()):
                raise ValueError("Imported catalogue lengths mismatch")
            add_template(catalogue,plan.microbatches[0],lengths,long,short,args.world,row["source"])
    for root in args.sources:
        for path in sorted(root.glob("*/*_result.json")):
            row = json.loads(path.read_text())
            lengths = dict(enumerate(json.loads((path.parent/"input.json").read_text())["lengths"]))
            if row.get("plan"):
                plan = Plan.from_dict(row["plan"])
                if plan.world == args.world:
                    for batch in plan.microbatches:
                        add_template(catalogue,batch,lengths,long,short,args.world,str(path)+":plan")
    if args.expand:
        if args.world != 16:
            raise ValueError("Expanded shape grid currently declared only for the 16-rank construction")
        model = Model(cp_response=load_response())
        options = SearchOptions(beam=32,layouts=12,max_states=50000000,node_affinity=True,balanced_packing=True)
        search = _Search({},args.world,PATTERN,model,options)
        shapes = [(1,s) for s in (*range(33),40,48,55)] + [(0,s) for s in (*range(1,33),48,64,80,96,112,119)]
        begin = time.monotonic()
        for h,s in shapes:
            lengths = dict(enumerate([long]*h+[short]*s))
            search.lengths = lengths
            answer = search.batch(tuple(lengths))
            if answer is not None:
                add_template(catalogue,answer[1],lengths,long,short,args.world,"expanded_shape_grid")
            print(json.dumps(dict(shape=[h,s],templates=len(catalogue),states=search.states,seconds=time.monotonic()-begin)),flush=True)
            (args.output/"catalogue.json").write_text(json.dumps(list(catalogue.values()),separators=(",",":"))+"\n")
        (args.output/"expansion.json").write_text(json.dumps(dict(shapes=shapes,states_examined=search.states,
            search_options=asdict(options),seconds=time.monotonic()-begin,global_optimality_certificate=False),indent=2)+"\n")
    if args.constant_cp_controls:
        shapes = sorted({(r["long_samples"],r["short_samples"]) for r in catalogue.values()})
        for h,s in shapes:
            lengths = dict(enumerate([long]*h+[short]*s))
            full = Layout((Group(tuple(range(args.world)),tuple(lengths)),),"zigzag")
            for attention in layout_candidates(lengths,lengths,args.world,False,12,8,True):
                for ingress in (attention,full):
                    for mlp_groups in (attention,full):
                        batch = Microbatch(ingress,tuple(
                            (attention.with_mode("contiguous" if c=="G" else "zigzag"),
                             mlp_groups.with_mode("contiguous" if c=="G" else "zigzag")) for c in PATTERN))
                        add_template(catalogue,batch,lengths,long,short,args.world,"constant_cp_control")
    entries = list(catalogue.values())
    (args.output/"catalogue.json").write_text(json.dumps(entries,separators=(",",":"))+"\n")
    summaries=[]
    for case in args.cases:
        folder=args.output/case;folder.mkdir()
        h,long,s,short=CASES[case]
        (folder/"input.json").write_text(json.dumps(dict(lengths=[long]*h+[short]*s,grouping_constraint=False))+"\n")
        for scenario in args.scenarios:
            result,analysis,summary=audit(entries,case,scenario,memory_time_tolerance=args.memory_time_tolerance)
            for suffix,data in (("result",result),("analysis",analysis)):
                (folder/f"{scenario}_{suffix}.json").write_text(json.dumps(data,separators=(",",":"))+"\n")
            summaries.append(summary)
            (args.output/"summary.json").write_text(json.dumps(summaries,indent=2)+"\n")
            print(json.dumps(summary),flush=True)


if __name__=="__main__":
    main()
