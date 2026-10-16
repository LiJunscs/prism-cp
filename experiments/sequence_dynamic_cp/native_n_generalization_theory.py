"""Small CPU-only constructions compared directly with unmodified native N.

These are complete-pool candidate costs, not GPU measurements or a search for
an optimal joint plan. Input families are declared before evaluation. No tensor
imports, profiling, source snapshots, file hashes, or native capacity tuning.
"""

import argparse
from collections import Counter
from dataclasses import replace
import csv
import json
from pathlib import Path

from joint_cp import Group, Layout, Microbatch, Model, Plan, validate_plan
from joint_cp.analysis import describe, measure
from joint_cp.ir import identity_route, route
from mixed_plan import coalesce_rounds
from native import load_native
from planner import Constraints, from_rank_ids, validate

HERE = Path(__file__).resolve().parent
WORLD, PATTERN, CAPACITY = 16, "GGG*GGG*", 16384


def cases():
    # Vary the forced-cross-node condition, short-work fraction and regularity.
    rows = [(f"L{n}Ki_64_S4Ki_512", [n*1024]*64, [4096]*512)
            for n in (64, 128, 160, 192, 224, 256)]
    rows += [(f"L256Ki_56_S4Ki_{count}", [262144]*56, [4096]*count)
             for count in (128, 512, 2048)]
    rows.append(("irregular_16Mi", [n*1024 for n in (160, 192, 224, 256)]*16,
                 [n*1024 for n in (2, 3, 5, 6)]*192))
    return rows


def native_plan(lengths, function, capacity=CAPACITY):
    remaining = sorted(lengths.items(), key=lambda pair: pair[1], reverse=True)
    ranks = []
    while remaining:
        _, leftovers, _, assignment = function(remaining, WORLD, capacity, 1)
        if len(leftovers) >= len(remaining):
            raise ValueError("Native N did not consume input")
        ranks.append(assignment)
        remaining = leftovers
    native = from_rank_ids(ranks)
    validate(native, lengths, Constraints(WORLD, capacity, alignment=32))
    rounds = coalesce_rounds(native, lengths, WORLD, capacity)
    batches = []
    for indices in rounds:
        # Merge only identical rank partitions. Other mixed-context layouts need
        # a different cost representation and must not be silently simplified.
        partition = tuple(g.ranks for g in native[indices[0]])
        if any(tuple(g.ranks for g in native[i]) != partition for i in indices):
            raise ValueError("Native execution has multiple context partitions in a microbatch")
        groups = tuple(Group(rs, tuple(s for i in indices for g in native[i]
                                       if g.ranks == rs for s in g.ids)) for rs in partition)
        ingress = Layout(groups, "zigzag")
        contiguous = ingress.with_mode("contiguous")
        batches.append(Microbatch(ingress, tuple(
            (contiguous, contiguous) if c == "G" else (ingress, ingress) for c in PATTERN)))
    return Plan(WORLD, PATTERN, tuple(batches))


def joint_candidate(long_count, short_count):
    if short_count < long_count:
        raise ValueError("This declared construction requires a short sample per long sample")
    batches = []
    for sample in range(long_count):
        short = long_count + sample
        ingress = Layout((Group(tuple(range(WORLD)), (sample, short)),), "zigzag")
        contiguous = ingress.with_mode("contiguous")
        gqa = Layout((Group(tuple(range(8)), (sample,)), Group(tuple(range(8, 16)), (short,))), "zigzag")
        batches.append(Microbatch(ingress, tuple(
            (contiguous if c == "G" else gqa, contiguous) for c in PATTERN)))
    remaining = list(range(2*long_count, long_count+short_count))
    while remaining:
        count = min(64, len(remaining))
        if count >= WORLD:
            count -= count % WORLD
            per_rank = count // WORLD
            groups = tuple(Group((r,), tuple(remaining[r*per_rank:(r+1)*per_rank])) for r in range(WORLD))
        else:
            count = 1 << (count.bit_length()-1)
            cp = WORLD // count
            groups = tuple(Group(tuple(range(i*cp, (i+1)*cp)), (remaining[i],)) for i in range(count))
        remaining = remaining[count:]
        ingress = Layout(groups, "zigzag")
        contiguous = ingress.with_mode("contiguous")
        batches.append(Microbatch(ingress, tuple(
            (contiguous, contiguous) if c == "G" else (ingress, ingress) for c in PATTERN)))
    return Plan(WORLD, PATTERN, tuple(batches))


def accounting(plan, lengths, model):
    validate_plan(plan, lengths)
    for batch in plan.microbatches:
        previous = None
        for symbol, (attention, mlp) in zip(plan.pattern, batch.layers):
            if symbol == "G":
                assert attention.mode == mlp.mode == "contiguous"
                assert identity_route(route(attention, mlp, lengths, plan.world))
                if previous is not None:
                    assert identity_route(route(previous, attention, lengths, plan.world))
                previous = attention
            else:
                previous = None
    score, evidence = measure(plan, lengths, model)
    result = dict(plan=plan.to_dict(), predicted_update_ms=score, microbatch_evidence=evidence,
                  status="explicit_candidate_cpu_only", state_budget_exhausted=False, states_examined=0)
    analysis = describe(result, lengths, model)
    analysis.pop("batches")
    analysis["model_fits"] = all(row["fits"] for row in evidence)
    return analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model = Model(**json.loads((HERE / "configs/joint_16gpu_measured.json").read_text()))
    function, _ = load_native()
    output, table = [], []
    for name, longs, shorts in cases():
        lengths = dict(enumerate(longs + shorts))
        native = native_plan(lengths, function)
        joint = joint_candidate(len(longs), len(shorts))
        cp_counts = Counter(cp for b in native.microbatches for sample, cp in b.layers[0][0].sample_cp()
                            if sample < len(longs))
        policies = {}
        for policy in ("paired_linear", "paired_quadratic"):
            profile = dict(model.cp_response, cp16_policy=policy)
            scenario_model = replace(model, cp_response=profile)
            n, candidate = accounting(native, lengths, scenario_model), accounting(joint, lengths, scenario_model)
            gain = 100*(1-candidate["predicted_update_ms"]/n["predicted_update_ms"])
            policies[policy] = dict(native=n, joint=candidate, predicted_time_reduction_pct=gain)
        row = dict(case=name, lengths=list(lengths.values()), input_tokens=sum(lengths.values()),
                   long_samples=len(longs), short_samples=len(shorts),
                   long_token_fraction=sum(longs)/sum(lengths.values()), native_long_cp_counts=dict(cp_counts),
                   policies=policies)
        output.append(row)
        n, candidate = policies["paired_linear"]["native"], policies["paired_linear"]["joint"]
        summary = dict(case=name, input_Mi=sum(lengths.values())/2**20,
            native_long_cp=json.dumps(dict(cp_counts)), native_microbatches=n["microbatches"],
            joint_microbatches=candidate["microbatches"], native_predicted_ms=n["predicted_update_ms"],
            joint_predicted_ms=candidate["predicted_update_ms"], native_model_peak_GiB=n["peak_gib"],
            joint_model_peak_GiB=candidate["peak_gib"], both_model_fit=n["model_fits"] and candidate["model_fits"],
            linear_gain_pct=policies["paired_linear"]["predicted_time_reduction_pct"],
            quadratic_gain_pct=policies["paired_quadratic"]["predicted_time_reduction_pct"])
        table.append(summary)
        print(json.dumps(summary), flush=True)
    (args.output / "analysis.json").write_text(json.dumps(dict(
        scope="CPU constructions, not optimized search or GPU measurements; complete native N comparator",
        model_profile="configs/joint_16gpu_measured.json", native_capacity=CAPACITY,
        world=WORLD, pattern=PATTERN, gpu_launches=0,
        sensitivity="two existing CP16 response interpolation policies, not confidence intervals",
        cases=output), indent=2)+"\n")
    with (args.output / "summary.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)


if __name__ == "__main__":
    main()
