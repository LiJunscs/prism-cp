"""Cached CPU comparison of fixed K, batch-adaptive K and joint selection.

GDN:GQA is fixed at 3:1. Joint candidates retain MLP placement freedom. A
finite-support random batch law is integrated exactly; this is not a GPU run,
an IID-per-sample simulation, or an online planner implementation.
"""

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path

from joint_cp import Model
from memory_performance_policy import tradeoff
from native import load_native
from native_n_generalization_theory import accounting, native_plan

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "results/joint_advantage_search_20261016_r1/analysis.json"
POLICIES = ("paired_linear", "paired_quadratic")
ALIGNMENT, FIXED_REFERENCE = 32, 16 * 1024


def capacity_shortlist(lengths, reference=FIXED_REFERENCE):
    """Small 1-D rule: native CP thresholds plus a common fallback capacity.

    K is aligned to 32 tokens, NOT rounded to powers of two or integer Ki.
    The study scores the part within the declared 8..40Ki candidate domain.
    """
    minimum = math.ceil(max(lengths) / 16)
    values = {reference}
    values.update(math.ceil(length / (cp * ALIGNMENT)) * ALIGNMENT
                  for length in set(lengths) for cp in (1, 2, 4, 8, 16))
    return sorted(k for k in values if max(8 * 1024, minimum) <= k <= 40 * 1024)


def is_eligible(score, reference, preference):
    return score["model_fits"] and (preference == "physical" or tradeoff(
        reference["predicted_update_ms"], reference["peak_gib"],
        score["predicted_update_ms"], score["peak_gib"],
        min_gain_per_memory_growth=2.0)["accepted"])


def choice_key(row, policy):
    s = row["scores"][policy]
    return s["predicted_update_ms"], s["peak_gib"], row["id"]


def choose(case, policy, preference):
    ref = next(n["scores"][policy] for n in case["native"] if n["capacity_tokens"] == FIXED_REFERENCE)
    native = [n for n in case["native"] if is_eligible(n["scores"][policy], ref, preference)]
    simple = min((n for n in native if n["capacity_tokens"] in case["simple_capacity_tokens"]),
                 key=lambda n: choice_key(n, policy))
    oracle = min(native, key=lambda n: choice_key(n, policy))
    joint = [j for j in case["joint"] if is_eligible(j["scores"][policy], ref, preference)]
    best_joint = min(joint, key=lambda n: choice_key(n, policy)) if joint else None
    selected = min([simple, *joint], key=lambda n: choice_key(n, policy))

    def compact(row):
        if row is None:
            return None
        score = row["scores"][policy]
        return dict(id=row["id"], capacity_tokens=row.get("capacity_tokens"),
                    time_ms=score["predicted_update_ms"], peak_gib=score["peak_gib"])

    return dict(simple=compact(simple), finite_oracle=compact(oracle),
                best_joint=compact(best_joint), selected=compact(selected),
                feasible_native={n["capacity_tokens"]: compact(n) for n in native},
                simple_matches_finite_oracle=math.isclose(simple["scores"][policy]["predicted_update_ms"],
                    oracle["scores"][policy]["predicted_update_ms"], rel_tol=1e-12))


def integrate(cases, weights, policy, preference):
    """Ratio of expectations, not the mean of per-batch speedups."""
    active = [(c, weights.get(c["case"], 0.0)) for c in cases if weights.get(c["case"], 0.0) > 0]
    assert math.isclose(sum(w for _, w in active), 1.0)
    decisions = [(c, w, choose(c, policy, preference)) for c, w in active]
    common = set.intersection(*(set(d["feasible_native"]) for _, _, d in decisions))
    fixed_k = min(common, key=lambda k: (sum(w * d["feasible_native"][k]["time_ms"]
                                                for _, w, d in decisions), k))
    fixed_time = sum(w * d["feasible_native"][fixed_k]["time_ms"] for _, w, d in decisions)
    result = dict(fixed_capacity_tokens=fixed_k, weights=weights,
                  expected_tokens=sum(w * c["input_tokens"] for c, w in active))
    for method in ("fixed", "simple", "finite_oracle", "selected"):
        entries = [(w, d["feasible_native"][fixed_k] if method == "fixed" else d[method])
                   for _, w, d in decisions]
        mean = sum(w * e["time_ms"] for w, e in entries)
        result[method] = dict(expected_update_ms=mean,
            effective_tokens_per_second=1000 * result["expected_tokens"] / mean,
            worst_case_model_peak_gib=max(e["peak_gib"] for _, e in entries))
    adapt = result["simple"]["expected_update_ms"]
    joint = result["selected"]["expected_update_ms"]
    result.update(simple_gain_vs_fixed_pct=100 * (fixed_time / adapt - 1),
        selection_gain_vs_simple_pct=100 * (adapt / joint - 1),
        selection_gain_vs_fixed_pct=100 * (fixed_time / joint - 1),
        capacity_saving_ms=fixed_time - adapt, additional_joint_saving_ms=adapt - joint,
        joint_selected_probability=sum(w for _, w, d in decisions if d["selected"]["id"].startswith("J_")),
        simple_matches_finite_oracle=all(d["simple_matches_finite_oracle"] for _, _, d in decisions))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = json.loads(SOURCE.read_text())
    excluded = [dict(case=c["case"], reason="different layer ratio" if c["pattern"] != "GGG*GGG*"
                     else "unresolved native mixed-partition capacity", details=c["native_unsupported"])
                for c in source["cases"] if c["pattern"] != "GGG*GGG*" or c["native_unsupported"]]
    cases = [c for c in source["cases"] if c["pattern"] == "GGG*GGG*" and not c["native_unsupported"]]
    base = Model(**json.loads((HERE / "configs/joint_16gpu_measured.json").read_text()))
    function, _ = load_native()
    extensions = []
    for case in cases:
        for n in case["native"]:
            n["capacity_tokens"] = int(n["capacity_Ki"] * 1024)
        capacities = capacity_shortlist(case["lengths"])
        case["simple_capacity_tokens"] = capacities
        present = {n["capacity_tokens"] for n in case["native"]}
        for capacity in sorted(set(capacities) - present):
            lengths = dict(enumerate(case["lengths"]))
            plan = native_plan(lengths, function, capacity)
            scores = {p: accounting(plan, lengths, replace(base, cp_response=dict(base.cp_response, cp16_policy=p)))
                      for p in POLICIES}
            row = dict(id=f"N_K{capacity/1024:g}", capacity_tokens=capacity,
                       capacity_Ki=capacity/1024, scores=scores)
            case["native"].append(row)
            extensions.append(dict(case=case["case"], candidate=row, plan=plan.to_dict()))
        case["decisions"] = {f"{p}/{m}": choose(case, p, m)
                             for p in POLICIES for m in ("physical", "alpha2")}
    laws = {"uniform_cached_support": {c["case"]: 1 / len(cases) for c in cases}}
    # Declared illustrative regimes, not a fitted estimate of a real dataset.
    low = ("L192_S4", "L208_S4")
    high = ("L224_S4", "L256_S4", "irregular_S8")
    for q in (0.0, 0.25, 0.5, 0.75, 1.0):
        laws[f"mixed_regimes_q{q:g}"] = {**{n: (1 - q) / len(low) for n in low},
                                        **{n: q / len(high) for n in high}}
    summaries = {name: {f"{p}/{m}": integrate(cases, weights, p, m)
                        for p in POLICIES for m in ("physical", "alpha2")}
                 for name, weights in laws.items()}
    result = dict(scope="CPU finite-support stochastic input analysis; exact expectation over cached histograms",
        source=str(SOURCE.relative_to(HERE)), pattern="GGG*GGG*", world=16, gpu_launches=0,
        randomness="Each batch draws a length multiset from the declared law, then randomly permutes sample order; not IID-per-sequence draws",
        fixed_baseline="Best single K over the full declared distribution and candidate domain; uses offline distribution knowledge",
        adaptive_baseline="Current-input native CP threshold shortlist, 32-token alignment, rank by the unchanged model with memory admission",
        oracle_scope="Per-batch best scored N over K8..K40 in 1Ki steps plus shortlisted fractional capacities; not all integer capacities",
        joint_scope="All cached 3:1 joint candidates, retaining MLP placement freedom, with offline N selection; no online branch implemented",
        preference_reference="Same-input N-K16, fixed for both methods; physical-only and alpha2 kept separate",
        excluded=excluded, new_native_evaluations=extensions, cases=cases, summaries=summaries,
        limitations=["Unvalidated cost/memory coefficients inherited unchanged",
            "Historical support is constructed, not representative random training data",
            "No planning, switching, input waiting, or cross-step allocator-state cost",
            "Small-shortlist ranking still uses the full shared cost/memory model; its online cost is unmeasured",
            "Model infeasibility is not measured physical infeasibility"])
    (args.output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(new_native_evaluations=len(extensions), cases=len(cases), summaries=summaries), indent=2))


if __name__ == "__main__":
    main()
