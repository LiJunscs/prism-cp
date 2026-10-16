"""Free-pool multi-sample search on 16 GPUs and explicit 32-GPU hypotheses."""

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import time

from joint_cp import Model, Plan, SearchOptions, solve
from joint_cp.response import load_response
from joint_cp.analysis import describe


# Long samples above half the necessary memory ceiling cannot be paired in any
# legal microbatch, even after full-pool repacking. Counts are global counts.
CASES = {
    "72_samples_2p25Mi": (8, 256*1024, 64, 4096),
    "568_samples_16Mi": (56, 256*1024, 512, 4096),
    "1136_samples_32Mi": (112, 256*1024, 1024, 4096),
    "32gpu_540_samples_16Mi": (28, 512*1024, 512, 4096),
    "memory_margin_304_samples_16Mi": (48, 320*1024, 256, 4096),
}
SCENARIOS = {
    "16gpu": (16, None, 1., None),
    "32gpu_cp16_only": (32, None, 1., 16),
    "32gpu_work_scaling": (32, "work_scaling", 1., None),
    "32gpu_carry_residual": (32, "carry_residual", 1., None),
    "32gpu_double_residual": (32, "double_residual", 1., None),
    "32gpu_gdn1p5": (32, "carry_residual", 1.5, None),
    "32gpu_gdn2": (32, "carry_residual", 2., None),
}


def run(case, scenario, beam=32, layouts=12, max_states=4000000):
    count, long, short_count, short = CASES[case]
    values = [long]*count + [short]*short_count
    lengths = dict(enumerate(values))
    world, cp32, gdn32, cap = SCENARIOS[scenario]
    model = Model(cp_response=load_response(cp32=cp32, gdn32_multiplier=gdn32),
                  provenance="frozen CP<=16 response transfer; 32-GPU topology/CP32 hypothetical when used; analytic CP1/storage/routes")
    options = SearchOptions(beam=beam, layouts=layouts, max_states=max_states,
        node_affinity=True, homogeneous_packing=True, balanced_packing=True,
        heavy_tail_packing=True, max_attention_cp=cap)
    start = time.monotonic()
    result = solve(lengths, world, "GGG*GGG*", model, options)
    result.update(model=asdict(model), search_options=asdict(options), case=case, scenario=scenario,
        world=world, input_tokens=sum(values), input_samples=len(values), wall_seconds=time.monotonic()-start,
        necessary_microbatch_token_ceiling=model.token_capacity_upper_bound("GGG*GGG*", world),
        grouping_constraint=False, hypothetical_hardware=world==32)
    if result["plan"] is None:
        return values, result, {}, {k:result[k] for k in ("case","scenario","world","status","input_tokens","input_samples")}
    analysis = dict(two=describe(result, lengths, model),
                    two_response_evidence=result["cp_response_evidence"])
    plan = Plan.from_dict(result["plan"])
    batch_shapes = Counter((sum(lengths[s]==long for g in b.ingress.groups for s in g.samples),
                            sum(lengths[s]==short for g in b.ingress.groups for s in g.samples)) for b in plan.microbatches)
    row = dict(case=case, scenario=scenario, world=world, input_samples=len(values), input_tokens=sum(values),
               necessary_microbatch_token_ceiling=result["necessary_microbatch_token_ceiling"],
               max_long_samples_per_microbatch=result["necessary_microbatch_token_ceiling"]//long,
               **{k:v for k,v in analysis["two"].items() if k not in ("batches","stage_costs")},
               cp_calls=result["cp_response_evidence"]["counts"],
               selected_microbatch_shapes=[dict(long_samples=a,short_samples=b,microbatches=n) for (a,b),n in sorted(batch_shapes.items())],
               wall_seconds=result["wall_seconds"], hypothetical_hardware=world==32)
    return values, result, analysis, row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=["72_samples_2p25Mi", "568_samples_16Mi"])
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=["16gpu", "32gpu_cp16_only", "32gpu_carry_residual"])
    parser.add_argument("--beam", type=int, default=32)
    parser.add_argument("--layouts", type=int, default=12)
    parser.add_argument("--max-states", type=int, default=4000000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summary=[]
    for case in args.cases:
        folder=args.output/case;folder.mkdir()
        for scenario in args.scenarios:
            values,result,analysis,row=run(case,scenario,args.beam,args.layouts,args.max_states)
            for name,data in (("input.json",dict(lengths=values,input_tokens=sum(values),grouping_constraint=False)),
                              (scenario+"_result.json",result),(scenario+"_analysis.json",analysis),
                              (scenario+"_model.json",result["model"])):
                (folder/name).write_text(json.dumps(data,separators=(",",":"))+"\n")
            summary.append(row)
            (args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
            print(json.dumps(row),flush=True)


if __name__=="__main__":
    main()
