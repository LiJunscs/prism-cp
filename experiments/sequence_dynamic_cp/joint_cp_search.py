"""Joint CP search over a complete frozen input pool, independent of N outputs."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

from joint_cp import Model, SearchOptions, solve
from joint_cp.replay import compile_replay
from joint_cp.model import execution_contract
from joint_cp.response import load_response
from joint_cp.solver import SearchLimit
from two_dimensional_inputs import validate_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world", type=int, default=16)
    parser.add_argument("--pattern", default="GGG*GGG*")
    parser.add_argument("--mode", choices=("exact", "bounded"), default="bounded")
    parser.add_argument("--node-affinity", action="store_true", help="Keep candidate node residency consistent across local CP widths")
    parser.add_argument("--homogeneous-packing", action="store_true", help="Also consider separate microbatches for each sequence length")
    parser.add_argument("--balanced-packing", action="store_true", help="Include heavy-CP groups with CP1 tails and balanced pack boundaries")
    parser.add_argument("--heavy-tail-packing", action="store_true", help="Jointly vary large-sample counts and shared tail allocation per batch")
    parser.add_argument("--max-attention-cp", type=int, help="Restrict attention CP, retaining all MLP placements")
    parser.add_argument("--memory-time-tolerance", type=float, default=0.001,
                        help="Bounded CP1 repacking may increase modeled update time by this fraction to lower peak memory")
    parser.add_argument("--cp-response", choices=("joint", "single_b1", "fixed_g8"), help="Reuse frozen profiling CP responses")
    parser.add_argument("--cp16-response", choices=("carry8", "paired_linear", "paired_quadratic"), default="paired_linear")
    parser.add_argument("--gqa-response-only", action="store_true", help="Do not transfer historical 32-head GDN response")
    parser.add_argument("--cp32-response", choices=("work_scaling", "carry_residual", "double_residual"), help="Explicit unmeasured CP32 hypothesis")
    parser.add_argument("--gdn32-multiplier", type=float, default=1., help="Unmeasured CP32 GDN stress factor, >=1")
    parser.add_argument("--model-profile", type=Path, help="JSON overrides for the explicitly provisional analytic model")
    parser.add_argument("--beam", type=int, default=8)
    parser.add_argument("--layouts", type=int, default=8)
    parser.add_argument("--max-states", type=int, default=1000000)
    parser.add_argument("--updates", type=int, default=3)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest)
    model = Model(**(json.loads(args.model_profile.read_text()) if args.model_profile else {}))
    if args.cp_response:
        if model.cp_response is not None:
            parser.error("Choose either a response embedded in --model-profile or --cp-response")
        response = load_response(args.cp_response, args.cp16_response, not args.gqa_response_only,
                                 args.cp32_response, args.gdn32_multiplier)
        model = replace(model, cp_response=response, provenance=model.provenance + "; CP response: " + response["name"])
    elif args.gqa_response_only or args.cp32_response or args.gdn32_multiplier != 1.:
        parser.error("Response options require --cp-response")
    options = SearchOptions(mode=args.mode, beam=args.beam, layouts=args.layouts, max_states=args.max_states,
                            node_affinity=args.node_affinity, homogeneous_packing=args.homogeneous_packing,
                            balanced_packing=args.balanced_packing, heavy_tail_packing=args.heavy_tail_packing,
                            max_attention_cp=args.max_attention_cp, memory_time_tolerance=args.memory_time_tolerance)
    lengths = dict(enumerate(s["length"] for s in manifest["samples"]))
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        result = solve(lengths, args.world, args.pattern, model, options)
    except SearchLimit as error:
        result = dict(status="search_budget_exhausted", plan=None, reason=str(error), gpu_admitted=False)
    result.update(model=asdict(model), search_options=asdict(options),
                  input_manifest=str(args.manifest.resolve()), input_tokens=sum(lengths.values()))
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    if result["plan"] is not None:
        replay = compile_replay(result, manifest, args.updates, args.output.name)
        (args.output / "replay.json").write_text(json.dumps(replay, separators=(",", ":")) + "\n")
        config = dict(mode="joint", model_contract=execution_contract(model),
            vocab_size=model.vocab, model_provenance=model.provenance)
        (args.output / "executor.json").write_text(json.dumps(config, indent=2) + "\n")
        print(json.dumps(dict(status=result["status"], microbatches=len(result["plan"]["microbatches"]),
            ingress_allocation_tokens=replay[0]["capacity_tokens"], predicted_update_ms=result["predicted_update_ms"],
            gpu_admitted=False)))
    else:
        print(json.dumps(dict(status=result["status"], gpu_admitted=False)))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
