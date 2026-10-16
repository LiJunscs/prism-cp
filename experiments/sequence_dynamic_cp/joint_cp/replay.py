"""Compile a joint plan to indexed packet replay; no native scheduler calls."""

from .ir import Plan, validate_plan


def compile_replay(result, manifest, updates=3, candidate_id="joint_cp"):
    if updates < 1 or result.get("plan") is None:
        raise ValueError("Replay requires a feasible plan and positive update count")
    plan = Plan.from_dict(result["plan"])
    lengths = dict(enumerate(s["length"] for s in manifest["samples"]))
    stable = [s["sample_id"] for s in manifest["samples"]]
    if len(set(stable)) != len(stable):
        raise ValueError("Duplicate stable sample IDs")
    validate_plan(plan, lengths)
    rank_ids = [b.ingress.rank_ids(plan.world) for b in plan.microbatches]
    # Compatibility bound for ingress tensors only, derived AFTER joint search.
    # This is neither an N generation K nor a constraint on GQA/GDN/MLP layouts.
    ingress_max = max(max(b.ingress.tokens(lengths, plan.world)) for b in plan.microbatches)
    if manifest["input_tokens"] != sum(lengths.values()):
        raise ValueError("Manifest input total differs from sample lengths")
    return [dict(step=step, stream_id="TWO_DIMENSIONAL", architecture="gdngqa_gpt", role="S",
                 candidate_id=candidate_id, case_id=manifest["case_id"], seed=manifest["seed"],
                 capacity_tokens=ingress_max, capacity_semantics="derived_ingress_allocation_bound",
                 rank_ids=rank_ids, expected_lengths=list(lengths.values()), logical_sample_ids=stable,
                 stable_to_scheduler_ordinal=[dict(stable_sample_id=s, scheduler_ordinal=i) for i, s in enumerate(stable)],
                 native_verified_cpu=False, execution_rounds=[[i] for i in range(len(rank_ids))],
                 source_microbatches=len(rank_ids), merged_microbatches=len(rank_ids),
                 input_tokens=manifest["input_tokens"], loss_tokens=manifest["loss_tokens"],
                 repeat_same_frozen_content=True, joint_plan=plan.to_dict(),
                 status="CPU_ONLY_NOT_GPU_ADMITTED") for step in range(1, updates + 1)]


def validate_replay(row, world, pattern=None):
    plan = Plan.from_dict(row["joint_plan"])
    if plan.world != world:
        raise ValueError("Joint replay world was changed")
    lengths = dict(enumerate(row["expected_lengths"]))
    validate_plan(plan, lengths)
    expected = [b.ingress.rank_ids(world) for b in plan.microbatches]
    capacity = max(max(b.ingress.tokens(lengths, world)) for b in plan.microbatches)
    if (plan.world != world or (pattern is not None and plan.pattern != pattern)
            or row["rank_ids"] != expected or row["capacity_tokens"] != capacity
            or row["execution_rounds"] != [[i] for i in range(len(expected))]
            or row["role"] != "S" or row.get("capacity_semantics") != "derived_ingress_allocation_bound"):
        raise ValueError("Joint replay ingress, world, pattern, rounds or capacity was changed")
    return plan
