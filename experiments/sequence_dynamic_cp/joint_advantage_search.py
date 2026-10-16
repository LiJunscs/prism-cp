"""CPU-only, finite candidate study with an unmodified native capacity control.

Primary candidates keep each MLP in its attention layout: layout changes occur
at operator-type boundaries and model I/O only. Historical joint plans also
redistribute GQA MLP tokens; that extra freedom is reported separately.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path

from joint_cp import Group, Layout, Microbatch, Model, Plan, validate_plan
from joint_cp.ir import identity_route, route
from memory_performance_policy import tradeoff
from native import load_native, replay
from native_n_generalization_theory import accounting, native_plan

HERE = Path(__file__).resolve().parent
CAPACITIES = tuple(range(8, 41))
POLICIES = ("paired_linear", "paired_quadratic")
PATTERN = "GGG*GGG*"


def layout_audit(plan, lengths):
    internal = mlp_routes = boundaries = io = 0
    for batch in plan.microbatches:
        previous, previous_symbol = batch.ingress, None
        for symbol, (attention, mlp) in zip(plan.pattern, batch.layers):
            incoming = not identity_route(route(previous, attention, lengths, plan.world))
            outgoing = not identity_route(route(attention, mlp, lengths, plan.world))
            if previous_symbol is None:
                io += incoming
            elif previous_symbol != symbol:
                boundaries += incoming
            elif incoming:
                raise ValueError("A same-operator run changes token ownership")
            if symbol == "G":
                assert attention.mode == mlp.mode == "contiguous"
                internal += 2 * outgoing
            else:
                mlp_routes += 2 * outgoing
            previous, previous_symbol = attention, symbol
        io += not identity_route(route(previous, batch.ingress, lengths, plan.world))
    assert internal == 0
    return dict(gdn_internal_routes=internal, gqa_mlp_routes=mlp_routes,
                operator_boundary_routes=boundaries, model_io_routes=io,
                scope="nonidentity forward routes; backward uses inverse routes")


def audit_frozen():
    native_rows, joint_rows = [], []
    for bundle in ("native_n_remeasure_20261016", "native_capacity_ablation_20261016_r1"):
        for path in sorted((HERE / "results" / bundle / "plans").glob("*.json")):
            data = json.loads(path.read_text())
            actual = replay(list(enumerate(data["expected_lengths"])), 16, data["capacity_tokens"])
            assert actual == data["rank_ids"], path
            run = path.parent.parent / "runs" / path.stem
            config = json.loads((run / "config.json").read_text())
            protocol = json.loads((run / "protocol.json").read_text())
            assert config["native_n"] and config["mixed_cp"]
            assert not config.get("two_dimensional") and not config["allow_compute_proxy"]
            assert config["input_transport"] == "native"
            assert protocol["environment"]["MODE"] == "DUPC"
            assert int(protocol["environment"]["MAX_TOKENS_PER_RANK"]) == data["capacity_tokens"]
            assert all(row["rank_ids"] == actual for row in config["replay"])
            native_rows.append(dict(path=str(path.relative_to(HERE)),
                capacity_tokens=data["capacity_tokens"], original_ast_matches=True,
                source_microbatches=len(actual), run_config_checked=True,
                executor="DUPC", joint_adapter=False, generation_execution_capacity_matches=True))
    for bundle in ("joint_cp_two_20261015", "joint_cp_memory_validation_20261016_r2"):
        for path in sorted((HERE / "results" / bundle / "plans").glob("*.json")):
            data = json.loads(path.read_text())
            inputs = json.loads((path.parent.parent / "inputs" / path.name).read_text())
            lengths = {i: s["length"] for i, s in enumerate(inputs["samples"])}
            plan = Plan.from_dict(data["plan"])
            validate_plan(plan, lengths)
            joint_rows.append(dict(path=str(path.relative_to(HERE)), **layout_audit(plan, lengths)))
    assert len(native_rows) == 5 and len(joint_rows) == 4
    return dict(native=native_rows, joint=joint_rows,
        native_scope="Current local Megatron next_hdp_group_packing_aware, beta_tokens=0; frozen replay, not online planning",
        finding="All GDN runs have zero internal routes; historical joint GQA MLP placement has additional routes")


def cases():
    rows = [(f"L{long}_S{short}", [long] * 8, [short] * 64, PATTERN)
            for long, short in ((192, 4), (208, 4), (224, 4), (240, 4),
                                (248, 4), (256, 4), (224, 8), (256, 8), (256, 16))]
    rows += [
        ("irregular_S8", [160, 192, 224, 256] * 2, [8] * 64, PATTERN),
        ("L256_few_short", [256] * 8, [4] * 8, PATTERN),
        ("L256_many_short", [256] * 8, [4] * 256, PATTERN),
        ("L256_G7A1", [256] * 8, [4] * 64, "GGGGGGG*"),
        ("L256_G4A4_blocked", [256] * 8, [4] * 64, "GG**GG**"),
        ("L256_G4A4_alternating", [256] * 8, [4] * 64, "G*G*G*G*"),
    ]
    return [(name, [x * 1024 for x in longs], [x * 1024 for x in shorts], pattern)
            for name, longs, shorts, pattern in rows]


def repattern_native(plan, pattern):
    return Plan(plan.world, pattern, tuple(Microbatch(batch.ingress, tuple(
        (batch.ingress.with_mode("contiguous"),) * 2 if symbol == "G"
        else (batch.ingress,) * 2 for symbol in pattern)) for batch in plan.microbatches))


def candidate(long_count, short_count, short_length, pattern, packet, split_short, mlp_policy):
    """Independent complete-pool construction; no native plan is consumed."""
    if long_count * packet > short_count:
        return None
    batches = []
    for sample in range(long_count):
        shorts = tuple(range(long_count + sample * packet, long_count + (sample + 1) * packet))
        ingress = Layout((Group(tuple(range(16)), (sample,) + shorts),), "zigzag")
        contiguous = ingress.with_mode("contiguous")
        if split_short:
            width = 8 // packet
            groups = tuple(Group(tuple(range(8 + i * width, 8 + (i + 1) * width)), (s,))
                           for i, s in enumerate(shorts))
        else:
            groups = (Group(tuple(range(8, 16)), shorts),)
        gqa = Layout((Group(tuple(range(8)), (sample,)),) + groups, "zigzag")
        batches.append(Microbatch(ingress, tuple(
            (contiguous, contiguous) if symbol == "G" else
            (gqa, gqa if mlp_policy == "follow_attention" else contiguous) for symbol in pattern)))
    remaining = list(range(long_count * (1 + packet), long_count + short_count))
    while remaining:
        count = min(16 * max(1, 16384 // short_length), len(remaining))
        if count >= 16:
            count -= count % 16
            per_rank = count // 16
            groups = tuple(Group((r,), tuple(remaining[r * per_rank:(r + 1) * per_rank])) for r in range(16))
        else:
            count = 1 << (count.bit_length() - 1)
            width = 16 // count
            groups = tuple(Group(tuple(range(i * width, (i + 1) * width)), (remaining[i],)) for i in range(count))
        remaining = remaining[count:]
        ingress = Layout(groups, "zigzag")
        contiguous = ingress.with_mode("contiguous")
        batches.append(Microbatch(ingress, tuple((contiguous, contiguous) if s == "G"
                                               else (ingress, ingress) for s in pattern)))
    return Plan(16, pattern, tuple(batches))


def write_report(output, result):
    lines = [
        "# N 与布局核对，以及二维优势场景筛选（2026-10-16）", "",
        "本次只执行 CPU 原函数回放、计划合法性核对和有限候选评分，GPU 启动为 0。"
        "所有新时延、显存、吞吐数字均为模型预测，不能引用为训练实测。", "",
        "## 基线与布局事实", "",
        "- 5 份 N 冻结计划（3 份 K16、72 条 K24/K32）与当前工作区 Megatron 的 "
        "`next_hdp_group_packing_aware` 原函数逐 rank/sample 一致。`load_native()` 未传 beta，"
        "保留 n²/CP、greedy/fill、tie-break。配置核对确认 DUPC、无 joint adapter、"
        "无代理成本选择，生成与执行 K 相同。这是冻结计划回放，未测在线决策。",
        "- N 的普通与 mixed parent-contiguous 路径仅在布局状态改变时转换；"
        "GDN 层配置为 contiguous，底层转换器遇到相同源/目标模式直接返回。"
        "连续 GDN 及其 MLP 没有布局往返。",
        "- 4 份历史二维计划的 GDN 内部非恒等路由为 0。GQA MLP 另有往返："
        "72/568/1136 条每次前向为 32/224/448 次计划路由，568 修复后仍为 224。"
        "这些路由在模型中计费，反向走逆路由；不是 GDN 段内转换。",
        "- 因而不能把历史二维描述成全模型仅在 GQA/GDN 边界转换。新的严格候选令所有 "
        "MLP 跟随 attention；同类型连续层布局不变，模型入口/出口转换仍保留。", "",
        "证据：`audit.json`。代码链：`native.py` → Megatron "
        "`megatron/core/datasets/data_schedule_utils.py:592`；`runtime.py`/`gpu_smoke.py` → "
        "`scripts/run_manual_gdn_cp_case.sh` 的 DUPC；Megatron `transformer_block.py` 的 "
        "`_parent_contiguous_forward`、`mixed_context_parallel.py` 的 "
        "`mixed_parent_contiguous_forward`、`context_parallel_layout/conversion.py`；"
        "二维为 `joint_cp/adapter.py:mlp_placement` 与 `joint_cp/transport.py`。", "",
        "## 有限比较合同", "",
        "16 卡、现有模型形状和冻结响应系数不变，复用 `configs/joint_16gpu_measured.json`。"
        "N 对每个完整输入从原函数独立生成 K=8…40Ki（步长 1Ki）的计划；"
        "不足以放下最长序列的 K 直接排除。未穷尽所有容量。",
        "二维独立枚举每个长样本搭配 1/2/4/8 个短样本、短样本 packed/split 放置，"
        "GDN16/GQA 长样本 CP8；剩余短批按每 rank 至多 16Ki 重打包。"
        "分别评分 MLP 跟随 attention 与 MLP 重分布。所有样本恰好一次、布局合法性与 "
        "GDN 连续性均执行检查。这是构造候选筛选，不是完整二维求解器的全局最优结论。", "",
        "物理模型预算为 71.178833GiB。表中峰值是含 external=7GiB、margin=2GiB 的"
        "完整训练模型峰值，不是实测 allocated。先报告仅物理模型预算，再独立应用 α=2，"
        "共同固定参考始终是同一输入的 N-K16。两种 CP16 响应政策为敏感性假设，"
        "范围不是置信区间。", "",
        "## 严格只在算子边界及模型 I/O 转换的结果", "",
        "默认每项为 8 条长序列＋64 条短序列，pattern=GGG*GGG*；few/many_short "
        "分别为 8/256 条短序列。irregular 为 [160,192,224,256]Ki 各 2 条＋64×8Ki。"
        "G7A1/G4A4 改变同一对照内双方的层组成；其余形状保持不变。", "",
        "|场景|最佳可评分 N K（Ki）|预测 N→二维秒，linear|模型峰值 N→二维 GiB|预测吞吐增益，两政策|α=2 离线选择|未评分 N 容量数|",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for case in result["cases"]:
        values = [case["selections"][f"{p}/physical/follow_attention"] for p in POLICIES]
        row = values[0]
        gains = [v["throughput_gain_pct"] for v in values]
        selections = [case["selections"][f"{p}/alpha2/follow_attention"]["selected"] for p in POLICIES]
        decision = "二维" if all(s and s.startswith("J_") for s in selections) else "N"
        if case["native_unsupported"]:
            decision += "（比较未决）"
        lines.append(f"|{case['case']}|{row['native'].removeprefix('N_K')}|"
                     f"{row['native_ms']/1000:.3f}→{row['joint_ms']/1000:.3f}|"
                     f"{row['native_peak_gib']:.3f}→{row['joint_peak_gib']:.3f}|"
                     f"{min(gains):+.2f}%～{max(gains):+.2f}%|{decision}|{len(case['native_unsupported'])}|")
    lines += ["", "## 优先推进的线索与限制", "",
        "1. **224/240/248Ki＋4Ki 短样本**：严格候选相对有限容量内最佳 N 的预测吞吐 "
        "+25.24%～+30.58%，两种响应政策和 α=2 均保留收益。N 最佳容量分别为 "
        "K14/K15/K16，而不是强制 K16；较大容量将长序列变为 CP8 时受模型显存约束。"
        "224Ki 是较小、较直接的下一实测输入候选。",
        "2. **不规则长序列＋8Ki 短样本**：仅物理预算下，相对 N-K24 的预测吞吐 "
        "+9.35%～+24.81%，模型峰值 66.621→59.502GiB。α=2 会排除 N-K24，"
        "此时比较 N-K16 得 +29.88%～+47.92%；不能混用两种基线。该场景同时预测降显存，"
        "但结果更依赖 CP16 响应政策，适合作为不规则输入的独立验证。",
        "3. **7GDN＋1GQA、256Ki＋4Ki**：严格候选预测 +16.75%，模型峰值 "
        "52.420→56.226GiB，α=2 通过。4GDN＋4GQA 虽预测约 +31%，显存增长使严格候选"
        "未通过 α=2；GQA 层更多不自动意味着更好的显存交换。",
        "4. **保留反例与未决项**：192/208Ki 的 N-K24/K26 快于本候选库的二维，应选 N。"
        "224Ki＋8Ki 的 K29 合并后含多种 context partition，现有 native IR 评分不支持，"
        "不能将它当作不可行并据此宣称胜过完整容量集合。",
        "5. **MLP 自由度有实际建模作用**：原 256Ki＋4Ki 的严格候选预测 +24.39%，"
        "模型峰值 52.244→59.187GiB，未通过 α=2；保留 GQA MLP 重分布后预测 +28.41%，"
        "峰值约 55.03GiB 并通过。去掉往返也会放弃 MLP 负载/显存平衡，不能仅减去路由时间。",
        "", "所有新候选都未 GPU 验证。CP1、packed、GDN CP16、head 数迁移、MLP/路由成本"
        "和完整显存均继承原模型局限；层组成改变仍需单独实测。未重新拟合系数，"
        "未计在线规划及切换，不作数值等价、长期收敛或全局最优结论。", "",
        "复现（输出须为新目录，不会启动 GPU）：", "", "```bash",
        "/workspace/envs/flagscale/bin/python experiments/sequence_dynamic_cp/joint_advantage_search.py --output /tmp/joint_advantage_fresh",
        "```", "", "`analysis.json` 保存全部评分、容量、拒绝/未决记录及选择；"
        "各 `*_plans.json` 保存完整候选 IR；`audit.json` 保存既有计划核对。",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    audit = audit_frozen()
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    base = Model(**json.loads((HERE / "configs/joint_16gpu_measured.json").read_text()))
    function, _ = load_native()
    rows = []
    for name, longs, shorts, pattern in cases():
        lengths = dict(enumerate(longs + shorts))
        models = {p: replace(base, cp_response=dict(base.cp_response, cp16_policy=p)) for p in POLICIES}
        plans, natives, joints, errors, cache = {}, [], [], [], {}

        def score(plan):
            layout_audit(plan, lengths)
            if plan not in cache:
                cache[plan] = {p: accounting(plan, lengths, model) for p, model in models.items()}
            return cache[plan]

        for capacity in CAPACITIES:
            if max(lengths.values()) > 16 * capacity * 1024:
                continue
            try:
                plan = repattern_native(native_plan(lengths, function, capacity * 1024), pattern)
            except (ValueError, AssertionError) as error:
                errors.append(dict(capacity_Ki=capacity, reason=str(error)))
                continue
            key = f"N_K{capacity}"
            plans[key] = plan.to_dict()
            natives.append(dict(id=key, capacity_Ki=capacity, scores=score(plan)))
        for packet in (1, 2, 4, 8):
            for split in (False, True) if packet > 1 else (False,):
                for mlp in ("follow_attention", "redistribute"):
                    plan = candidate(len(longs), len(shorts), shorts[0], pattern, packet, split, mlp)
                    if plan is None:
                        continue
                    key = f"J_p{packet}_{'split' if split else 'packed'}_{mlp}"
                    plans[key] = plan.to_dict()
                    audit_row = layout_audit(plan, lengths)
                    if mlp == "follow_attention":
                        assert audit_row["gqa_mlp_routes"] == 0
                    joints.append(dict(id=key, packet=packet, split_short=split, mlp_policy=mlp,
                                       layout=audit_row, scores=score(plan)))
        selections = {}
        for policy in POLICIES:
            ref = next(n["scores"][policy] for n in natives if n["capacity_Ki"] == 16)
            for alpha in (None, 2.0):
                def accepted(row):
                    s = row["scores"][policy]
                    return s["model_fits"] and (alpha is None or tradeoff(
                        ref["predicted_update_ms"], ref["peak_gib"], s["predicted_update_ms"],
                        s["peak_gib"], min_gain_per_memory_growth=alpha)["accepted"])

                def best(pool):
                    eligible = [c for c in pool if accepted(c)]
                    return min(eligible, key=lambda c: (c["scores"][policy]["predicted_update_ms"],
                                                       c["scores"][policy]["peak_gib"], c["id"])) if eligible else None

                native = best(natives)
                for mlp in ("follow_attention", "redistribute"):
                    joint = best([j for j in joints if j["mlp_policy"] == mlp])
                    n = native["scores"][policy] if native else None
                    j = joint["scores"][policy] if joint else None
                    selected = joint if j and (not n or j["predicted_update_ms"] < n["predicted_update_ms"]) else native
                    selections[f"{policy}/{'physical' if alpha is None else 'alpha2'}/{mlp}"] = dict(
                        native=native["id"] if native else None, joint=joint["id"] if joint else None,
                        selected=selected["id"] if selected else None,
                        native_ms=n["predicted_update_ms"] if n else None, joint_ms=j["predicted_update_ms"] if j else None,
                        native_peak_gib=n["peak_gib"] if n else None, joint_peak_gib=j["peak_gib"] if j else None,
                        throughput_gain_pct=100 * (n["predicted_update_ms"] / j["predicted_update_ms"] - 1) if n and j else None,
                        unresolved_native_capacities=len(errors))
        row = dict(case=name, lengths=list(lengths.values()), pattern=pattern,
                   input_tokens=sum(lengths.values()), native=natives, joint=joints,
                   native_unsupported=errors, selections=selections)
        rows.append(row)
        (args.output / f"{name}_plans.json").write_text(json.dumps(plans, separators=(",", ":")) + "\n")
        print(json.dumps(dict(case=name, selections=selections)), flush=True)
    result = dict(scope="CPU finite constructed candidates, not optimal joint search, GPU admission or online performance",
        capacities_Ki=CAPACITIES, model_profile="configs/joint_16gpu_measured.json", gpu_launches=0,
        primary_layout="MLP follows attention; zero intra-GDN and intra-GQA layout changes",
        policy_scope="Both CP16 policies are sensitivity assumptions, not confidence bounds",
        comparison_scope="Best representable N in finite grid; unsupported mixed partitions are explicitly unresolved",
        model_budget_gib=base.budget / 2**30, cases=rows)
    (args.output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    write_report(args.output, result)


if __name__ == "__main__":
    main()
