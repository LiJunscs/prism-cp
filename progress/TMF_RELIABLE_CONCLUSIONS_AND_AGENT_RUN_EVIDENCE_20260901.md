# 知识库主题：TMF 可靠结论与 agent_run 证据索引

> 更新：2026-09-01  
> 对应路线图：`progress/TMF_1F1B_ROADMAP_20260831.md`  
> 下一步执行计划：`progress/REMOTE_AGENT_TMF_1F1B_PLAN_20260831.md`  
> 范围：审计当前 `agent_run/` 下 9 次已归档运行，区分可长期使用的结论、仅用于实验设计的先验，以及已被后续证据取代的判断。

## 1. 证据使用规则

本文按以下等级使用实验结果：

| 等级 | 含义 | 可以支持的判断 |
| --- | --- | --- |
| A：当前冻结证据 | 当前 1F1B/TMF 代码路径、环境和原始日志完整，正确性门禁或稳态采样完成 | 当前已实现范围的正确性；下一阶段是否可以开始 |
| B：历史重复证据 | 旧实现或 dirty patch，但多 seed、多 launch 或长序列结果一致 | 机制方向、实验范围和优化优先级 |
| C：诊断证据 | 单次、冷启动、短序列、失败门禁或已被修复的实现 | 定位问题与设计后续门禁，不作为最终性能数字 |

所有性能比较必须写完整并行布局。`DP4×CP1`、`DP2×CP2` 和 `DP1×CP4` 是不同的并行分解；DP 更大的配置本来就可能更快，不能把它们之间的差值归因为 Adapter 开销或实现回归。最终性能结论还要求相同 layer mix、相同 DP/CP layout、相同 logical workload、相同统计口径和多个独立 launch。

## 2. 当前可冻结的可靠结论

1. **macro-cohort TMF 已在当前冻结范围内完成 1F1B 集成。** 已验证范围为单机 8×H800、PP=2、M=8、MBS=1、`K=1/4`、stage 内 `CP4→CP1/CP2→CP4`。logical identity、canonical non-interleaved 1F1B 事件、逐 tensor PP P2P、grouped cohort backward、参数梯度同步和非法配置提前失败均闭环。该结论不覆盖跨 stage CP、checkpoint/recompute、多节点、真实 optimizer update 后的多步轨迹和训练恢复。
2. **当前数值正确性是校准后的 BF16 backward 等价，不是所有梯度逐位一致。** 两个 TMF case 的 output/loss bit-exact，BF16 与 FP32 shadow parameter gradient 均通过；stage-0 input-gradient rel-L2 约 0.739%，与同一 launch、同一 all-CP4 配置的 reference replay 本底约 0.742% 相当，并受 1% hard cap 约束。因此当前冻结范围可判为 calibrated PASS，同时必须保留 `strict_passed=false` 这一事实。
3. **在固定全局 workload 下，GDN 使用更低 CP、相应更高 DP 的收益具有跨 seed 和长序列重复性。** 历史无 PP 实验在 sequence=65,536 上持续得到 `CP4→CP1` 最大、`CP8→CP2` 次之、`CP8→CP4` 很小的收益；三 seed、45 次独立 launch 中排序没有反转。这支持优先把低 CP 分配给 GDN，但不能直接作为当前 1F1B 实现的最终 speedup。
4. **Attention 与 GDN 对 CP 的敏感性不同。** 固定全局 workload 的单层 profile 中，GDN 从 CP1 到 CP8 的时间惩罚在长序列上约为 1.65–1.70×；Attention 的高 CP 惩罚随序列增长相对减弱。它支持 heterogeneous CP 的层类型选择，但不等于 ring attention 通信在真实系统中完全免费。
5. **transition 路径和次数会侵蚀 mixed CP 收益。** sequence=65,536 的独立 transition 测量约为 CP4↔1 9.1 ms、CP8↔2 11.0 ms、CP8↔4 10.4–10.6 ms；三组历史 stack 中，转换较少的 pattern 始终更有利。planner 应按整条路径累计 transition 与布局代价，不能只比较某一条边，也不能把这些旧测量当作新的硬件常数。
6. **temporal bundle 是 MBS=1 下实现低 CP 语义的有效路径，repeat 只能作为 control。** bundle 保持 logical sample 多样性；repeat 会重复物理执行相同 logical sample，虽然 elapsed time 接近，但 unique throughput 会按 1/K 降低，不能报告为训练吞吐提升。
7. **temporal 路径的主要显存风险来自 backward activation graph。** 历史长序列 profile 中 temporal mixed 相比 fixed 增加约 4.1–4.8 GiB peak allocated，而 pack/transition 临时缓冲并非峰值根因；更长 stack 在 98K/131K sequence OOM。后续长序列、真实 stage 深度实验必须把 activation checkpoint/recompute 和 peak allocated/reserved 纳入门禁。
8. **现有 PP profiling 只证明 profiler 可用并提供短配置诊断，不代表生产性能。** sequence=4096、每 stage 3 层、单个 launch 的稳态结果为 static `DP1×CP4` 195.861 ms、GDN `DP4×CP1` 170.472 ms、stage-1 GDN `DP2×CP2` 193.070 ms。它说明三种路径都能稳定采样，也显示 macro `K=4` 将 pipeline units 从 8 降到 2 后出现约 21 ms stage 尾部等待；但序列太短、stage 太浅、launch 数不足，不能用于论文或 planner 标定。
9. **macro-cohort 的 bubble 是否值得用 streaming 消除，取决于真实的 `M/K`。** 当前短 profile 的 M=8、K=4 只有 2 个 pipeline units，是有意放大的不利场景。百卡和千卡训练常有更大的 global batch，M 往往高于 PP size 数倍，因此 R5 streaming 是需要同布局实测的候选优化，不应预设它在生产配置中一定显著优于 macro。
10. **目前没有最终性能模型。** 现有模块 profile、transition 通信量公式和 `model_no_pp_cp.py` 只适合筛选实验，例如比较 `8→1→1` 与 `8→4→1`；最终模型必须由真实长序列、真实 stage 深度、相同 layout、多个 launch 的数据校准，并显式包含 DP/CP compute、GDN all-to-all、transition、PP bubble 和显存约束。

## 3. 每个 agent_run 的证据结论

### 3.1 `20260815T075123Z_b0226ace2_fix1` — B 级，首次真实尺度闭环

首轮真实尺度运行暴露了 sequence=32 小测试没有覆盖的 transition gather bug 和 world-size 检查问题；修复后 P0 16/16 通过，并完成 15 次 sequence=65,536 的独立 mixed/fixed launch。三组 pattern 的 median speedup 分别为 1.2481×、1.0688×、1.0230×，形成 A>B>C 的初始排序。该运行能证明真实尺度 mixed 路径可执行，并为后续 pattern 选择提供先验；由于使用旧 commit/dirty patch、无 PP、`dp_grad_sync=false`，不能作为当前实现的最终正确性或性能结论。

### 3.2 `20260815T120648Z_b0226ace2_seed_sweep` — B 级，多 seed 重复性

在 seed 1234/2345/4321 上累计 45 次独立 launch。A pattern 的 speedup 为 1.2379–1.2573×，B 为 1.0609–1.0733×，C 为 1.0161–1.0272×，A>B>C 没有反转；2-transition、3-transition、4-transition 三种已测 schedule 的收益依次减弱。它可靠支持“低 GDN CP 的方向性收益可跨 layer placement 重复”和“transition 数量需要进入 planner”。但只有三个 schedule，不能把转换次数与收益的关系外推成普遍单调定律；运行仍无 PP、optimizer 和 DP gradient sync。

### 3.3 `20260816T070154Z_36b8f3172_post_refactor` — B 级，长序列模块基线

P0 后续闭环后，forward exact，BF16 gradient relative-L2 约 0.3%–0.6%、cosine 约 0.99999。固定全局 workload 的单层 profile 覆盖 sequence=16K/65K/262K、CP1/2/4/8：在 65K 上 Attention 为 224.13/232.85/247.36/265.80 ms，GDN 为 83.25/121.97/132.60/140.79 ms；在 262K 上分别为 3232.03/3306.65/3449.91/3456.03 ms 和 340.15/485.84/522.32/560.17 ms。65K 的 8 层 stack 再现约 1.246×、1.067×、1.021× 的 A/B/C 排序，并给出 transition 约 9–11 ms 的量级。它是当前最有价值的长序列实验指导，但每个 frontier 点只有一次 launch、无 PP/DP sync，且 98K/131K stack OOM，因此仍不是最终模型数据。

### 3.4 `20260822T042129Z_tmf_proto` — B/C 级，temporal 原型可行性

该运行首次比较 temporal bundle 与 repeat。sequence=65,536 的 bundle 在旧原型上得到 A 约 1.283×、B 约 1.213×；repeat elapsed time 接近 bundle，但 unique throughput 分别降为 1/4 和 1/8。temporal mixed 的 peak allocated 比 fixed 高约 4.1–4.65 GiB，说明 temporal 语义虽然能解决 MBS=1 的 CP/DP 整除问题，却引入显著 activation graph 持有成本。方向性和显存诊断可保留；原型期间修过 direction table/world-size 问题，且是 dirty 实现，具体 speedup 只作历史先验。

### 3.5 `20260822T100108Z_tmf_validation_profile` — B/C 级，primitive 正确性与显存根因

CP4↔1、CP8↔2、CP8↔1 的 transition primitive 在不同 dtype、seed、连续性和 table/formula 路径上完成 216/216 exact 检查。profile 将 temporal 额外显存定位到 backward activation graph：pack/transition buffer 小于约 1.5 GiB，不足以解释 4.1–4.8 GiB 的 peak allocated 增量，checkpoint 低 CP GDN 输出或 transition 输出是合理方向。该报告当时对 bundle/repeat 给出的整体 correctness “Go” 后来被 8 月 30 日更严格的 input/parameter-gradient 门禁推翻，因此只保留 primitive mapping、显存位置和 control 语义结论，不保留宽泛的训练正确性结论。

### 3.6 `20260830T081917Z_tmf_repeat_fixed` — C 级，负向门禁证据

严格 correctness gate 确定性失败，性能测试按计划没有启动。legacy fixed-vs-mixed 的 output/gradient 失败；bundle/repeat 的 output 通过，但 input gradient 与逐参数比较失败，replica consistency 通过。这次运行证明早期“整体正确性已确认”的说法不可靠，并建立了“数值门禁通过前禁止性能结论”的执行规则。它不证明 TMF 语义本身错误，因为后续发现 harness、BF16 reduction/reference 和实现版本共同影响结果；当前冻结范围由 8 月 31 日 R4 replay 证据取代。

### 3.7 `20260831T083742Z_tmf_1f1b` — A/C 级，1F1B 结构闭环

fixed `K=1`、TMF `K=4`、stage 0 CP1/stage 1 CP2 mixed width 和 6 个负向启动 case 全部通过，无死锁。日志确认 `K=1` 有 8 个 pipeline units，`K=4` 有 2 个；CP1 每 cohort 执行一个 width-4 payload，CP2 执行两个 width-2 payload；logical IDs、P2P tensor 数、remesh width 和 grouped backward 次数符合设计。此轮只做 finite gradient，没有独立 reference 数值对照；4.45 s、10.93 s、29.78 s 都是单个冷 step，只能用于发现 CP2 首次 backward/JIT 等等待链，不能比较性能。

### 3.8 `20260831T105739Z_tmf_1f1b_v2` — A/C 级，R4 严格门禁与本底发现

A1–A4 再次通过。两组 R4 的 output/loss bit-exact，BF16 与 FP32 shadow parameter gradient 通过，但 stage-0 input-gradient rel-L2 为 0.737%/0.735%，超过预设 0.5%，因此按原规约记 FAIL 并跳过 profiling。all-CP4 identical-config 自对照本身也有约 0.354% input-gradient 和 0.333% param-gradient rel-L2，且 TMF 重跑完全确定，说明固定阈值低于 BF16 backward 重放噪声尺度。本轮的可靠贡献是发现并量化 harness 本底，而不是宣布正确性失败或给出性能结果。

### 3.9 `20260831T134908Z_tmf_r4_profile` — A 级，当前冻结正确性与 profiler 基线

R4 增加同一 launch、相同 all-CP4 layout 的 reference replay。CP1/mixed CP2 candidate 的 input-gradient rel-L2 为 0.7384%/0.7392%，replay 为 0.7417%/0.7421%，candidate/replay 为 0.9956/0.9961；两者均低于 1% hard cap，output/loss bit-exact，BF16 与 FP32 shadow parameter gradient 通过，所以记 `passed=true`、`strict_passed=false`、`calibration_applied=true`。三种 layout 各完成 5 warmup+10 measured steps，8 rank 数据完整，证明 profiler 和当前 macro-cohort 集成能稳定工作。195.861/170.472/193.070 ms 只是一组 sequence=4096、每 stage 3 层、单 launch 的配置结果，CP1 与 CP2 又同时改变 DP，不能外推为生产 speedup 或 Adapter overhead。

## 4. 对当前 roadmap 的支持与约束

| Roadmap | 已有证据支持 | 继续执行时的硬约束 |
| --- | --- | --- |
| R5 streaming TMF Adapter | R4 已确认 macro 语义、logical identity、反向依赖和 PP P2P，可作为 streaming oracle | 保持 logical microbatch 级 PP 调度；`K=1` 退化；与 macro 做 output/loss/input-grad/param-grad 同口径对照。把 streaming 视为待验证优化，不预设真实大 M 下一定有收益 |
| R6 same-layout profiling | profiler 已能分离 step、stage、forward、cohort backward、transition、P2P 和 grad sync | macro/streaming/repeat 必须使用相同 DP/CP layout；至少 5 个独立 launch。正式实验转向长 sequence、每 stage 约 5 层或真实模型深度、较大 M，并记录 checkpoint 与 peak memory |
| R7 cross-stage CP | 当前 plan 已明确禁止并能在 distributed init 前拒绝非法边界 | 只能在 R5/R6 正确性和同布局比较完成后引入；必须定义动态 wire layout、forward/backward boundary cohort protocol 和恢复顺序 |
| R8 model/planner | 长序列模块 profile、transition 测量和 temporal 显存诊断提供初始参数与搜索方向 | 分析模型只用于选实验；最终拟合需新版本多 launch 数据，并同时建模 DP/CP、GDN all-to-all、transition、PP bubble、显存和 checkpoint |

当前 roadmap 可以继续从 R5 开始，因为 R1–R4 的冻结范围已有 A 级证据闭环。R5 的退出条件不能只是代码可运行；它还要回答真实大 `M` 下 streaming 相对 macro 的 bubble 收益是否足以抵消状态管理、等待和显存成本。若长序列、真实 stage 深度和大 `M` 的同布局实验显示 macro 已足够接近最优，roadmap 应允许保留 macro 为生产路径，并把 streaming 降为特定小 `M/K` 场景优化。

## 5. 后续结果进入本文的最低要求

每个新 `agent_run` 至少保存 commit、dirty diff、代码 hash、完整命令、环境/GPU 拓扑、逐 rank 原始日志、汇总脚本版本和失败样本。正确性先于性能；BF16 数值门禁必须同时报告 fixed threshold、同 launch replay 本底和 hard cap。性能结果必须报告完整 DP/CP/PP/M/K/sequence/layer mix、logical tokens、warmup/measured steps、独立 launch 分布和 peak memory。未满足这些字段的运行仍可作为 C 级诊断证据，但不能改变本文的 A 级结论或 roadmap 里程碑。
