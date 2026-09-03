# TMF 1F1B Roadmap

> 更新：2026-09-02
> 当前里程碑：32K、M=64 的五 launch 消融和 7 组 Nsight Systems 已完成。CP1/CP2 完整 layout 相对 static CP4 分别缩短约 17.7%/4.3%，但 Triton layout 融合只有约 ±0.04%，逐 lane 前向 overlap 只有约 0.25%–0.35%。当前优先验证 lane 通信聚合粒度，随后转向反向 exposed transition 和 macro `M/K`/stage tail；out-proj 切 chunk 与 NVSHMEM 降为有证据才触发的条件项。

## 当前实现与完成状态

实现位于 `1f1b/`。TMF Adapter 统一管理 logical identity、pack/split、CP transition、共享 autograd graph 和 PP P2P；scheduler 使用 canonical non-interleaved 1F1B 顺序，并在一个 cohort 的输出梯度到齐后执行一次 grouped backward。fixed CP 时 `K=1`，退化为标准 1F1B。

| 里程碑 | 状态 | 已验证范围 |
| --- | --- | --- |
| 计划层与静态约束 | 完成 | K、M/K、mesh、stage boundary、非法 transition |
| fixed-CP 1F1B | 完成 | PP2、M=8、K=1、8F/8B、P2P 和 gradient sync |
| macro-cohort TMF 集成 | 完成 | CP4→CP1→CP4、K=4、成组 P2P、一次 cohort backward |
| mixed stage width | 完成 | stage 0 CP1、stage 1 CP2，logical identity 保持 |
| 数值正确性 | 完成 | calibrated GPU PASS；output/loss、input/parameter gradients 通过冻结门禁 |
| 分阶段 profiling | 完成 | 5 warmup + 10 measured，8 rank 数据完整 |
| Triton transition layout kernel | 完成 | 16/16 CUDA oracle、4K distributed smoke、32K 五 launch 消融和 Nsys kernel 存在性均通过 |
| CP 收缩前向 lane overlap | 功能完成，收益有限 | 正确性通过；Nsys 确认 NCCL 与后续 producer Attention 重叠，但端到端仅改善约 0.25%–0.35% |
| 前向 lane group 粒度参数 | 本地代码完成，待远端验收 | CP4→CP1 测 `1/2/4`，CP4→CP2 测 `1/2`；用于权衡 enqueue 碎片与 overlap 窗口 |

当前主要稳定性能证据来自
`agent_run/20260902T080428Z_a7c4d09_tmf_0831_perfopt_g4/`。固定 32K、M=64、
PP2、每 stage 5 层、5 warmup + 10 measured，每 case 5 个独立 launch：

| 配置 | median worst-step | 相对 static CP4 | 解释 |
| --- | --- | ---: | ---: |
| static CP4 + Triton/overlap flags | 6104.093 ms | 1.000× | 无 transition，作为同轮完整配置参考 |
| GDN CP1 + Triton + overlap | 5024.459 ms | 1.215×（time −17.69%） | 当前最强完整 layout |
| GDN CP2 + Triton + overlap | 5839.366 ms | 1.045×（time −4.34%） | 有正收益，但明显小于 CP1 |

DP4×CP1 和 DP2×CP2 是不同并行分解。上表只能说明完整配置的端到端表现，
不能把跨 layout 差值归因给 transition、Adapter 或单个 kernel。相同 layout 内的
2×2 消融为：

| Layout | Triton 效应 | 前向 lane overlap 效应 | 交互项 |
| --- | ---: | ---: | ---: |
| CP1 | −0.94 ms（−0.02%） | −12.57 ms（−0.25%） | −0.09 ms |
| CP2 | +2.52 ms（+0.04%） | −20.75 ms（−0.35%） | +8.63 ms |

因此，Triton 本地布局融合已经完成 correctness/工程闭环，但不是当前端到端
瓶颈；逐 lane overlap 确实生效，却只回收了很小一部分 step。下一阶段应优化
通信/计算的调度粒度或处理尚未 overlap 的阶段，而不是继续微调 pack/assemble
kernel。

历史 `agent_run/20260831T134908Z_tmf_r4_profile/` 继续作为 4K calibrated
correctness 和 profiler 演进证据，不再作为当前性能排序的主数据。

## 执行职责边界

lane-granularity 远程实验已经完成，未产生决定性提升；其临时执行计划已清理，
结论已合并到 `HETEROGENEOUS_CP_TMF_MASTER_ROADMAP_20260901.md`。0831 计划仅作为
历史执行合同，不再追加任务。远程 Agent 运行冻结代码、采集原始证据并执行有限诊断，
不负责研究设计或项目功能开发。

以下事项只在本地完成：

- 修改 profiler/schema、实现 peak-memory 和跨 launch 聚合；
- 设计并实现多步 optimizer、save/resume 与 checkpoint/recompute；
- 判断 macro bubble 是否足以触发 streaming；
- 实现 streaming Adapter、跨 stage CP、完整多维 runtime 和 planner；
- 根据远程原始数据升级可靠结论、修改 roadmap 和形成论文主张。

远程运行若暴露必须改代码的问题，应交付 `NEEDS_RESEARCHER.md` 后停止对应分支，由本地修复、复核并重新冻结版本，再启动下一轮远程测试。

## Transition 通信掩盖路线（2026-09-02）

G4 Nsys 已证明逐 lane overlap 真实发生，但也改变了瓶颈优先级：

- CP1 Triton 同步到 Triton overlap 的 NCCL SendRecv 数由每 rank 1873 增至
  1921；CP2 的现有 Nsys 对照中为 3297→3329，但 CP2 同时混入 backend 差异，
  只能作为粒度诊断信号。
- CP1/CP2 overlap case 分别观察到 8489/1721 个 transition NCCL 与后续
  producer Attention kernel 的时间区间相交，说明 overlap 不是“只 enqueue、
  实际未并发”。
- CP1 D 的 consumer wait 为 128 ranges、总 wall 39.359 ms、单 range 最大
  0.947 ms；CP2 D 为 256 ranges、总 wall 74.936 ms、单 range最大 0.416 ms。
  最后通信尾部存在，但单个 tail 已较短。
- 相同 Triton backend 下，overlap 的端到端收益只有 CP1 约 12.6 ms、CP2 约
  16.4 ms（D 对 B）；profile 中 Attention forward CUDA 时间同时出现上升，
  表明通信与计算并发可能在争用 SM、copy/NVLink 或调度资源。组件 range 可重叠，
  该现象是后续粒度实验的动机，不是可相加的损失分解。
- 全 capture 的 `cudaStreamWaitEvent`/synchronize 数包含 PP、框架和 profiler
  行为。仅凭调用数量不能认定 `cuda synchronize` 是 TMF 主瓶颈，必须先建立
  callsite 和关键路径归属。

据此将优化顺序更新为：

| 顺序 | 状态 | 工作 | Profiling 依据与退出条件 |
| --- | --- | --- | --- |
| T0 | **完成并冻结** | Triton 融合 transition 两侧 pack/layout/direct-write | 16/16 CUDA oracle 与 Nsys kernel 存在性通过；端到端约 ±0.04%，不继续投入局部 kernel 微优化 |
| T1 | **完成并保留开关** | CP 收缩按 temporal lane 提前 enqueue NCCL，consumer 前等待残余 tail | 正确性和真实时间交叠已验证；收益约 0.25%–0.35%，不能宣称解决主瓶颈 |
| T2 | **本地实现完成，待远程执行（最高优先级）** | `--tmf-transition-lane-group-size`：CP1 测 1/2/4，CP2 测 1/2 | 判断减少 NCCL group/小 kernel 是否能抵消 overlap 窗口缩短；要求五 launch + Nsys 同 layout 证据 |
| T3 | **下一本地研究项** | 精确标记并实现 backward lane/inter-cohort inverse transition overlap | `cohort_backward` 仍占约 3.35 s/5.04 s（CP1 D）和 4.10 s/5.88 s（CP2 D）的 max-rank profile；先消除 inclusive hook 歧义，再要求梯度 oracle 和关键路径缩短 |
| T4 | **并行诊断项** | `M/K`、stage tail 和 schedule simulation，决定 macro/streaming | G4 的 M=64 已有 16/32 cohorts，但 CP1 stage0/1 step 仍相差约 195 ms；必须用 M/K sweep 判断是 pipeline tail、跨 stage 等待还是局部负载差异 |
| T5 | **条件项，优先级下降** | 将 Attention/GDN out-proj 沿 sequence 切为 1/2/4/8 个通信 chunk | 当前单次 consumer tail <1 ms，继续切 chunk 的理论空间有限且可能降低 GEMM 效率；仅当 T2 后仍有稳定 exposed forward tail 才实现 |
| T6 | **条件项** | CUDA+NVSHMEM symmetric final-layout buffer、put-with-signal、GPU-side wait | 仅当 T2/T3 后确认 NCCL completion/CPU wait 本身仍在关键路径，且预计收益覆盖工程与可移植性成本时进入 |
| T7 | **后续研究项** | 跨 cohort 聚合 transition 通信或重排 segment，减少消息/transition 次数 | CP2 每 step 的 transition 实例数约为 CP1 的 2 倍（32 vs 16 cohorts）；只在延迟主导且不会显著推迟 early start 时启用 |

当前 T1/T2 的实现边界：PyTorch ProcessGroupNCCL 不公开可供用户 stream 直接
等待的 completion event，因此采用“lane/group 异步 enqueue、consumer 边界集中
`Work.wait()`”的安全路径。它能隐藏较早通信，但会引入独立通信流与计算流的资源
竞争；T2 正是为了搜索 early-start 与批量效率的折中，而不是假设粒度越细越好。

T3–T7 属于历史研究候选。T2 correctness、五 launch 和 Nsys 矩阵已经完成；
后续是否继续这些候选，以总 Roadmap 的 ContextFlow/HCP 主线及 Go/No-Go 为准。

## 下一阶段路线

| 顺序 | 执行位置 | 工作 | 退出条件 |
| --- | --- | --- | --- |
| R5a | **已完成** | G3 完成 32K/64K、M=8/16/32/64 的 120 launch；G4 完成 45 launch 消融和 7 组 Nsys | 原始日志、逐 rank profile、环境、失败样本和 SQLite 完整 |
| R5b | 本地实现 + 远程验证 | 完成前向 lane group 1/2/4 参数探索 | 4K correctness、32K 五 launch 和 Nsys 能选出 CP1/CP2 的有效粒度或证明无收益 |
| R5c | 本地研究/开发 + 远程验证 | 精确拆分 backward transition 关键路径并实现 lane/inter-cohort overlap | 非 inclusive 的时间证据、梯度 oracle 和同 layout 端到端收益同时成立 |
| R5d | 本地研究决策 | 用真实 stage depth、长序列和 `M/K` 数据解释 stage tail，做 macro/streaming Go/No-Go | 明确保留 macro，或批准 streaming 的适用 workload与验收口径 |
| R5e | 本地开发 + 远程验证 | 实现真实 optimizer 多步轨迹、save/resume；按需实现 checkpoint/recompute | static-high 与至少一个 heterogeneous macro 配置通过正确性、恢复和显存门禁 |
| R6a | 本地条件开发 | 仅在 R5d 为 Go 时实现 streaming TMF Adapter | 状态机、`K=1` 退化、mixed width 和 macro 数值 oracle 全部通过 |
| R6b | 远程测试 | 对冻结后的 streaming 与 macro 做相同 DP/CP layout、多 launch 对照 | 时间、bubble、等待和显存差异可由原始数据解释 |
| R7 | 本地条件研究 | 仅在 stage-boundary 恢复成本显著时设计跨 stage CP | boundary protocol、动态 wire layout 和反向恢复通过 |
| R8 | 本地研究/开发 | 建立 PP-aware 性能模型与 planner | held-out 配置上能预测可行性和候选排序，并报告 plan-quality gap |

```text
macro integration [DONE]
        |
        v
remote macro evidence (R5a DONE)
        |
        v
lane granularity + backward critical path (R5b/R5c)
        |
        v
macro/streaming Go/No-Go (R5d)
     /       \
 No-Go       Go
   |          |
macro default streaming implementation (R6a)
   |          |
   +------> frozen remote comparison (R6b, if Go)
                    |
                    v
        multidimensional runtime / model / planner
```

## 本地研究与开发清单

### L1：远程证据审计与实验口径

- [x] 已检查 G3/G4 的 commit/diff、环境、命令、逐 rank profile、Nsys SQLite 和失败样本；G4 P0–P5 全部 PASS。
- [ ] 只在相同 logical workload、layer mix、DP/CP/PP、M/K、dtype 和统计口径下比较结果。
- [x] 已将 128K CP1 的稳定 OOM 记录为可行域边界，没有要求远程 Agent 改 workload 或实现 checkpoint。
- [ ] 若数据字段不足，只把该轮定为诊断证据，不从缺失数据推导性能结论。

### L2：本地 profiling 与训练语义能力

- [ ] 在本地为 profile metadata 增加完整 layout、sequence、M/K、logical tokens、版本和 checkpoint 标签。
- [ ] 本地实现 worst-rank peak allocated/reserved、activation/transition/P2P buffer 生命周期和跨 launch 统计。
- [ ] 本地实现确定性多步 optimizer 对照，保存 loss、gradient norm、parameter/optimizer checksum。
- [ ] 本地实现 save/resume 与 RNG/sample identity 恢复门禁。
- [ ] 每次本地修改通过 CPU 测试和小规模正确性后冻结，远程 Agent 只验证冻结版本。

### L3：Checkpoint/recompute 条件分支

- [ ] 仅在长序列、真实 stage depth OOM 或 temporal graph 显存显著恶化时触发。
- [ ] 先比较 transition boundary 与低 CP GDN segment 的单一 checkpoint 策略。
- [ ] 报告同布局 checkpoint off/on 的可运行边界、peak memory、重算时间和 step time。
- [ ] checkpoint 后重跑数值、多步 optimizer 和 save/resume 门禁。

### L4：Streaming 研究决策与实现

- [ ] 用 R5a/R5b 的真实 `M/K`、stage tail、跨 launch 波动和显存证据决定 Go/No-Go。
- [ ] No-Go 时保留 macro 为默认路径，streaming 只作为特定小 `M/K` 的候选，不为了路线连续性强行实现。
- [ ] Go 时先在本地完成 enqueue/ready/release/resume 状态机、outstanding cohort 上限和纯 Python 测试。
- [ ] 本地 correctness 通过并冻结后，才把 macro/streaming 同布局 profiling 交给远程 Agent。

### L5：后续 idea 推进

- [ ] 只有 canonical stage boundary 的强制恢复成为显著瓶颈时，才研究跨 stage CP。
- [ ] 用同版本长序列 PP 数据替换历史先验，建立包含 compute、CP、transition、DP sync、bubble、overlap 和 memory 的模型。
- [ ] planner 在 held-out 配置上验证预测误差和相对 oracle 的 gap；预测无法区分候选时先补 profile，不增加经验规则掩盖问题。
- [ ] 多节点、TP、distributed optimizer、THD SFT/RL 和 64/128 GPU 留在总 roadmap 的 G4–G7 推进。

## 当前限制

- macro-cohort 把 pipeline scheduling unit 从 `M` 缩减为 `M/K`；32K、M=64 下 CP1/CP2 分别有 16/32 个 cohorts，但 max-rank profile 仍显示约 195/112 ms 的 stage0/1 step 差，需要 `M/K` sweep 和 schedule trace 才能归因；
- 32K 主点已有五 launch 稳定数据，但当前只覆盖单机 8×H800、PP2、固定 5 层 pattern；还不能外推多节点、真实模型深度或论文主结果；
- Triton layout 融合与逐 lane overlap 的增量都小于 0.4%；前向最后 consumer tail 已较短，进一步收益更可能受通信粒度和计算/通信资源竞争限制；
- backward transition 仍同步执行，现有 hook range 为 inclusive diagnostics，尚缺少严格不重叠的 backward 关键路径分解；
- 128K、M=8 的 GDN CP1 在 stage 0 稳定 OOM，而 static CP4/CP2 可运行；checkpoint 与 graph/buffer 生命周期仍是生产长序列的必要分支；
- 当前不支持跨 PP stage CP 切换；
- 当前验证不覆盖 checkpoint/recompute、多节点、真实 optimizer update 多步轨迹和训练恢复；
- Adapter 开销必须使用相同 per-layer DP/CP layout 的 macro/streaming control 量化。

## 保留的分析证据

- `TMF_RELIABLE_CONCLUSIONS_AND_AGENT_RUN_EVIDENCE_20260901.md`：逐次 `agent_run` 的证据等级、可靠结论及 roadmap 对应关系；
- `TMF_CP_TRANSITION_COMMUNICATION_ANALYSIS_20260829.md`：CP transition 通信量模型；
- `TMF_CORRECTNESS_INITIAL_RUN_ANALYSIS_20260830.md`：前序 correctness 闭环；
- `TMF_1F1B_R4_PROFILING_IMPLEMENTATION_AND_SLOWDOWN_ANALYSIS_20260831.md`：R4/profiler 实现和慢点演进；
- `TMF_1F1B_RUN2_R4_ANALYSIS_20260831.md`：v2 数值门禁分析；
- `TMF_1F1B_RUN3_PROFILE_ANALYSIS_20260831.md`：最终正确性、性能结果和慢因结论。
- `agent_run/20260901T133800Z_a7c4d09_tmf_g3_remote_macro/REPORT.md`：32K/64K、M sweep 和 128K CP1 OOM 证据；
- `agent_run/20260902T080428Z_a7c4d09_tmf_0831_perfopt_g4/REPORT.md`：Triton/overlap 五 launch 消融与 7 组 Nsys 证据；
- lane group 远程验证已完成，结论已进入总 Roadmap；临时执行计划不再保留。
