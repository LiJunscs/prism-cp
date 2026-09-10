# 知识库主题：异构注意力并行的实验口径、已有证据与下一步实验计划

归档日期：2026-09-10。

归档范围：本轮关于固定 microbatch 扩展性、固定 GPU/全局负载训练、已有混合 block 结果筛选及后续实验安排的分析。以下保存完整计划快照及关键统计，便于后续独立阅读。本次仅完成分析和归档，未运行新增 GPU 实验。

项目根目录：`E:/LLM论文/mine/heterogeneous_cp`。正文中的相对路径均相对于该目录；知识库副本中的路径同样指向该项目，不相对于知识库。

## 当前采用的结论与研究约束

- 不继续将 TMF、cohort 或 GDN headwise 作为新方案主线；既有数据保留为历史证据。
- CP 曲线区分三种口径：固定 B/S 的单样本扩展；固定 W/G、A=1、调整 B 的形状效率；固定 W/G/B、调整 A 的完整训练。三者不能混算吞吐收益。
- 当前优先缺口是完整训练的最佳统一布局、峰值时刻显存组成，以及 CSA 异常的因果归因，不是更多 CP 组合。
- 单模块 latency 的最优配置不等于固定 GPU 预算下的最优吞吐。异构 mesh 必须扣除数据转换、同步及额外驻留成本，并比较最优可行统一基线。
- 候选机制由实际瓶颈决定，可以涉及算子工作集、checkpoint、FSDP 参数生命周期、FFN/输出头或局部 TP，不以 CP 为预定中心。
- CSA 零辅助损失系数下仍计算部分 teacher/score 的现象有历史源码依据；它对异常的定量贡献尚待 profile，不能记为已证实根因。
- 历史 H800 单模块数据与当前 H100 完整训练分开；用户报告的 FSDP 数字尚不等于完整独立运行档案。

## 归档组成与可复核入口

- 完整执行计划原件：`progress/ATTENTION_TRAINING_EXPERIMENT_PLAN_20260910.md`。
- 审计脚本：`analysis/attention_evidence_20260910/audit_existing.py`。
- 成功退出的 30 条 MBS1 launch：`analysis/attention_evidence_20260910/accepted_mbs1_launches.csv`。
- 12 格聚合结果：`analysis/attention_evidence_20260910/mbs1_aggregates.csv`。
- 1 条退出状态未确认的日志指标：`analysis/attention_evidence_20260910/unconfirmed_log_metrics.csv`。
- 单独保存的 8 条旧固定全局负载数据：`analysis/attention_evidence_20260910/legacy_fixed_global_64k.csv`。
- 前序外部研究材料：`progress/research_sources_20260910/`，包括 Megatron MoE、LoongTrain、DistCA 等读取记录。该目录作为前序讨论的材料索引，本次没有重新验证这些外部资料的所有论断。

本快照中的计划代表本轮最新取舍。`FSDP_COHORT_CONCLUSIONS_20260910.md` 等旧文档保留历史语境，其中继续探索 cohort/head 切分的建议不作为当前待执行任务。

---


# 已有注意力实验筛选与下一阶段执行计划

日期：2026-09-10。范围：读取已有归档、重算关键统计、制定增量实验；本次未执行 GPU 实验，未修改训练实现。

## 1. 决策摘要

下一阶段不再补一张全面的 CP 扩展性大表。已有数据已覆盖大量单模块与混合栈配置，真正缺失的是：

1. CSA 异常是否来自当前实现的额外训练分支，而非注意力本身的并行性质。
2. 固定 MBS 的算子收益，经过梯度累积、FFN、模型状态通信和激活驻留后，还剩多少。
3. 在相同训练目标与资源预算下，最优统一布局是否确实输给一个值得实现的异构方案。

优先建立第三项的基线，并用前两项解释原因。TMF、cohort、GDN headwise/二维 head 切分不列入新候选。旧 roadmap 中与此冲突的条目视为历史记录，不作为本计划的依赖。

## 2. 已有证据：保留什么，能说明什么

### 2.1 固定 MBS=1：曲线基本已有

以下为 S=65536，W=8，B=1，A=1，D=8/C。每个 CP 组各处理一条独立样本；不同 C 下全机样本数不同。它们是并发副本环境中的单样本扩展测量，不是“只启动一个 CP 组，其余 GPU 空闲”的隔离测试。

| 模块 | CP1 ms | CP2 ms | CP4 ms | CP8 ms | CP8 延迟加速 | CP8 扩展效率 | CP8/CP1 每样本 GPU 时间 |
|---|---:|---:|---:|---:|---:|---:|---:|
| GDN chunkwise | 85.346 | 54.120 | 29.046 | 16.511 | 5.169× | 64.6% | 1.548× |
| CSA | 552.480 | 639.990 | 536.090 | 479.855 | 1.151× | 14.4% | 6.948× |
| HCA | 140.595 | 69.055 | 36.440 | 19.985 | 7.035× | 87.9% | 1.137× |

统计为独立 launch 的均值再取中位数，不将迭代内 std 当作独立重复。GDN 每格 n=3；CSA/HCA 各 CP1/2/8 n=2、CP4 n=3。

数据审计发现 HCA CP8 launch3 日志有 19.51 ms，但 summary.tsv 未记录该次退出状态。正式表只采用 summary 中 rc=0 的两次，中位数 19.985 ms；若纳入第三条指标，中位数为 19.73 ms。两种口径不能不加说明地混用。无需仅为补齐 n=3 重跑旧版本。

结论：GDN/HCA 的 CP 降低了单样本延迟和显存，但未自动增加固定预算下的吞吐；CSA 的 CP2 延迟和显存变化尤其值得归因。不能据此把某个 CP 数写成模块固有最优值。

来源：

- `agent_run/20260904T152108Z_manual_gdn_chunkwise_mbs1_cp/summary.tsv`
- `agent_run/20260904T085544Z_manual_csa_hca_cp_profiles/summary.tsv`
- 重算文件：`analysis/attention_evidence_20260910/mbs1_aggregates.csv`

### 2.2 固定 W、固定一次前反向的全局负载：也已有，勿重复扫描

8 月 16 日 `dp_cp_frontier/merged.csv`：W=8，G=8，A=1，因此 B=C。64K 结果：

| 模块 | CP1/B1 | CP2/B2 | CP4/B4 | CP8/B8 |
|---|---:|---:|---:|---:|
| GQA ms | 224.130 | 232.855 | 247.357 | 265.804 |
| GDN，旧 headwise，ms | 83.251 | 121.967 | 132.598 | 140.786 |

两者在这些点上都偏好 CP1，不能由“敏感性不同”直接推出“最优布局冲突”。这批数据没有 DP 梯度同步、optimizer；旧 headwise 结果不得与 9 月 chunkwise 曲线拼接。它适合筛选问题，不是当前实现的性能基准。

CSA/HCA 的固定 G 数据同样已有。32K 的 2CSA+2HCA 栈：最优固定 CP1 为 481.35 ms，CSA1/HCA2 为 482.24 ms，显存从 41.14 增至 42.15 GiB；64K 的五种固定/混合配置均 OOM。这证明被测候选尚无端到端收益，不能用较慢固定 CP8 作为主要比较对象。

该“混合栈”只包含注意力模块，不能称为包含 FFN、输出头和优化器的完整 decoder 训练。其固定 G 模式每卡 token 总量不随 CP 减少；OOM 不构成固定 B1、增加累积次数的训练必然 OOM 的证据。

来源：

- `agent_run/20260816T070154Z_36b8f3172_post_refactor/dp_cp_frontier/merged.csv`
- `agent_run/20260903T123605Z_6215878_f1b24ec5c_dsv4_csa_hca_thd_cp/summaries/mixed_comparison.csv`
- 同目录 `isolated_confirmation_launches.csv`；跨 CP 加速比从原始耗时重算，不采用 `isolated_aggregate.csv` 中含义不适合跨 CP 比较的 `relative_to_fastest`。

### 2.3 混合 GDN/GQA：有稳定性证据，但缺最优统一基线

8 月 15 日三 seed 结果中，部分混合布局相对固定 CP4 有约 1.24–1.26× 收益；8 月 16 日重构后也保留方向。但它们不含完整训练，且缺同版本、同工作量的完整混合栈固定 CP1 对照。

因此保留“相对指定 CP4/8 基线有效”的结论，撤掉“已胜过最优统一并行”的外推。若下一阶段确实研究该混合栈，应成对重跑当前版本的最佳统一布局和候选，不能仅补一个新版本 CP1 再与旧数相除。

### 2.4 FSDP 与 PP：作为边界证据

用户报告的 8×H100、128K、14 层、B1/G32、FSDP 训练约 52.1 s/step，以及低 CP 路径 OOM，是最贴近目标的证据，但缺完整独立运行档案。GDN 申请 BF16 `[1,65536,64,128]` 失败只定位了最后一笔 1 GiB 申请，不能识别此前显存由模型状态、跨层激活还是算子 workspace 占用。

PP G3/G4/G5 大量实验可作为停止旧路线的依据；layout Triton 化、overlap 和 lane 微调没有给出实质收益。本阶段不复测这些矩阵。旧 nsys 的 NCCL 汇总占比包含等待/重叠，不等于可消除的关键路径通信比例。

## 3. 三种测量制度必须分别记账

令 W 为 GPU 数，C 为 CP，D=W/C（本阶段 TP=PP=1），B 为每个 DP 副本每次 microbatch 的完整样本数，A 为梯度累积次数，G 为每个 optimizer step 的完整样本数。

`G = D × B × A`，等长样本情况下每 rank 每次前反向的 token 数为 `B × S / C`。

| 制度 | 不变量 | 随 C 变化 | 用途 |
|---|---|---|---|
| R1 单样本扩展 | B、S；注明全机并发方式 | 本地 token 减少 | 算子延迟、工作集、通信/形状归因 |
| R2 一次前反向固定负载 | W、G、S，A=1 | B=GC/W | batch 与 sequence 形状变化；已有很多数据 |
| R3 真实训练固定负载 | W、G、S、B=1 | A=GC/W | 最终比较；当前主要缺口 |

R1 报告 `T1/TC`、`T1/(C×TC)`、`C×TC`。若 CP1 OOM，以最小可行 C0 为参考，效率为 `C0×T(C0)/(C×T(C))`，不得编造 CP1 外推值。

R1 粗略映射 R3：`Tstep_compute ≈ A×Tmicro = G/(WB)×C×Tmicro`。这是解释方向的估计，不包括真实同步、optimizer、FSDP 参数生命周期及状态驻留。最终必须实测 R3。固定 W 时端到端 tokens/s 与 step time 是同一工作量下的等价指标。

## 4. 执行顺序和增量矩阵

### P0：证据整理与实验身份冻结

本次已完成：30 条 summary rc=0 的 MBS1 launch 重算为 12 格；1 条仅日志指标单独列出；8 条旧 64K 固定 G 数据单独保留。审计脚本不修改原始归档。

首次远端运行前须补齐：代码 SHA 和未提交 diff、容器、GPU/互联、驱动/TE/FLA/FlashMLA 版本、完整命令、层结构、Q/K/V 头与维度、CP 算法、loss、checkpoint、FSDP 参数组和精度。历史 GDN 32 个 value head 不等于 OOM 配置中的 64 个 value head。

H800 上的机制曲线和 H100 上的训练曲线分别成表。新 kernel 版本只成对比较新 baseline/treatment，历史结果只负责选择采样点。排除 contaminated 目录、失败正确性门禁的运行、只 forward 的结果和与目标不同的损失语义。

### P1：先建立生产基线并定位峰值，而非先写 remesh runtime

目标：回答当前 128K/14 层训练的显存墙组成，以及统一布局的可行范围。

固定当前真实模型、W8/H100、S131072、G32、B1、TP=PP=1、BF16/FSDP。从真实命令冻结 checkpoint 与所有 loss 设置；未知设置不能用 benchmark 默认值补齐。

| 探测顺序 | CP | DP | B | A | 每 rank 每 microbatch token |
|---|---:|---:|---:|---:|---:|
| 1 | 8 | 1 | 1 | 32 | 16384 |
| 2 | 4 | 2 | 1 | 16 | 32768 |
| 3 | 2 | 4 | 1 | 8 | 65536 |
| 4，条件执行 | 1 | 8 | 1 | 4 | 131072 |

先做最多四个容量探测，各须到达 optimizer 状态初始化后的完整 step；首步能运行不能代替稳态可运行。若更低 CP 已超过容量且事件记录能解释，停止继续下探。不要靠临时改 B/G 或偷偷加重计算把 OOM 点变为成功点；新重计算策略另开 treatment。

对 CP8 与最低可行 CP（若只有 CP8 可行则只做 CP8）捕获独立显存诊断：模型/optimizer 初始化后、每层前后、重计算前后、GDN 核心前后、FFN、输出头/loss、反向末尾、optimizer 前后。记录 FSDP 当前参数 gather、prefetch、梯度缓冲与 allocator snapshot。

分类要求：常驻状态、跨算子保存激活、当前工作集、通信缓冲、分配器余量。保存激活按底层 storage 去重；不能把多个 view 或同时重叠的计数相加。CPU hook 只能辅助生命周期定位，不能当 GPU 核心耗时。

产出：同负载统一布局可行表、峰值时刻的 live-set 组成、每 step 的 gather/RS 次数。若 FSDP 分片组随 DP/CP 改变，显式记录，结果解释为整套配置比较，不能将收益全部归因 CP。

### P2A：CSA 异常归因；先 3 个 profile，确认后才做消融

已有基准配置将 `dsa_indexer_loss_coeff=0.0`、`dsa_indexer_use_sparse_loss=False` 写死。冻结的 MCore `f1b24ec5c` 中，`use_indexer_loss` 由 training/grad 与 top-k 决定；fused 路径中 dense teacher/score 等计算出现在 loss 系数检查之前。CP1 路径也有类似行为。

这是源码支持的疑点，尚无 profile 能证明它解释了多少 CSA CP2 退化。也可能有全局 max-seqlen、padding、压缩、kernel 形状或通信问题，须分别验证。

第一步：沿用 64K/B1 的明确历史配置，CP1/2/8，各 1 次独立 profile，共 3 次。捕获投影、压缩、indexer/top-k、teacher/score、稀疏 core attention、布局整理、collective、反向；同步观察真实张量形状、临时显存和 max-seqlen 参数。初始不补 CP4，因为 CP1→2 异常与 CP8 端点更能区分原因。

若 teacher 路径确实消耗可观时间/显存，再实现“λ=0 时跳过无用辅助 teacher”的受限对照：CP1/2/8 × 原路径/跳过路径 × 3 次独立运行，共 18 次无 profiler 的稳态测量。若负担可忽略，停止此分支，改测 profile 指向的主要项；不要预设答案。

前提与正确性：

- 不删除决定稀疏注意力选点的 indexer/top-k；仅跳过不影响该目标的辅助计算。
- 保持主输出、top-k、主 loss、所有应有梯度及一次 optimizer update 一致；尤其检查辅助参数的 zero grad 与 None，二者在 weight decay 下可能不等价。
- 小尺寸参考覆盖各目标 CP，并至少做一个 64K/CP2 目标形状检查；不能由 CP2/4 的旧门禁声称 CP8 已验证。
- 数值容差先用当前相同路径的重放噪声及现有测试规范确定，不在看到速度后放宽。
- 实际训练若 λ>0，需另按真实 λ 测量。dense/sparse 辅助 loss 是不同语义的配置，不能将切换目标后的速度当作等价优化。先筛 CP1/8；只有目标需要两种 loss 比较时，追加 2 CP × 2 loss × 3 launch=12 次。

当前 harness 没有上述 loss/bypass/profile 全部参数；需要先显式暴露配置并写入日志。这里列的是待实现接口与实验条件，不是假称已有的可运行 CLI。

### P2B：GDN 扩展性只补原因，GQA 只补当前锚点

GDN 64K chunkwise 的 12 次测量直接复用，不重新全扫 CP。首先 CP1/2/8 各 1 次 profile：核心 kernel 的 fwd/bwd/recompute、跨 chunk 状态、collective、头/序列形状和峰值工作集。目标是解释 GPU 时间从约 85.35 增至 132.09 GPU-ms 的来源。

若单 CP 组隔离运行与 W8 副本并发可能改变结论，仅补 CP1/2 的隔离对照；不能把副本并发的网络竞争当作纯算子缩放规律。无证据时不展开新的隔离全矩阵。

GQA 在新版本/生产形状缺配对锚点：先测 64K 的 CP1/8，各 3 次，共 6 次；若有 OOM 或中间拐点迹象再补 CP2/4。若 P1 已说明生产瓶颈几乎完全位于别处，可以推迟这 6 次。

128K 只在生产形状上测 P1 的两个关键 CP，不能从旧 32-head 单层线性外推 64-head 完整栈。headwise 不列入待验证的主要优化方向。

### P3：让 profiler 决定机制；每次只推进一个

| 实测主要问题 | 第一候选 | 最小辨别实验 | 放弃/推进条件 |
|---|---|---|---|
| CSA 无效辅助路径或形状膨胀 | 跳过 dead teacher；修正 local/global shape 使用 | P2A 配对 | 不能保持目标/梯度，或影响太小则停止 |
| GDN 当前算子临时工作集主导峰值 | workspace 分块、避免大中间量物化、融合 | 相同 CP/输入，旧/新核心；查是否影响递推与 backward | 需要实际减峰，不能仅减少 allocator reserved |
| 跨层保存张量主导峰值 | 按 GDN/GQA/FFN 选择 checkpoint 区域或保存策略 | 相同布局，旧/新 checkpoint；核算重算次数 | 能打开更快统一布局或提高固定 G 吞吐才推进 |
| FSDP gather/梯度缓冲主导峰值或等待 | 参数生命周期、prefetch 深度、wrap 粒度 | 同布局、同 checkpoint，改变一个策略 | 同时报告 gather 次数/字节；避免用更大驻留换来隐藏 OOM |
| FFN/输出头主导峰值 | FFN/output 投影分块或局部 TP | 独立组件形状与端到端显存对照 | TP 的梯度和布局代价必须计入，不能从 attention 曲线推断 |
| 不同子模块在可行区确有相反最优布局 | 仅该边界的异构 TP/CP/状态分片 | 相同样本、相同 W 的子图与边界往返 | 先满足收益预算，再做窄原型 |

这一步的研究对象可以是算子、重计算、模型状态并行或边界数据流，不预设 CP 是中心。不同时实现多个候选后再猜收益来自谁。

先在真实 3GDN+1GQA 周期（包含 FFN，具体顺序依真实模型）或目标 CSA/HCA decoder 周期上验证。若 64K 通过，再到 128K 和完整模型。不要把当前 attention-only 模块脚本当成完整 block。

### P4：完整训练验收

从 P1 选两个可行且有竞争力的统一配置，与 P3 的一个候选配对：最多 2 配置 × 2 实现 × 3 独立 launch=12 次；只有一个可行配置则 6 次。采用相同全局样本序列和更新规则，B1/G32 控制组保持不变。

每次先完成状态初始化/JIT，再观察至少 3 个 warmup step；默认记录 5 个稳态完整 step，若仍漂移延长。正式计时不带 profiler，明确数据加载是否纳入，并保持一致。约 52 s/step 的旧数来自不同状态，只能用于粗略安排机时，不能承诺新运行耗时。

报告每次 launch 中位数和跨 launch 范围；p95 只有在足够 step 样本下作为描述指标，不能用 3 次独立运行声称严格统计显著。3 次胜负不一致或收益接近噪声时扩到 5 次，不无限增加重复。

工程推进门槛建议：稳定 step-time 改善至少 5%，或让原本不可行的目标进入可行区并在最终同负载比较中有价值。5% 是项目投入门槛，不是统计学定理；低于它的简单修复仍可保留，但不据此建设复杂 runtime。所有加速都比较各自可用的最佳统一基线，不选较慢基线。

最后才做系统调优：在 G32、W8 与显存预算不变下，让各方案各自选择 B/A（满足 DBA=32）。同时保留 B1 结果区分收益来自机制还是 batch 调整。不同方案不能拥有不对称的 checkpoint/微批调优机会。

## 5. 异构 mesh 方向的进入条件

当前证据尚不足以优先开发通用 mesh planner。需要先量化可兑现上界：

`可用预算 = 最佳统一布局耗时 − 各子图在相同样本与 GPU 预算下的理想执行耗时`。

独立最优值相加只给出乐观估计；若不满足共同的数据依赖、显存与并发资源约束，连可执行计划都不是。不能用 R1 的最小延迟直接相加当作固定资源吞吐。

只有该预算明显大于实测边界通信、同步、复制计算以及额外驻留代价，才做一个边界原型。对每个候选明确：样本/token 在哪张卡、是否复制、参数和梯度由谁拥有、反向映射和累积如何归约。B1 没有第二条样本可以凭重排产生新的独立 DP 计算，不能把 group size 改小就计作并行收益。

下一步可能值得研究的是：核心 attention 与投影/FFN 的不同切分，或模型状态分片与激活布局的独立配置。但必须由 P1/P3 的瓶颈触发。若进入 MoE，再增加真实 EP dispatch/combine、路由偏斜与 eFSDP 的证据；现有 dense-FFN 数据不能证明 EP folding 的收益。仅有“多个 mesh + 自动 reshard + 搜索”不足以区别于既有通用自动并行工作。

QSA 等新变体只在明确论文/代码版本、训练 kernel 和 backward 可用后纳入，每个先取两个端点与一个异常点；不用尚未冻结的产品简称或推理 kernel 替代训练结论。

## 6. 最小新增预算与交付物

| 阶段 | 首轮新增 | 后续条件扩展 |
|---|---|---|
| P1 | 最多 4 个统一布局容量探测；1–2 个完整训练内存 profile | 可行候选进入 P4 |
| P2A | CSA 3 个 profile | 证实分支负担后 18 次 λ0 消融；λ>0 按目标另测 |
| P2B | GDN 3 个 profile | GQA 端点 6 次；中间点/隔离控制按异常追加 |
| P3 | 一个机制的小尺寸正确性与一个目标形状验证 | 通过后扩大到真实 block/128K |
| P4 | 6–12 次完整训练配对 | 仅边际结果扩大重复；最后再调 B/A |

各路径可分时进行，不要求先完成全部 P2 才做 P1。优先获得真实 FSDP baseline 与内存证据；CSA 路径服务于更广模型的机制判断，不能阻塞 GDN/GQA 生产问题。不得并发运行会争用相同 GPU 的正式 benchmark。

每个新 launch 保存：config JSON、命令、代码身份、exit code、正确性状态、样本标识、原始逐 step 时间、全 rank 峰值、总结。必填 W/TP/PP/CP/DP/B/A/G/S、有效 token 数、CP 算法、全部 attention 维度、训练 loss 语义、checkpoint/FSDP 参数组。profile 文件单列，不能混入稳态样本。

最终应产生四份可决策结果：已有证据可复用表；生产统一布局可行/性能前沿；一个被消融证实的时间或显存主因；针对该主因的端到端机制对照。未得到最后一项前，不把新 dataflow 抽象宣布为下一条主路线。


## mbs1_aggregates.csv

```csv
operator,cp,n,median_ms,min_launch_ms,max_launch_ms,peak_alloc_gib,latency_speedup,group_gpu_ms,scaling_efficiency,regime
csa,1,2,552.48,552.37,552.59,27.82,1.0,552.48,1.0,W8_B1_A1_D8divC_replica_concurrent
csa,2,2,639.99,639.63,640.35,27.02,0.863263,1279.98,0.431632,W8_B1_A1_D8divC_replica_concurrent
csa,4,3,536.09,535.83,536.71,13.72,1.030573,2144.36,0.257643,W8_B1_A1_D8divC_replica_concurrent
csa,8,2,479.855,479.37,480.34,7.07,1.151348,3838.84,0.143918,W8_B1_A1_D8divC_replica_concurrent
gdn_chunkwise,1,3,85.346,85.189,85.671,15.66,1.0,85.346,1.0,W8_B1_A1_D8divC_replica_concurrent
gdn_chunkwise,2,3,54.12,54.112,54.23,8.26,1.576977,108.24,0.788489,W8_B1_A1_D8divC_replica_concurrent
gdn_chunkwise,4,3,29.046,29.022,29.097,4.3,2.938305,116.184,0.734576,W8_B1_A1_D8divC_replica_concurrent
gdn_chunkwise,8,3,16.511,16.498,16.511,2.33,5.169039,132.088,0.64613,W8_B1_A1_D8divC_replica_concurrent
hca,1,2,140.595,140.12,141.07,22.76,1.0,140.595,1.0,W8_B1_A1_D8divC_replica_concurrent
hca,2,2,69.055,68.67,69.44,12.11,2.035986,138.11,1.017993,W8_B1_A1_D8divC_replica_concurrent
hca,4,3,36.44,35.27,36.57,6.29,3.85826,145.76,0.964565,W8_B1_A1_D8divC_replica_concurrent
hca,8,2,19.985,19.73,20.24,3.37,7.035026,159.88,0.879378,W8_B1_A1_D8divC_replica_concurrent
```


## unconfirmed_log_metrics.csv

```csv
log,time_ms,status
agent_run/20260904T085544Z_manual_csa_hca_cp_profiles/logs/mbs1_hca_seq65536_cp8_launch3.log,19.51,METRIC_ONLY_EXIT_UNCONFIRMED
```


## legacy_fixed_global_64k.csv

```csv
operator,cp,dp,batch,accumulation,global_batch,time_ms,peak_alloc_gib,dp_grad_sync,n,regime
attention,1,8,1,1,8,224.13035736083984,5.711322784423828,false,1,W8_G8_A1_BequalsC_legacy_headwise
attention,2,4,2,1,8,232.85492401123048,7.215229034423828,false,1,W8_G8_A1_BequalsC_legacy_headwise
attention,4,2,4,1,8,247.35660400390626,7.476824760437012,false,1,W8_G8_A1_BequalsC_legacy_headwise
attention,8,1,8,1,8,265.80360412597656,10.508074760437012,false,1,W8_G8_A1_BequalsC_legacy_headwise
gdn,1,8,1,1,8,83.25075988769531,14.15376901626587,false,1,W8_G8_A1_BequalsC_legacy_headwise
gdn,2,4,2,1,8,121.96714859008789,14.153814792633057,false,1,W8_G8_A1_BequalsC_legacy_headwise
gdn,4,2,4,1,8,132.59780654907226,14.153791904449463,false,1,W8_G8_A1_BequalsC_legacy_headwise
gdn,8,1,8,1,8,140.78637084960937,14.153780460357666,false,1,W8_G8_A1_BequalsC_legacy_headwise
```
