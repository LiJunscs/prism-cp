# 知识库主题：TMF Profiling 数据审计、1F1B 集成与性能模型路线图

> 日期：2026-08-31
> 结论：现有 profiling **可用于假设验证、实验选点和模型先验**，但**不可直接作为当前实现或 1F1B 的最终性能证据**。进入正式实验前需要一次小规模同版本桥接复测；不需要无差别重跑全部历史矩阵。

## 1. 决策

当前可以继续 1F1B 集成和性能模型开发，不必等待全量重采集。现有数据已经稳定支持以下判断：

1. 在固定设备数与 global batch 下，GDN 提高 CP 会产生稳定且显著的延迟税；
2. 对 2 Attention + 6 GDN 的目标 stack，降低 GDN CP 的收益大于旧版 transition 与 temporal bookkeeping 成本；
3. temporal mixed 路径的额外显存主要出现在 backward 保存的 activation graph，而不是 pack 或 transition 临时张量；
4. bundle 是后续主路径，repeat 仅保留为合法性与容错对照，并必须报告 unique/logical tokens/s；
5. 性能模型必须建成事件图模型，显式包含 pipeline wait、bundle rendezvous、P2P、transition、重算和峰值存活集合，不能把历史单层均值直接相加。

但现有绝对 latency 不能直接进入最终图表或模型训练集，原因有四个：

- 2026-08-22 数据来自 commit `4cf9af8...` 及 dirty patch；当前 `bench_fixed_mixed.py` 虽与 2026-08-30 审查版同 hash，但 `utils.py` 已继续修改，且 correctness 修复改变了 repeat/gradient 路径；
- 2026-08-16 CP frontier 来自 commit `36b8f317...`，每个点仅一个独立 launch，且 `dp_grad_sync=false`；
- 旧 temporal profiler 只有阶段级统计，backward 被合并成一个大桶，无法分离 layer backward、transition backward、PP P2P 和等待；
- 当前 `bench_1f1b_attention.py` 只执行一次 optimizer-step-shaped schedule，只检查事件顺序和 finite，不含 GPipe 数值对照、重复 launch、逐 rank timeline 或 TMF boundary。

因此采用两级策略：历史数据作为 **prior / regression band**；当前 commit 的桥接数据和后续 1F1B 数据作为 **fit / validation set**。

## 2. 已审计数据清单

| 数据集 | 覆盖 | 重复性 | 可用结论 | 不可用于 |
| --- | --- | --- | --- | --- |
| `20260816T070154Z_36b8f3172_post_refactor/dp_cp_frontier/merged.csv` | Attention/GDN × seq 16K/64K/256K × CP1/2/4/8，共 24 点 | 每点 1 launch；5 warmup + 10 iters；无 DP grad sync | CP 趋势、显存可行域、实验选点 | 置信区间、当前代码绝对 latency、1F1B stage latency |
| `20260822T100108Z_tmf_validation_profile/performance/` | A/B × 32K/65K × bundle/repeat × fixed/mixed | 2,800 个逐迭代样本；32K 每单元 2 launch，65K 每单元 5 launch；每 launch 20 warmup + 50 iters | 历史回归带、收益量级、噪声水平 | 当前代码最终结果、修复后 repeat 结论、PP/1F1B 结论 |
| `20260822T100108Z_tmf_validation_profile/profiling/` | A/B × bundle/repeat × fixed/mixed，65K | 每单元 5 warmup + 3 measured；8 个阶段汇总 | 瓶颈归因、显存阶段定位 | layer/transition backward 拟合、worst-rank 模块时间、overlap 模型 |
| `20260830T081917Z_tmf_repeat_fixed/` | correctness only | T2 后按 gate 停止 | 版本与失败根因历史 | 任何性能结论 |
| 2026-08-31 correctness 闭环 | 当前 benchmark stack 的 output/input-grad/parameter-grad | 多路径 BF16 范数门控通过 | 允许开放性能阶段 | optimizer update、长程训练、PP、checkpoint |

### 2.1 数据质量量化

2026-08-22 `performance_raw.csv` 实际包含 56 个 launch 单元和 2,800 个逐迭代样本，没有空 latency。65K 的主要 mixed 单元单 launch CV 为约 `0.16%–0.35%`，B fixed 为约 `0.09%–0.12%`；A fixed 的少数 launch 有长尾，CV 约 `0.7%–2.6%`。跨 launch median 的极差小于约 2 ms，说明旧数据的收益方向稳定。

历史 65K 结果为：

| 配置 | fixed | mixed | 历史 speedup |
| --- | ---: | ---: | ---: |
| A：Attention CP4 / GDN CP1 | 约 1369.7 ms | 约 1065.3 ms | 1.286× |
| B：Attention CP8 / GDN CP2 | 约 1599.7 ms | 约 1323.9 ms | 1.208× |

阶段分解显示 mixed 的 forward transition 约 15–16 ms，而 backward 占总时间约 67%–69%。mixed backward 峰值 allocated 比 fixed 高约 4.1–4.8 GiB，峰值发生在 backward；pack/transition forward 并非峰值来源。

2026-08-16 frontier 则给出稳定的结构性趋势：

- 64K 时 Attention CP8 相对 CP1 的 `fwd+bwd` 延迟约 1.19×；256K 时约 1.07×；
- GDN CP8 相对 CP1 在 16K/64K/256K 分别约 1.71×/1.69×/1.65×；
- GDN 的峰值 allocated 在同一序列长度下基本不随 CP 下降，说明高 CP 对该实现主要增加通信/调度成本，未带来对应显存收益；
- 这些点使用 `global_batch=8` 且 local batch 随 CP 改变，不能直接替代 1F1B 中 `micro_batch=1` 的 stage profile。

## 3. 数据使用规则

### 3.1 可以立即复用

- 选择 A（CP4→1）和 B（CP8→2）作为主配置；
- 优先使用 32K/64K 做开发，128K 做扩展，256K 只在显存允许时做边界点；
- 用 1.2×–1.3× 作为历史收益量级，而不是当前性能承诺；
- 用 15–16 ms 的旧 transition forward 作为 profiler 合理性检查；
- 用 backward activation graph 作为 checkpoint/recompute 的首要优化对象；
- 用历史 CV 设定独立 launch 数和异常值审计方法。

### 3.2 只能作为先验

- 2026-08-16 单层 CP 曲线；
- 2026-08-22 的 fixed/mixed 绝对 latency；
- rank 0 本地 CUDA-event 的阶段时间；
- 旧 repeat 的 executed/unique throughput。

这些数据可初始化模型系数、确定采样范围和发现回归，但模型最终拟合必须带 `code_hash`、`hardware_topology_id` 和 `measurement_semantics`，不得与当前数据无标签合并。

### 3.3 必须重新采集

1. 当前 commit 上 bundle/repeat 的端到端桥接数据；
2. 当前实现的 transition forward/backward 独立计时；
3. `micro_batch=1` 下 Attention/GDN 的 forward/backward primitive profile；
4. fixed-CP 1F1B 的 GPipe 数值对照、逐 rank timeline、P2P 与 wait；
5. bundle-aware 1F1B 的 cohort wait、activation lifetime 和峰值显存；
6. 至少一个未参与拟合的 PP/CP/M/sequence 配置作为模型 holdout。

## 4. 最小桥接复测

桥接复测用于判断历史数据能否继续作为 regression band，不是全量重跑。

### 4.1 固定条件

- 8×H800 单节点，记录完整拓扑、GPU clocks/power state、driver、CUDA、PyTorch、NCCL、TE/Megatron 版本；
- 冻结 Git commit 与五个文件 hash：`utils.py`、`bench_fixed_mixed.py`、`test_correctness.py`、`bench_1f1b_attention.py`、`model_cp_transition.py`；
- BF16、Flash attention、hidden 4096、2 Attention + 6 GDN、GBS 8、MBS 1；
- 每 launch 20 warmup + 50 measured iterations；5 个独立 launch，fixed/mixed 使用 ABBA 或交错顺序；
- 保存逐迭代 latency，报告 median、p5、p95、CV、worst-rank peak allocated/reserved 与 exit code。

### 4.2 最小矩阵

| 优先级 | 配置 | 序列 | 策略 | 用途 |
| --- | --- | --- | --- | --- |
| 必跑 | A：CP4→1 | 65K | bundle fixed/mixed | 与最强历史点桥接 |
| 必跑 | B：CP8→2 | 65K | bundle fixed/mixed | 与第二主配置桥接 |
| 必跑 | A/B | 65K | repeat fixed/mixed | 刷新修复后 repeat 语义与 unique throughput |
| 建议 | A/B | 32K | bundle fixed/mixed，各 2 launch | 检查序列缩放趋势 |

若当前与历史的跨 launch median 相差不超过 5%、排序一致、peak allocated 相差不超过 1 GiB，可保留历史数据作为回归带。即使桥接通过，最终模型仍只用当前版本数据拟合。若任一主配置差异超过 5%，则补跑 32K/65K/128K × A/B × bundle/repeat × fixed/mixed 的全矩阵。

## 5. 1F1B 集成与验证路线

### P0：把 v0 变成可验证基线

当前 `bench_1f1b_attention.py` 已有 non-interleaved warmup/steady/cooldown 调度和固定 CP 的 stage P2P，但仍缺少数值 reference。先完成：

1. 把 stage 构建、输入生成和 schedule runner 拆成可复用函数；
2. 加入同参数、同输入的 GPipe/reference runner；
3. 保存 canonical output、stage-0 input gradient 和各 stage reduced parameter gradient；
4. 使用与当前 correctness 一致的 BF16 gate 比较 GPipe 与 1F1B；
5. 在 schedule trace 中记录 `(rank, stage, microbatch_id, F/B, begin/end)`；
6. 增加 warmup steps、measured steps、独立 launch id 和机器可解析 CSV/JSONL；
7. 把 grad sync、P2P 和 step wall time分别计时，不把单次 finite 检查当作 correctness。

固定 CP 验证矩阵：

| 目的 | PP×CP | M | seq | 验收 |
| --- | --- | --- | --- | --- |
| 无流水对照 | 1×8 | 8 | 4K | GPipe/1F1B 完全同调度语义 |
| 主闭环 | 2×4 | 4、8、16 | 4K | 无死锁、事件精确、数值 gate PASS |
| 深 warmup/cooldown | 4×2 | 4、8、16 | 4K | 无死锁、事件精确、数值 gate PASS |
| 性能 | 2×4、4×2 | 8、16 | 32K、64K | 5 launch，逐 rank timeline 完整 |

### P1：建立 stage-local mesh 与 boundary API

TMF 不能直接复用以 WORLD rank 编号为前提的 CP cache。每个 PP stage 需要显式的：

```text
StageMesh(stage_rank_set, high_cp, low_cp, dp_replicas)
BoundaryRecord(pipeline_microbatch_id,
               optimizer_step_id,
               temporal_bundle_id,
               logical_sample_id,
               source_layout,
               destination_layout,
               replication_factor)
```

组创建必须在所有 rank 上保持相同全局顺序，但 transition 的 rank 映射只发生在 stage rank set 内。先做 identity boundary 和 high-CP→high-CP no-op，以验证 metadata 与反向队列，再接入 CP4↔1 和 CP8↔2。

### P2：bundle-aware 1F1B

对 CP4→1，低 CP block 需要 4 个独立 logical microbatch；对 CP8→2 同样需要 cohort width 4。scheduler 必须：

1. 在同一 optimizer step 内收集 cohort，不跨 step；
2. 以 microbatch identity pack，完成 stage-local remesh；
3. 执行低 CP block，再按 identity inverse-remesh；
4. 将每个 logical output 放回原 pipeline activation queue；
5. backward 按同一 `BoundaryRecord` 逆序执行，并保持 loss normalization 只依赖 logical samples；
6. 显式记录 `cohort_wait_ms`、ready queue depth、activation live interval 和 boundary buffer；
7. 当 `M` 不能整除 cohort width 时，在启动前失败，不允许跨 optimizer step 补齐。

首轮只允许一个 stage 内存在：

```text
Attention(high CP) -> bundle boundary -> GDN(low CP) -> inverse boundary
```

通过后再扩展多个重复 A↔GDN 边界和不同 stage placement。repeat 保留为同接口下的 spatial-replication control，不作为默认性能方案。

### P3：TMF 1F1B correctness

每个配置都执行以下三角对照：

1. 同计划 GPipe vs 1F1B，验证 pipeline scheduler；
2. static-high-CP vs bundle-TMF，验证布局与 cohort；
3. 两次独立 launch，验证事件和数值稳定性。

比较 canonical output、stage-0 logical input gradient、全部 stage parameter gradient、一次 optimizer update，并审计：

- bundle identity 不丢失、不重复；
- forward/backward 使用同一 layout record；
- 每个 logical sample 恰好参与一次 loss；
- cohort 不跨 optimizer step；
- 所有通信组调用顺序一致；
- invalid `M/cohort` 在 launch 前明确失败。

### P4：TMF 1F1B performance

基线至少包括：

- static-high-CP 1F1B；
- bundle-TMF 1F1B；
- repeat-TMF 1F1B control；
- 合法时的 static-low-CP oracle；
- GPipe 只作调度消融，不作为主要吞吐基线。

报告 logical tokens/s、worst-rank step median/p95、pipeline bubble、stage idle、cohort wait、PP P2P、transition F/B、layer F/B、grad sync、peak allocated/reserved、activation live-set 和 recompute 时间。

## 6. 性能模型

### 6.1 模型结构

采用离散事件图而不是总和模型。节点至少包括：

```text
Attention_F/B(S, H, CP, local_batch)
GDN_F/B(S, H, CP, local_batch)
TMF_pack/unpack
TMF_transition_F/B(src_cp, dst_cp, bytes, topology)
PP_send/recv_F/B(bytes, link_class)
grad_sync(bytes, group_size)
bundle_rendezvous(bundle_width, arrival_times)
checkpoint_recompute(block_spec)
```

边表示数据依赖、同 stream 串行约束、通信组顺序和可重叠关系。模拟器输出 optimizer-step makespan、逐 stage idle/bubble、关键路径和 activation live-set 峰值。

transition 的初始形式保留闭式 payload 和消息数：

```text
T_transition = launch + alpha * messages
             + bytes / effective_bandwidth
             + pack_unpack
             + topology/contention correction
```

但 `alpha`、有效带宽和 correction 必须由当前实现的 forward/backward profile 分别拟合。单节点 NVSwitch 与多节点链路使用不同参数。

### 6.2 必采 primitive 数据

| 类别 | 最小覆盖 | 重复 |
| --- | --- | --- |
| Attention/GDN F/B | seq 16K/32K/64K/128K；CP1/2/4/8；logical MBS1 | 3 launch × 50 iters |
| Transition F/B | 4↔1、8↔2、8↔4；对应上述 payload | 5 launch × 100 iters |
| PP P2P | PP2×CP4、PP4×CP2 的 activation/gradient bytes | 5 launch × 100 round trips |
| Grad sync | 各 stage 参数字节与 CP group size | 3 launch × 50 iters |
| Boundary pair | A→boundary→GDN 以及反向完整 pair | 5 launch × 50 iters |
| Memory | 每类节点后的 current/max allocated/reserved 与 live tensor bytes | 每个性能单元 |

所有 CUDA-event 时间至少保存 rank、stage、microbatch、stream、event type、begin/end；汇总时使用 worst-rank critical path，同时保留原始 per-rank 数据。

### 6.3 拟合与验收

- 按完整配置划分 fit/holdout，不能随机拆同一 launch 的 iteration；
- 历史数据只作 prior，不进入当前版本 loss；
- primitive latency holdout MAPE 目标 ≤5%；
- fixed-CP 1F1B step-time MAPE 首轮 ≤8%，收敛目标 ≤5%；
- TMF 1F1B step-time MAPE ≤8%，候选计划排序准确率 ≥90%；
- peak allocated 预测误差 ≤5%；
- planner 选择的计划在小搜索空间中距实测 oracle ≤5%；
- 任一模型必须同时报告误差分布、最差配置和 extrapolation 标记。

## 7. 执行顺序与 Go/No-Go

```text
R0 当前版本冻结与桥接复测
 -> R1 fixed-CP 1F1B GPipe parity
 -> R2 fixed-CP 1F1B timeline/profile
 -> R3 stage-local identity boundary
 -> R4 bundle-aware 1F1B correctness
 -> R5 bundle-aware 1F1B performance
 -> R6 primitive + boundary-pair profile
 -> R7 事件模拟器拟合与 holdout
 -> R8 checkpoint/recompute 与 planner
```

进入 R3 的 gate：R0 当前 bundle 仍有正收益，且 R1 全部数值与 schedule gate 通过。进入 R5 的 gate：R4 的 output/input-gradient/parameter-gradient/optimizer update 全部通过。进入 planner 的 gate：模型在未参与拟合的 fixed 与 TMF 配置上同时满足 step-time 与排序阈值。

若桥接复测显示 A/B 的 bundle 收益均低于 1.05×，先做当前版本的 layer/transition 分解；只有 GDN low-CP primitive 收益本身消失时才触发方向性 No-Go。若收益仍存在但 1F1B 中被 cohort wait 或 activation memory 吞噬，则研究重点转向 checkpoint、stage placement 和 bundle-aware schedule。

## 8. 当前结论的表述边界

可以表述为：

> 历史单节点 8×H800 profiling 稳定显示异构 CP 在目标 2A+6GDN stack 上具有约 1.2×–1.3× 的潜在收益，并将主要额外显存定位到 backward activation graph。该数据足以支持 1F1B 集成和事件性能模型开发，但由于代码版本与测量语义已变化，最终性能结论需要当前版本桥接复测和 1F1B 专用 profiling。

当前不得表述为：

- 修复后的 bundle/repeat 已复现历史 speedup；
- TMF 在 1F1B 中已经加速；
- 历史阶段时间是 worst-rank 或可直接相加的模型 ground truth；
- 完整训练、optimizer trajectory、checkpoint/recompute 或多节点性能已经验证。
