# 知识库主题：TMF 1F1B 集成、验证与性能实验 Roadmap

> 日期：2026-08-31
> 状态：benchmark correctness 已在当前 BF16 门控内通过，下一阶段为同版本性能桥接和 fixed-CP 1F1B 验证。
> Agent 计划：`REMOTE_AGENT_TMF_1F1B_PLAN_20260831.md`

## 1. 当前结论

现有 profiling 可以用于实验选点和历史回归，但不能作为当前代码或 1F1B 的最终性能结果：

- 2026-08-22 数据有 2,800 个逐迭代样本，历史 Config A/B 在 65K 上约有 1.286×/1.208× 加速，方向和噪声水平可信；
- 历史数据来自旧 commit，repeat、gradient 和 correctness 路径后来有修改；
- 旧 profiler 不包含 1F1B、逐 rank timeline、cohort wait 和独立 transition backward。

因此不全量重跑历史矩阵。先做小规模桥接复测；历史数据只作 regression band，当前版本数据才进入后续结论。

## 2. 总体路线

| 阶段 | 工作 | 退出条件 |
| --- | --- | --- |
| R0 | 当前版本桥接复测 | A/B bundle、repeat 均可运行，数据完整 |
| R1 | fixed-CP GPipe vs 1F1B correctness | schedule、output、input/parameter gradient 全部通过 |
| R2 | fixed-CP 1F1B profiling | 逐 rank timeline、P2P、wait、显存数据完整 |
| R3 | stage-local TMF boundary | identity boundary 与固定 CP 基线一致 |
| R4 | bundle-aware 1F1B correctness | cohort、反向、一次 optimizer update 全部通过 |
| R5 | TMF 1F1B performance | static-high、bundle、repeat 对照完成 |
| R6 | 性能模型与 planner | 等 R2/R5 数据完成后再展开设计和拟合 |

执行顺序：

```text
R0 -> R1 -> R2 -> R3 -> R4 -> R5 -> R6
```

R0 与 R1 的代码准备可以并行；任何 correctness gate 失败都停止对应的后续性能任务。

## 3. R0：最小桥接复测

固定参数：8×H800、BF16、Flash attention、hidden 4096、2 Attention + 6 GDN、GBS 8、MBS 1。

必跑：

| 配置 | 序列 | 策略 | 重复 |
| --- | --- | --- | --- |
| A：Attention CP4 / GDN CP1 | 65K | bundle、repeat；fixed/mixed | 5 launch × 50 iterations |
| B：Attention CP8 / GDN CP2 | 65K | bundle、repeat；fixed/mixed | 5 launch × 50 iterations |

建议补充 A/B、32K、bundle fixed/mixed，各 2 launch。每个 launch 使用 20 warmup，保存逐迭代 latency、logical tokens/s 和 worst-rank peak allocated/reserved。

若当前与历史 median 差异不超过 5%、排序一致且 peak allocated 差异不超过 1 GiB，历史结果继续作为回归带。否则再补 32K/65K/128K 全矩阵。

## 4. R1–R2：fixed-CP 1F1B

先为 `bench_1f1b_attention.py` 增加：

- 同参数、同输入的 GPipe reference；
- canonical output、stage-0 input gradient、各 stage parameter gradient capture；
- warmup/measured steps、机器可解析输出和逐 rank schedule trace；
- layer F/B、PP P2P、grad sync、idle wait 与峰值显存计时。

Correctness 矩阵：

| PP×CP | Microbatches | Seq |
| --- | --- | --- |
| 1×8 | 8 | 4K |
| 2×4 | 4、8、16 | 4K |
| 4×2 | 4、8、16 | 4K |

性能矩阵：PP2×CP4、PP4×CP2；M=8/16；seq=32K/64K；5 个独立 launch。

## 5. R3–R5：bundle-aware 1F1B

每个 PP stage 使用独立的 stage-local mesh。boundary 至少维护：

```text
optimizer_step_id
pipeline_microbatch_id
temporal_bundle_id
logical_sample_id
source/destination layout
```

集成顺序：

1. high-CP→high-CP identity boundary；
2. 单 stage 的 `Attention(high CP) -> bundle -> GDN(low CP) -> inverse bundle`；
3. GPipe vs 1F1B、static-high vs bundle-TMF 两组数值对照；
4. 通过后采集 static-high、bundle-TMF、repeat-TMF control 的性能。

首轮配置：

- PP2×CP4：CP4→1，cohort=4，M=8/16；
- PP4×CP2：CP2→1，cohort=2，M=8/16；
- correctness 用 4K，performance 用 32K/64K。

cohort 不能跨 optimizer step；`M` 不能整除 cohort width 时必须在执行前失败。

## 6. R6：性能模型（暂不展开）

待 R2/R5 数据完成后，再单独设计和实现性能模型。当前只冻结输入要求：

- 逐 rank、stage、microbatch 的 layer F/B、P2P、transition、wait 和 grad sync 时间；
- bundle arrival/cohort wait；
- optimizer-step worst-rank latency；
- peak allocated/reserved 与 activation live-set；
- commit、拓扑和测量语义标签。

后续模型应预测 1F1B step time、bubble、cohort wait 和峰值显存，并在未参与拟合的 PP/CP/M/sequence 配置上验证。具体公式、拟合方法和阈值在 agent 数据回收后另建文档。

## 7. Go/No-Go

- R0 当前 bundle 仍有正收益且 R1 全部通过，进入 TMF boundary 集成；
- R4 correctness 全部通过，才允许执行 R5；
- 若 A/B bundle 均低于 1.05×，先检查 GDN low-CP primitive 收益、transition 和 cohort wait，不直接否定方向；
- 若收益被 activation memory 或 cohort wait 吞噬，优先研究 checkpoint/recompute 与 stage placement；
- 当前结论不覆盖长程训练、checkpoint、多节点或最终 planner。
