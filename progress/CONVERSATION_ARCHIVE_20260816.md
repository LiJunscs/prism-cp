# 异构 DP–CP 实验阶段归档：从 P0 修复到三 seed 复测

> 归档日期：2026-08-16  
> 对应运行：`20260815T120648Z_b0226ace2_seed_sweep`  
> 主题：混合 Attention/GDN 长序列训练中的层级异构 DP–CP、边界重分片与 TMF 前置证据

## 1. 本阶段解决了什么

本阶段完成了三件关键工作：

1. 修复并验证了 DP–CP transition 的基础实现，包括 direction table、合法性检查和 gather 阶段的 zigzag inverse reshape；
2. 在固定 8 张 H800、固定 global batch 的条件下，重新测量 Attention/GDN 的 CP 性能与显存趋势；
3. 对 2 Attention + 6 GDN stack 的三种异构配置，在三个 layer schedule 下进行进程级复测，确认降低 GDN CP 的收益是否稳定。

当前证据已经足以支持“静态高 CP 会给 GDN 带来可测的性能税，因此层级异构 DP–CP 值得研究”这一动机，但尚不足以证明 Temporal Mesh Folding（TMF）的系统贡献。

## 2. 对话与研究思路演进

### 2.1 初始问题

最初设想是在固定设备数、固定基础模型并行布局的前提下，让长序列 Attention 使用较高 CP，让 GDN 使用较低 CP，并在模块边界同时重映射 DP 与 CP：

```text
DP_i × CP_i = W
```

目标不是用 CP 替代 TP。TP/PP/EP 属于保证模型参数与基础计算图可训练的前置布局；DPCP 是在此基础上针对长序列模块进行的运行期调优。

### 2.2 显存认识修正

固定 `W` 和 global batch 时：

```text
local_batch ∝ CP
local_sequence ∝ 1 / CP
local_tokens = local_batch × local_sequence ≈ constant
```

因此，近似依赖 local token 数的 GDN activation memory 不会随 CP 明显下降。模块 frontier 也支持这一点。Attention 在当前 Flash Attention 实现和测试范围内，CP 增大反而伴随额外 buffer/workspace，不能再笼统主张“提高 CP 必然降低固定资源下的 Attention 峰值显存”。

研究叙事应聚焦于：当 Attention 因长序列计算可行性或更完整模型的显存约束而必须使用高 CP 时，GDN 是否有必要承担相同 CP，以及如何低成本恢复 GDN 的数据并行度。

### 2.3 TMF 的重新定位

相关工作已经覆盖模块独立布局、boundary communicator、动态 CP 和异构流水线。因此，“不同模块使用不同 CP”本身不是充分创新。

TMF 的潜在独特问题是：当 nominal local microbatch 很小、尤其为 1，且目标 DP 增大时，空间内 DP–CP 转换会受到样本整除约束。若保持固定 optimizer-step 语义并要求所有设备执行非冗余工作，就需要从同一 optimizer step 内的时间维取得额外 logical microbatch。

本轮实验使用 `global_batch_size=8`，转换在空间上已经合法，因此它验证的是异构布局收益，而不是 TMF 的 temporal cohorting 机制。

## 3. 代码修复与正确性状态

### 3.1 已修复问题

- 扩展 8-rank 下的 `DIRECTION_2_4` 与 `DIRECTION_4_1`；
- 放宽错误的 `world_size == max_cp` 限制，使 CP4↔CP1 可在 world size 8 下表达；
- 修复 `_GatherSequenceForCPTransition.forward`：接收 tensor 必须先恢复成两个 zigzag sequence block，再写回前后对称位置；
- 让专项测试导入 `heterogeneous_cp` 主副本，避免测试旧的 `benchmarks` 副本。

此前的关键错误是：小测试中 `seq=32, CP=8` 时本地序列长度恰好等于 4，意外掩盖了 `index_copy_` 的错误布局；真实 `seq=65536` 下接收 tensor 第一维为 8192，才暴露 `4 indices != 8192 rows`。

### 3.2 当前正确性证据

- 8-rank P0 测试：`16/16 PASS`；
- 覆盖 CP `2↔8`、`4↔8`，global batch `{8,16}` 的 value、round-trip 和 source-gradient；
- toy mixed stack 的 input gradient、parameter gradient 和一次 optimizer update 通过；
- 修复后 45 次真实 `seq=65536` mixed stack 均完整完成 forward/backward。

### 3.3 仍未关闭的正确性边界

- P0 transition 的逐元素测试仍使用 `seq=32`，必须增加 `seq≥512` 的 case 防止同类回归；
- 真实 Attention/GDN stack 目前只证明可以运行，没有和静态 reference 做逐元素 output、input-gradient、parameter-gradient、optimizer-update parity；
- 尚未进行 BF16 误差基线、长程 loss parity 或收敛验证。

因此，当前正确性状态应写为：**transition toy semantics 已验证，真实尺度执行路径已验证可运行，但真实模块数值等价性仍待完成。**

## 4. 运行环境与可复现性

- 设备：8 × NVIDIA H800，约 81.5 GiB/GPU；
- Python 3.12.0；
- PyTorch 2.8.0+cu128；
- NCCL 2.27.3；
- Transformer Engine 2.14.0；
- 记录 commit：`b0226ace2ef6755b62c27371d5c110346522b8bf`。

重要限制：结果由该 commit 加上三个未提交文件的修复产生：

- `heterogeneous_cp/bench_fixed_mixed.py`；
- `tests/unit_tests/test_heterogeneous_dp_cp.py`。

因此，仅 checkout `b0226ace2` 不能严格复现本轮结果。下一次正式实验前必须把修复形成明确 commit，或至少保存完整 patch 与文件 hash。

## 5. 固定资源模块 frontier

Agent 报告记录了 48 个计划配置，其中 43 个成功、5 个 OOM。当前归档目录没有包含该轮完整的 `module_frontier/merged.csv` 与 `failed.csv`，所以下述内容来自 Agent 报告，不能在本次归档中重新逐行审计。

固定条件为 world size 8、hidden size 4096、Flash Attention、warmup 20、iters 50；模块 benchmark 不包含 DP parameter-gradient synchronization。

主要观察：

- 所有可运行 block 中，CP1 的 fwd+bwd latency 最低；
- GDN 对高 CP 非常敏感：长序列下 CP1 相对 CP8 的吞吐优势约 1.6–1.7×；
- Attention 在长序列下对 CP 更友好：`seq=262144, gbs=8` 时 CP1 相对 CP8 约 1.07×；
- GDN worst-rank peak allocated 基本不随 CP 变化；
- Attention peak allocated 在本测试中随 CP 增加；
- 5 个 OOM 都出现在 `seq=262144, gbs=16`。

这组结果支持“GDN 不应无条件继承 Attention 的高 CP”，但不支持“提高 CP 是固定资源下普遍有效的显存调节器”。

## 6. 三 seed、45 次 mixed/fixed 复测

### 6.1 实验设计

固定条件：

- world size 8；
- global batch size 8；
- sequence length 65536；
- 2 Attention + 6 GDN；
- Flash Attention；
- warmup 20，iters 50；
- 每个 `seed × config` 做 5 次独立进程 launch；
- fixed 与 mixed 在同一次 launch 中共享 layer schedule。

三种 layer schedule：

| Seed | Schedule        | DP–CP transitions |
| ---: | --------------- | ----------------: |
| 1234 | A-G-G-G-G-G-A-G |                 3 |
| 2345 | G-G-G-G-A-A-G-G |                 2 |
| 4321 | G-G-G-A-G-G-A-G |                 4 |

三种配置：

| Config | Attention CP | GDN CP | Fixed baseline CP |
| ------ | -----------: | -----: | ----------------: |
| A      |            4 |      1 |                 4 |
| B      |            8 |      2 |                 8 |
| C      |            8 |      4 |                 8 |

### 6.2 复核后的核心结果

以下范围使用每个 schedule 内 5 次 launch 的 median；“跨 schedule 中位值”是三个 schedule median 的中位数，不是把 15 次 launch 当作 15 个独立结构样本。

| Config     | 各 schedule speedup median 范围 | 跨 schedule 中位值 | 吞吐收益判断         |
| ---------- | ------------------------------: | -----------------: | -------------------- |
| A：CP4→CP1 |                  1.2379–1.2573× |        **1.2481×** | 稳定且较大，约 24.8% |
| B：CP8→CP2 |                  1.0609–1.0733× |        **1.0688×** | 稳定但中等，约 6.9%  |
| C：CP8→CP4 |                  1.0161–1.0272× |        **1.0230×** | 很小，约 2.3%        |

配置排序 `A > B > C` 在三个 schedule 中完全一致。这是本轮最可靠的结论：**只有将 GDN CP 降低足够多，模块效率收益才明显超过边界重分片成本。**

### 6.3 重复性与统计解释

每个固定 schedule 下 5 次 launch 的离散非常小，说明：

- GPU 运行噪声低；
- 同一配置可以稳定复现；
- A、B 的收益不是偶然的单次 timing spike。

但 Agent 给出的 bootstrap 95% CI 是在同一 schedule 的 5 次 launch 上重采样，只描述技术重复的测量不确定性。它不能代表对任意 layer schedule、模型结构或训练任务的泛化不确定性。

真正用于跨 schedule 推断的独立单位只有 3 个 seed。当前不应使用“统计显著”表述，也不应把 45 次 launch 写成 `n=45` 的结构样本。

### 6.4 transition 次数

观察值满足：

```text
2 transitions > 3 transitions > 4 transitions
```

2-transition 与 4-transition schedule 相比：

- Config A 的 speedup 相差约 1.6%；
- Config B 相差约 1.2%；
- Config C 相差约 1.1%。

方向符合“更多 reshard 会侵蚀收益”的直觉，但 transition 次数与 Attention/GDN 的具体位置、连续 block 长度、激活生命周期同时变化，存在完全混杂。当前只能写“与 transition 次数负相关”，不能写“由 transition 次数导致”。

若要建立因果归因，应构造层顺序受控、只改变转换次数的 schedule，或独立测量相同 tensor shape 下的 reshard latency。

### 6.5 显存

- seed 1234：mixed 与 fixed 的 worst-rank peak allocated 持平；
- seed 2345/4321：Config A/C 下降 2 GiB，Config B 下降 3 GiB；
- 45 次运行中没有出现 mixed peak allocated 高于 fixed。

可以支持的表述是：**在已测的三个 schedule 中，异构布局没有增加 worst-rank peak allocated，并在两个 schedule 中降低了 2–3 GiB。**

不能进一步断言“降低 GDN CP 必然节省显存”。显存差异与 schedule、峰值发生位置、临时 buffer 和 activation lifetime 共同相关，而且数值呈精确的 2/3 GiB 台阶，需要后续 memory timeline/decomposition 才能解释机制。

### 6.6 梯度同步口径

需要区分两个 benchmark：

- `bench_dp_cp.py` 的单模块 frontier 明确是 `dp_grad_sync=false`；
- `bench_fixed_mixed.py` 的 stack benchmark 在 backward 后对参数梯度执行 WORLD all-reduce，因此 Task D 不能统一标为 `dp_grad_sync=false`。

Task D 仍不是完整训练，因为没有 optimizer step、数据加载、loss、长期训练与收敛验证，但其 fixed/mixed timing 已包含脚本实现的参数梯度同步开销。

## 7. OOM 边界

- `seq=65536`：三组配置均可稳定运行；
- `seq=81920`、98304、131072：现有 probe 在 fixed 阶段 OOM；
- `seq=73728`：日志只到 fixed benchmark 开始，没有成功、OOM 或明确退出信息，状态应记为 incomplete/unknown。

因此只能说当前 8-layer、gbs=8、80 GiB 配置的已知可用点是 65536，已知 OOM 点从 81920 开始；不能把精确上限写成 64K 或 73K。

长序列下一步需要 activation checkpointing、减少层数、降低 global batch，或采用更接近真实训练的内存优化后再测试。

## 8. 与旧结果的关系

旧 `res_v2.txt` 报告的 A/B/C 收益更高，约为 1.53×、1.164×、1.065–1.096×；本轮统一口径约为 1.248×、1.069×、1.023×。

两轮配置排序一致，但旧数据缺少完整 sequence length、warmup/iters、代码修复状态和独立重复信息，并可能使用错误 direction table。旧结果只能作为历史动机，不应与新数据合并、取平均或作为论文主结果。后续定量结论以修复后的本轮数据为准。

## 9. 当前可支持的论文级表述边界

### 可以作为阶段性结果陈述

> 在 8×H800、sequence length 65536、global batch 8 的 2-Attention/6-GDN stack 上，三个选定 layer schedule 均显示：将 GDN 从 Attention 所需的 CP4 降至 CP1，可获得约 1.238–1.257× 的 fwd+bwd speedup；从 CP8 降至 CP2 获得约 1.061–1.073×；CP8 降至 CP4 仅获得约 1.016–1.027×。所有已测 mixed 配置的 worst-rank peak allocated 均未超过对应 fixed baseline。

### 当前不能声称

- 不能声称异构 DP–CP 是新的系统抽象；
- 不能声称 transition 次数是性能差异的唯一原因；
- 不能声称 mixed 必然节省显存；
- 不能声称已经验证 microbatch=1 或 TMF；
- 不能声称完整训练吞吐、优化器语义等价或收敛一致；
- 不能使用旧 `res_v2.txt` 的较大 speedup 作为当前主结果。

## 10. 对研究路线的影响

### 已通过的 Go/No-Go

静态异构布局的性能动机已经通过：

- CP4→CP1 的收益足够大；
- CP8→CP2 的收益在多个 schedule 下仍高于 5%；
- mixed peak allocated 没有恶化；
- 配置排序跨三个 schedule 稳定。

### 尚未通过的 Go/No-Go

TMF 本身尚未验证，因为当前 global batch 足以让空间 transition 合法，没有触发 microbatch divisibility barrier，也没有 temporal bundle、checkpoint 或 pipeline scheduling。

因此论文主线不能停留在本轮 static mixed benchmark。它只能作为 TMF 的动机图和 cost-model 输入。

## 11. 下一步工作优先级

### P0：先冻结可复现版本

1. 将产生结果的三个 dirty 修复形成明确 commit，或保存完整 patch；
2. 保存 benchmark 文件 hash、启动命令和测试路径；
3. 将 `heterogeneous_cp` 固定为实验主副本，避免再次与 `benchmarks` 混用。

### P1：补齐真实尺度正确性

1. 将 transition value/gradient 测试扩展到 `seq≥512`；
2. 加入 CP1↔CP4、CP2↔CP4 等实际使用的转换；
3. 用真实 Attention→GDN→Attention 比较 output、input gradient、parameter gradient 和一次 optimizer update；
4. 先 FP32，再 BF16，并记录 homogeneous baseline 的自然误差。

### P2：解释收益而不是继续堆 seed

1. 单独测量 CP8↔2、CP8↔4、CP4↔1 的 forward/backward/round-trip reshard latency；
2. 构造只改变 transition 次数、不改变模块数和主要位置关系的受控 schedule；
3. 分解 Attention、GDN、reshard、gradient all-reduce 和 temporary buffer；
4. 记录 per-rank memory timeline，解释 2–3 GiB 台阶差异。

### P3：进入 TMF 最小原型

在上述正确性和成本测量关闭后，构造 nominal local microbatch=1、空间上不合法的转换：

1. 先实现无 PP 的 packed temporal cohort；
2. 严格限制 bundle 位于同一 optimizer step；
3. 与相同 logical samples 的 static-high-CP gradient accumulation 对拍；
4. 比较保留完整 graph、boundary checkpoint 和 recompute；
5. 报告每 logical microbatch 的吞吐、bundle wait、peak allocated 和 correctness。

只有 TMF 在 memory-bounded 场景中相对最佳合法静态方案仍保持明显收益，才值得继续进入 bundle-aware PP 和 planner。

## 12. 数据与证据索引

- 本轮主报告：`REPORT.md`；
- 三 seed 综合报告：`TASK_D_3SEED_COMBINED_REPORT.md`；
- 45 行合并原始数据：`mixed_repeatability/parsed_3seed_merged.csv`；
- seed 2345/4321 汇总：`mixed_repeatability/summary.csv`；
- 30 个新 launch 日志：`mixed_repeatability/logs/`；
- seed 1234 运行：`../20260815T075123Z_b0226ace2_fix1/`；
- P0 日志：`../20260815T075123Z_b0226ace2_fix1/p0_transition/logs/rerun_all.out`；
- OOM 探针：`../20260815T075123Z_b0226ace2_fix1/mixed_repeatability/probe/`；
- 环境：`metadata/`；
- 所有启动命令：`commands.tsv`。

## 13. 阶段结论

本轮最重要的结果不是某一个最大 speedup，而是三个更稳定的判断：

1. **GDN 的高 CP 性能税确实存在，且降低幅度越大，端到端收益越明显；**
2. **边界重分片没有吞掉 CP4→CP1 和 CP8→CP2 的全部收益，但 CP8→CP4 的收益过小，不值得单独构成系统设计；**
3. **异构 DP–CP 的动机已经成立，TMF 的必要性与实现价值仍需在 microbatch=1、整除不可行和 checkpoint/bundle 约束下单独证明。**

当前研究状态可概括为：**P0 基础实现已基本关闭，静态异构动机得到可复现支持，下一阶段应从“继续重复 timing”转向“真实模块数值等价 + reshard 成本分解 + microbatch=1 TMF 原型”。**
