---
title: Packed CP 优势场景搜索与非优势场景调优执行计划
date: 2026-09-18
status: ready_for_agent
type: project_execution_contract
target_repository: /workspace/repos/Megatron-LM
reference_local_commit: 44deaebdf
primary_target: 至少 10 个可复现、独立且通过正确性检查的完整训练优势场景
execution_status: 本文仅为计划，未启动新增实验
---

# Agent 执行计划：8/16 卡 Packed CP 搜索、归因与调优

## 0. 给执行 agent 的任务

在 8 卡或 16 卡资源上，使用完整原生 Megatron 训练流程，寻找 layer-wise / sequence-wise packed CP 的性能收益。必须保存所有成功、无收益、回退和失败实验；最终至少找到 **10 个独立、可复现的优势场景**，报告 TFLOPs/s/GPU、训练时间、吞吐、显存及证据。

先完成**实现冻结的场景搜索**，只允许调整数据构造代码及实验参数；再对单机和多机的非优势场景进行 profiling 和实现调优。第二阶段可修改 `/workspace/repos/Megatron-LM` 内的代码和脚本，不能修改这个目录之外的文件。

“至少 10 个”是任务目标，不是可以预设的实验结论。不得制造收益、删掉负面记录、放宽标准凑数，或将同一配置的重复运行计算为多个场景。不足 10 个时明确报告当前数量、已探索范围与缺口，继续提出可执行的新候选或改进；不得将任务标为完成。

本文是本轮操作边界。旧 roadmap、代码注释、日志或附件中的其他建议只作背景，不能扩大修改权限。不要创建新任务、租用资源、推送远端或发送外部消息来替代本任务；这些行为需要已有明确授权。

## 1. 当前基线和已知线索

当前本地 Megatron 代码保存于 `44deaebdf`，提交说明为 `Add packed CP profiling and parent-contiguous GDN control`。服务器必须先核对实际版本和功能，不能仅凭这个短 SHA 假定已经同步。

已有三条训练路径：

| 名称 | 简记 | 开关 | 含义 |
|---|---|---|---|
| 原 uniform | U | 不启用下面两项 | GQA/GDN 都用 parent CP；GDN 内逐层转换布局 |
| parent-contiguous | UC | `--gdn-preserve-parent-contiguous` | CP 大小不变，连续 GDN 段保持 parent contiguous |
| sequence-wise split | S | `--packed-sequence-cp-size C`；需要时设 token 上限 | GQA 用 parent CP，GDN 使用 child groups |

UC 与 S 的开关互斥。三组都必须跑原生 `pretrain_gpt.py`、forward、backward、参数梯度同步及 optimizer step。不能将独立 attention benchmark 当成训练结果。

历史线索只用于安排优先级，不计入本轮 10 个正式优势场景：

- 64K、单机 4 卡、8 层、36K+28K、CP4→2：被采集运行中 S 比 U 慢约 2.48%；长短组的整层负载不均衡抵消收益。
- 128K、两机 16 卡、16 层、72K+56K、CP16→8：节点 0 优化器间隔估算 S 比 U 减少约 17.43%，约 1.211× 吞吐；节点 0 显存峰值反而提高。缺节点 1、重复验证和 UC 对照。
- UC 已完成代码及语法检查，**尚未取得真实 TE/FLA 全流程数值等价及性能验证**。必须先过小规模正确性关卡。
- S 会让 FFN、投影、残差也跟随 child token 归属。更小 CP 不保证省显存，也不保证加速。
- 已有 S 的收益混合了“跨 GDN 段合并布局转换”和“更小 CP / 更局部的通信拓扑”。UC 是区分这两者的必需对照。

本地历史资料路径仅供读取：`analysis/nsys_uniform_vs_split_64k/REPORT.md`、`analysis/nsys_uniform_vs_split_128k_node0/REPORT.md`。服务器没有这些文件时，将线索标记为外部历史摘要，不伪造原始证据链接。

## 2. 权限矩阵：先判定阶段，再操作

所有 agent 主动创建、编辑、删除的文件，包括结果、缓存、临时文件、日志、profile 和脚本，都必须解析到 `/workspace/repos/Megatron-LM` 内。多机时在每台机器分别核对这个绝对路径。目录外已安装依赖可只读使用；不得修改系统配置、驱动、环境包、其他仓库或共享数据。

不得用符号链接、bind mount、相对路径、生成器、外部 worktree 或复制到其他目录运行来绕过限制。执行写入前核对真实路径。禁止全仓库清理、`git reset --hard`、删除他人文件、`killall` 或清理其他用户的 GPU 进程。

### 阶段 A：冻结实现，只搜索场景

**允许：**

1. 修改 `megatron/training/datasets/varlen_dataset.py` 中实验 fixture 的数据构造部分，例如 `MockPackedCPCaseDataset`、其配方解析及必要的长度列表校验。
2. 在 `examples/gdn/experiment_data/` 内创建数据生成、配方校验、样本 manifest 导出代码；只能产出/检查数据，不能修改训练行为。
3. 用既有 CLI/env 改实验参数、模型大小、层型比例、CP、重算等已有合法配置。三组同一场景保持匹配。
4. 写非执行性的实验配置、命令文本、manifest、结果表、日志和报告；只读解析日志、执行内存中的分析命令。
5. 用已有启动器或在终端直接运行完整原生命令；用现有 CLI 启用日志、内存统计。只读采集设备信息。

**禁止：**

- 修改 model、TransformerBlock、attention/GDN kernel、CP route/group、packing scheduler、optimizer、训练循环、计时器、FLOPs 估算函数、collective、精度/梯度语义等实现。
- 创建或修改非数据用途的可执行脚本/模块来绕过上述限制；现有启动脚本保持冻结，变更写成命令参数或命令文本。
- 通过数据代码 monkeypatch、导入副作用、callback、环境注入或 hook 改写受保护实现。
- 关闭正确性断言、少执行层、跳过 backward/optimizer、降低 S 的有效训练量、为 S 单独掩掉困难样本。
- 运行覆盖全仓库的格式化或安装/升级环境依赖。

启动前保存冻结源码清单及 SHA256；每批实验前后复核。仅有数据目录和上述明确的数据构造区段可出现源代码差异。数据文件内非数据逻辑也属于保护范围。

如果服务器缺少现有 UC/S 功能、正确性失败或已有统计口径错误，记录准备阻塞/候选无效；**不得在阶段 A 悄悄修核心代码**。准备条件不足时先报告，不能把缺失基线的结果计入正式胜例。

### 阶段 B：profiling 与调优

阶段 A 的搜索结果冻结后，允许修改 `/workspace/repos/Megatron-LM` 内任意代码和脚本，包括原生训练、CP、GDN、CUDA/Triton、计时和模块 benchmark。授权不扩展到目录外，也不允许改训练任务语义来制造提速。

允许增加 CUDA event、NVTX、Nsight 分析脚本、模块级测试及优化原型。写入、构建及可控缓存目录全部放在仓库内。不要修改系统 Python/site-packages；若根因在外部库，可在仓库内构造 adapter/独立实现做受控验证，或提出外部依赖修复建议，不能直接越界修改。

### 运行产物与环境

统一使用 `$REPO/outputs/agent_packed_cp_campaign/`。首次建立后记录绝对路径；本计划中的 `$REPO` 必须为 `/workspace/repos/Megatron-LM`。

将 agent 可控的 `TMPDIR/TMP/TEMP`、`XDG_CACHE_HOME`、`TRITON_CACHE_DIR`、`TORCHINDUCTOR_CACHE_DIR`、`TORCH_EXTENSIONS_DIR`、`CUDA_CACHE_PATH` 等指向该目录的子目录；禁写依赖目录的 pyc，可设 `PYTHONDONTWRITEBYTECODE=1`。仅设置实际使用的缓存变量，不修改系统配置。

Git 只提交代码/脚本；结果、日志、profile、缓存、checkpoint、本文派生的实验记录保存在产物目录，不放进代码提交。保留原始证据，可以压缩归档，不能只留下总结后删除负例。用户不要求推送，因此只本地提交。

## 3. 三路比较、优势定义与“10 个”的计数

### 3.1 每个场景必须比较 U、UC、S

同一 case 固定：物理 GPU 数与映射、全局独立样本集合、每条长度、token/label、位置、prompt/loss mask、packing 顺序、模型参数、初始 checkpoint、精度、optimizer、重算、有效 batch、测量步数和环境。

区分独立 sequence 数、packed microbatch 数、梯度累积步数；不能把 GBS 当作 packed microbatch 数。每个 optimizer step 的真实输入 tokens、监督 tokens、`sum(L_i²)` 必须一致。

允许在不同 case 改配置；不允许在同一个成对比较中单独给 S 改 batch、数据、重算或模型。为保持等价，在不同 CP 布局上无需逐 kernel 随机数完全一致，但必须通过预设数值检验，且 dropout 固定为 0。

另外在代表性场景评估既有实现可行的更小统一 CP/更大 DP 基线：总 GPU 数和全局任务量不变，必要时调整合法 batch/累积以保持训练语义。若当前 fixture 不支持，记录限制，不在 A 阶段修改调度器。不要把只胜过指定 CP 的结论表述为胜过所有普通并行方案。

### 3.2 分类

| 分类 | 判定 | 计入主目标 10 个？ |
|---|---|---|
| `STRONG_WIN` | S 对 U、UC 均达到下面的稳定阈值 | 是 |
| `LAYOUT_ONLY_WIN` | UC 胜 U，或 S 胜 U 但不能胜 UC | 否，单独登记布局合并收益 |
| `SMALL_GAIN` | 有正向中位数，但未过阈值/置信检查 | 否 |
| `NO_GAIN` | 差异不足、噪声覆盖或基本持平 | 否 |
| `REGRESSION` | S 稳定更慢 | 否 |
| `CAPACITY_WIN` | baseline OOM 而 S 可运行 | 否，不将 OOM 视为无限 speedup；单独登记容量收益 |
| `OOM/TIMEOUT/ERROR` | 某路径未完成 | 否，但必须有记录 |
| `INVALID` | 数据/数值/版本/资源/指标不匹配 | 否，修正后新 run 复测 |

性能主目标默认是 S 的独立收益。UC 自身的有效优势也必须保存，但不能混入 S 的十个场景。若发现更强的普通 CP/DP 基线胜过 S，给该 case 加 `not_best_available` 标签，不能作为“最佳方案优势”宣传；重新复核其主目标资格。

### 3.3 正式优势门槛

- 粗筛后，独立启动至少 **5 组三路配对重复**，交替/随机化 U、UC、S 顺序，固定顺序种子并保留全部结果。
- 至少 10 步 warmup；正式测量至少 30 个连续有效 optimizer steps。编译尚未稳定时延长 warmup，并对全部模式使用相同的预先确定规则。不要根据快慢事后裁剪区间。
- `speedup_U = T_U/T_S`、`speedup_UC = T_UC/T_S`。对两种 baseline 分别要求重复运行的配对 speedup 中位数 **≥1.05**，按完整 run 为独立单位估计的 95% 区间下界 **>1.00**。
- 不将同一个 run 的 30 个相关 steps 当成 30 次独立实验。统计方法在首个结果前固定；可使用 run 级配对重采样，保存重采样种子。重复数太少或区间不稳定时增加到至少 10 组；区间不是跨所有候选的普适显著性证明。
- 筛选通过后，用未用于选择的固定数据 manifest/seed 做额外复核；不得只保留最好的 seed。该复核仍属于原 case，不增加场景数。
- 所有模式通过正确性检查，无跳过的更新/NaN；FLOPs 和全局工作量一致，all-rank 显存数据完整。性能 run 禁用 profiling 和额外 NVTX node hooks。
- 记录性能与显存的取舍：允许“更快但更占显存”计为性能优势，但必须注明增量和设备余量，不能称为显存优势。

### 3.4 独立性

case key 至少包含 `topology + total_gpus + parent_cp + child_cp + model + layer_pattern + length_distribution + packing_rule + global_workload + recompute`。

重复 seed、重复启动、不同日志目录、rank 顺序交换、只更改采样窗口均不算新 case。不同长度、模型/层模式、CP 组合或拓扑可以是不同 case，但十例至少覆盖 **3 个总长度档、2 种数据形态、2 个 CP 或层模式设置**，避免在同一参数附近轻微扰动凑数。单机和多机都必须搜索、报告；争取两类都出现优势，若某类没有则明确报告，不能强凑配额。

## 4. 阶段 A 的实验队列

### 4.1 初始范围

- 优先拓扑：单机 8 卡（CP8），两机各 8 卡共 16 卡（CP16）。若实际可用，可补两机各 4 卡共 8 卡，以及单机 16 卡；不要假设所有拓扑都存在。
- 起始模型：16 层、hidden4096、FFN16384、32 attention heads、8 query groups、GDN head dim128、GDN heads8、`linear_attention_freq=4`，其余对齐当前原生命令。
- BF16、TP1/PP1/EP1、dropout0、固定 parent CP、无 FP8/graphs/offload/MoE/MTP。初期 parent CP=总 GPU 数，DP1；先无重算，OOM 后开独立的匹配重算 case。
- child 优先 parent/2，再 parent/4；仅用当前代码已支持的整数嵌套组，不在 A 阶段增加分组算法。
- 长度均指 next-token shift 后的真实输入长度。K=1024；无物理 padding，满足 parent zigzag 的 `2*parent_cp` 对齐。数据标签不得跨独立序列边界。

### 4.2 第一轮种子矩阵

以下是候选，不是预期的“成功结果”。长配置先做容量检查，OOM 也入账。

| 家族 | 总 GPU/拓扑 | parent→child | 总 pack 长度 | 数据形态 | 子组数量 |
|---|---|---|---|---|---|
| A1 | 8，单机 | 8→4 | 32K/64K/128K/256K | 两条等长，各 S/2 | 2 |
| A2 | 8，单机 | 8→4 | 同上 | 两条 9:7 | 2 |
| A3 | 16，2×8 | 16→8 | 同上 | 两条等长 | 2 |
| A4 | 16，2×8 | 16→8 | 同上 | 两条 9:7 | 2 |
| A5 | 8，单机 | 8→2 | 64K/128K | 至少四条，先每条 S/4 | 4 |
| A6 | 16，2×8 | 16→4 | 64K/128K | 至少四条，先每条 S/4 | 4 |
| A7 | 8 或 16 | parent/2 | 64K/128K | 多条长短混合，但 child 总 tokens 尽量相等 | 2 |
| A8 | 8 或 16 | parent/2 | 64K/128K | 更强不均衡，如 3:1；用于负例 | 2 |

A1–A4 共 16 个首轮单元，A5–A8 再扩展。两条序列不能填满四个非空 child groups，遇此约束记录不可行，不复制序列填组。多序列案例要输出实际 greedy/bin 分配及每组总 tokens，不能只记录最长一条。

若现有小 launcher 只接受两条长度，不修改该 launcher 的执行逻辑来绕过 A 阶段权限：通过数据构造允许的配方扩展和完整原生命令运行。当前 scheduler 不支持的模式记 `UNSUPPORTED`，不在 A 阶段补实现。

### 4.3 自适应扩展

每批 4–8 个 case 更新一次搜索决策，不盲目穷举整个笛卡尔积：

1. 收益接近门槛：扩大连续 GDN 段长度（合法 CLI），用 1/3/7 个连续 GDN 的模式成对比较；层总数和精确模式入账。
2. 长组计算重：比较均衡、多序列补齐、不均衡三类 manifest；不能只交换长短组 rank 就称为解决不均衡。
3. 负载过轻：增加 pack 长度或层数；记录通信延迟摊薄是否发生。
4. 显存受限：另立 full/uniform/一层重算的三路 case；三个模式使用相同重算。不能只对 baseline 开重算。
5. FFN 代价明显：用已有 CLI 比较 hidden2048/4096 和相应 FFN；同 case 三路模型完全相同。
6. 多机收益明显：在可用的 8/16 卡拓扑上测试 child 是否完整落在节点内，并记录真实 rank→host→GPU 映射。
7. 数据形态至少包括等长、固定不均衡、多个完整序列的均衡 bin。再加入合法的变长 manifest，在同一步内守住工作量和组负载上限。

数据生成不只使用易拟合的 `token+常数`。保留该 fixture 用于连通性，正式代表性案例加入固定随机 token/不同监督比例等；这些不能代替真实 SFT 质量验证。监督比例改变通常不减少主干计算，必须同时记录输入吞吐和监督吞吐。

### 4.4 A 阶段结束条件

先完成种子矩阵中硬件/容量允许的项，再至少两轮按证据扩展。达到十个正式优势后仍须冻结完整负例清单，并覆盖至少单机、多机各两个可复现的非优势或最接近无收益的边界场景，供 B 阶段分析。

如果未达到十个，不能改低标准或立即偷偷优化核心代码。先继续搜索明显未覆盖的独立轴；当候选队列已完成、连续两轮没有新的有效优势且没有新的可执行假设时，写 `SEARCH_CLOSED_TARGET_UNMET`：列出数量、覆盖范围、硬件/容量约束和完整候选去向。此时可以明确切换 B 阶段分析和调优，但**整体目标仍未完成**；调优后冻结新实现版本，再进行同标准确认，补足十例或如实交付未达到目标的证据。

不存在的单机/多机资源不能用模拟结果替代。记录 `BLOCKED_RESOURCE`，完成可用范围内的工作，待真实资源补齐后关闭对应里程碑。

## 5. 正确性关卡

在昂贵搜索之前，先用小长度和相同 init state，在真实 Megatron/TE/FLA 路径比较 U、UC、S：

- 输入 tokens、labels、position、cu_seqlens、loss mask、sample ID 和每步样本顺序一致；保留 hash。
- 输出/loss、输入梯度（可获取时）、选定层参数梯度、更新后参数与 optimizer 状态进行比较。跨 CP 布局需先恢复同一逻辑 token 顺序。
- 初始合理 BF16 容差可采用 loss 相对差 1e-3、梯度归一化 L2 差 1e-2、梯度 cosine≥0.999；这些是初始 gate，不是所有 kernel 的保证。先用 U 对 U 重复确认数值噪声，记录固定容差及其依据，再观察 S，不能看完候选误差后临时放宽。
- 用短、非退化的训练任务跑至少 5 次 optimizer 更新，确认没有序列串扰、错误归一化、重复/遗漏样本、意外梯度缩放或 checkpoint 重算布局污染。
- A 阶段只能使用已存在的正确性入口和产物进行检查。若尚无可获取梯度的原生入口，不通过新 hook 绕过修改限制；标记未验证，不能计正式胜例，待合法的 B 阶段补齐并回溯复核。
- 性能重复与正确性运行分开；不在计时窗口复制梯度或保存完整中间张量。

仅有 loss 下降、20 步成功退出或 CPU reference test 不能证明真实训练等价。阶段 B 每个实现变更都要重新过关；发现错误的旧结果改为 `INVALIDATED`，保留原记录及失效原因。

## 6. 计时、FLOPs 与显存的统一口径

### 6.1 训练耗时

完整 step 必须包含 forward、backward、所需通信和 optimizer 更新。首选原生 iteration timer，保存全部 rank 日志并明确哪一 rank 输出全局训练日志。

正式性能运行关闭 Nsight、CUDA event 插桩、额外 Python autograd hooks；`MCORE_PACKED_CP_NVTX=0`。checkpoint/eval 不进入稳态窗口，相关配置三路一致。编译、数据准备、warmup、稳态和保存/eval 时间分别记录，不用整个进程墙钟冒充训练吞吐。

每次 run 记录 step mean/median/p90/p95/std、测量区间和样本数；主 speedup 使用同口径 mean step time 的配对比值。不要挑最快 rank 或最快一个 step。

没有原生 timer 时，优化器事件间隔只能标记为 `estimated`。不能把 profiler 估计与无 profiler 原生 timer 混在同一正式胜例比较中；先统一测量方式。

### 6.2 TFLOPs/s/GPU

用户要求的 `TFLOPs/s/sGPU` 在所有表中统一写为 **TFLOPs/s/GPU**：每张物理 GPU 的有效模型训练 FLOPs 吞吐。

定义：

```text
effective_tflops_per_gpu = sum(F_model_step) / (T_window_seconds × N_physical_gpus × 1e12)
input_tokens_per_second = sum(real_input_tokens_step) / T_window_seconds
supervised_tokens_per_second = sum(supervised_tokens_step) / T_window_seconds
```

- N 是整个作业的 8 或 16 张卡，不能用 child CP 大小，不能把节点 0 的 8 卡误当成 16 卡作业的总卡数。
- `F_model_step` 是同一逻辑训练任务的模型 FLOPs；包括实际 GQA/GDN 层型、投影/MLP/head 和约定的前后向项。精确记录估算器版本和假设。
- Packed GQA 的平方项使用实际 `sum(L_i²)`，不能使用整个 pack 的 `sum(L_i)²`。token 线性项使用实际 `sum(L_i)`；不要用监督 tokens 代替所有输入 tokens。
- 当前参考入口是 `megatron/training/training.py::num_floating_point_operations`，已有 `total_real_tokens_in_batch` 与 `seqlen_squared_sum_in_batch` 参数。必须核对服务器当前 GDN 分支和训练日志是否真的传入这两个量；只看到函数存在不够。
- 保存原生日志值 `native_tflops_per_gpu` 和可审计值 `effective_tflops_per_gpu`。若两者不一致，记录公式、输入和差异原因；A 阶段不修改估算器。无法可靠核对时填 null+原因，不编造数字，该项未齐全前不计正式胜例。
- 固定同一 case、同样逻辑 FLOPs 时，TFLOPs/s/GPU 的比值应与 step speedup 一致；不一致先检查统计错误。
- 不把 recompute、冗余计算、padding、通信或 memcpy 加进主指标以提高数字；如另报 executed FLOPs，应单列并说明其与有效模型 FLOPs不同。
- MFU 可选。分母必须对应真实 GPU 型号、BF16/FP8 和稠密/稀疏口径；没有可靠峰值来源时不报 MFU。

### 6.3 显存

所有 rank 分别记录：

1. PyTorch peak allocated，MiB 或 GiB，必须标明单位。
2. PyTorch peak reserved。
3. 可获得时的 NVML/nvidia-smi 设备 used memory 采样峰值、采样间隔及是否有其他进程。
4. 统计开始/结束时刻、是否 reset peak、初始化/优化器状态/稳态各阶段。

主表使用 **全作业所有 rank 的最大 peak allocated**，并保留 per-rank 数组。另报长组/短组、各节点的分布；只拿节点 0 的最大值不叫全局最大值。reserved 与 allocated 不能相加，设备采样峰值也不是精确 allocator 峰值。

A 阶段复用现有内存日志；若它是从进程启动累计的高水位，就标为 cumulative，不能声称是稳态某一层峰值。B 阶段可加统一的 peak reset/采样，但必须三路对称测量，不能把不同口径混用。

OOM 记录发生阶段、rank、已知峰值及失败栈；缺失数字为 null，绝不填 0。大模型 allocator OOM 和进程被系统杀死要区分。

## 7. 每次实验的落盘协议

输出结构全部位于 `$REPO/outputs/agent_packed_cp_campaign/`：

```text
campaign.json
STATUS.md
MILESTONES.md
cases.csv
attempts.jsonl
results.csv
source_manifests/
data_manifests/
cases/<case_id>/case.json
cases/<case_id>/decision.md
runs/<run_id>/command.txt
runs/<run_id>/manifest.json
runs/<run_id>/metrics.json
runs/<run_id>/steps.csv
runs/<run_id>/rank_memory.csv
runs/<run_id>/node_0/rank_*.log
runs/<run_id>/node_1/rank_*.log
profiles/<profile_id>/
milestones/<milestone_id>.md
final/advantage_cases.md
final/non_advantage_cases.md
final/root_causes_and_fixes.md
final/reproduce.md
```

`case_id` 示例：`C-001`；`run_id` 示例：`C-001-S-r03-v01`。版本、repeat、variant 是不同字段，不通过改 case_id 隐藏重跑。

**启动前写 manifest，结束立即补结果。失败也一样，不能等成功后才创建记录。** `attempts.jsonl` 追加事件，旧 attempt 不覆盖；汇总表可以从不可变 run 记录重建。中断后以磁盘状态恢复，禁止重复计数。

`manifest.json` 至少包括：

```json
{
  "case_id": "C-001",
  "run_id": "C-001-S-r01-v01",
  "phase": "A",
  "variant": "S",
  "status": "PLANNED",
  "code_commit": "TO_BE_RECORDED",
  "protected_source_hash": "TO_BE_RECORDED",
  "data_manifest_hash": "TO_BE_RECORDED",
  "init_checkpoint_hash": "TO_BE_RECORDED",
  "topology": {"nodes": 2, "gpus_per_node": 8, "total_gpus": 16},
  "parent_cp": 16,
  "child_cp": 8,
  "model_config": {},
  "sequence_lengths": [73728, 57344],
  "prompt_lengths": [36864, 14336],
  "sample_ids_by_step": [],
  "actual_group_assignment": [],
  "global_input_tokens_per_step": null,
  "global_supervised_tokens_per_step": null,
  "global_seqlen_squared_sum": null,
  "warmup_steps": 10,
  "measured_steps": 30,
  "profiler_enabled": false,
  "start_time_utc": null,
  "end_time_utc": null,
  "exit_code": null
}
```

该 JSON 只是模板，不是已运行结果；所有 `TO_BE_RECORDED` 和空字段必须由真实运行填充。

每行 `results.csv` 至少包括：

```text
case_id,run_id,phase,variant,code_commit,data_hash,repeat_id,status,
nodes,total_gpus,parent_cp,child_cp,pack_tokens,independent_sequences_per_step,
packed_microbatches_per_step,input_tokens_per_step,supervised_tokens_per_step,
step_mean_ms,step_median_ms,step_p95_ms,input_tokens_per_s,supervised_tokens_per_s,
native_tflops_per_gpu,effective_tflops_per_gpu,flops_formula_version,
peak_allocated_max_gib,peak_reserved_max_gib,device_used_sampled_max_gib,
memory_scope,all_ranks_present,correctness_status,exit_code,reason,evidence_path
```

case 汇总另存 U/UC speedup、区间、显存增量、formal_win、独立性分层和排除理由。失败行指标可为空，但 status/reason/evidence 必填。

设备/软件 manifest 记录 GPU 型号与 rank 映射、互联拓扑、驱动/CUDA/PyTorch/TE/FLA/NCCL 版本、相关通信环境变量及后台负载。只记录相关变量，不导出整个环境中的凭据或代理密钥。

## 8. 阶段 B：从非优势案例定位原因

先冻结 A 阶段结果、源码和数据 hash。至少选择：单机明显回退、单机边界/持平、多机明显回退或最接近边界、多机显存/不均衡受限案例。若某类没有真实回退，选择最弱收益边界并如实标记，不能改名成负例。

### 8.1 每个 case 先写诊断问题

对每个问题同时给出：假设、可推翻它的观测、计时边界、需要的 rank、预期的证据和最小干预。例如：

| 假设 | 需要的证据 | 可尝试的改进 |
|---|---|---|
| child token 不均衡 | per-group tokens、GEMM 时间、collective 到达差、长短组显存 | 改数据分桶/组分配或回退 parent CP |
| 边界转换成本过高 | pack/A2A/unpack 前后向、真实次数和字节数 | 减少重复转换、优化复制/索引、缓冲复用 |
| GDN 子组通信收益很小 | GDN conv/state 通信及核心计算分离 | 改 child CP，或此 case 不 split |
| 大量 metadata 同步 | CPU API 栈、小 D2H、同步点、context 构建次数 | 按完整长度/布局/group/device 键缓存 |
| FFN/投影主导 | attention-only 与 whole-layer 时间 | 测 attention 内转换，FFN 保持 parent；计算新增转换代价 |
| 参数/梯度通信主导 | optimizer shard、AllGather/ReduceScatter、梯度 ready 时序 | 合法调整 overlap/bucket 或调度，不改变梯度语义 |
| 小 shape / launch-bound | kernel 时长、launch 空隙、形状效率 | 合法融合/批处理，验证全流程收益 |
| 显存增加抵消容量 | all-rank 生命周期、workspace、转换缓冲峰值 | 缩短缓冲生命期、重算或布局策略；避免更长图驻留 |
| 拓扑不符合预期 | 实际 group ranks、host/GPU 对应、通信通路 | 在现有节点内约束分组，并验证全 WORLD 建组一致 |

不能把所有长 NCCL kernel 都解释成网络带宽不足。它可能等待其他 rank、前置计算、参数就绪或跨 stream 依赖。

### 8.2 CUDA event 与 NVTX

- 只在独立 profiling run 加插桩；测量额外开销，再用无插桩运行验证优化。
- CUDA event 在实际执行的 device/stream 上 record；读取 elapsed 前等待对应 event 完成，尽量在测量窗口结束统一等待，不在每层无条件 synchronize。
- event 只能测同一设备的有效区间，不用两张卡的 event 时间直接相减。跨 rank 到达差使用有时间关联的 trace；跨节点需要时钟关联，未经校准不能直接比较时间戳。
- 通信可能在 NCCL 内部 stream 执行。默认 compute stream 的一对 event 不自动等于完整通信时间。明确 completion dependency 或通过 Nsight launch/stream 关联确认；wait 本身改变 overlap 时须注明。
- 为 GQA、GDN、整层 FFN、transition pack/A2A/unpack、forward/backward/recompute、optimizer 加可区分标签；标明 layer、CP、rank、tokens。
- 多 stream 重叠用区间并集、重叠和依赖路径分析；不要把所有 kernel 时长相加当成 step wall time，不重复累加父/子 NVTX 范围。
- 多机必须收齐两节点；只得到节点 0 时标记局部证据，不能将其显存、等待或通信归因为全作业结论。
- 记录 Nsight 丢事件告警、采集窗口、所选 rank 和输出大小。NVTX CPU 范围是提交区间，不等于 GPU 完成时间。

### 8.3 模块级 profiling

允许在仓库内构造独立模块测试，但其 tensors、dtype、layout、cu_seqlens、CP group、mask、shape 和 backward 必须来自真实负例配置。必要时将小型输入/shape manifest 保存于仓库内。

分别测 GDN 核心、完整 attention、FFN、转换，再测包含依赖和通信的组合。做 warmup、多个重复；实际网络测试使用真实 process groups 和真实节点，不用 sleep 或假通信替代。

每个模块实验标注 `MICROBENCH_ONLY`。只有模块提速但完整训练不变/变慢时，记录为未转化收益，不能计十个优势场景。

### 8.4 调优循环

每次只验证一个明确机制：

```text
记录负例与假设 → 保存 baseline → 单项修改 → 正确性 → 模块诊断
→ 无插桩完整训练 U/UC/S 对照 → 接受或拒绝 → 写里程碑
```

如果修改了共享代码，重新测量 U 和 UC；不能用新 S 对比过时或故意较弱的旧 baseline。同时保存原始冻结版本对照，区分共同改善与 S 特有改善。

缓存键必须覆盖长度、布局、group、rank、device 和相关 dtype/配置，防止跨 batch 错用。异步缓冲复用须证明前后向生命期安全；建组修改须保证所有要求参与的 WORLD ranks 一致进入。不能通过改变 loss 归一、减少有效梯度、少做优化器步来换速度。

每项改进至少回归：原始单机负例、原始多机负例、一个均衡优势、一个不均衡优势、一个较长序列/显存边界。配置不支持时说明。失败的优化也保存源码 commit/patch、原因及回退记录；不使用破坏性 reset 覆盖证据。

## 9. 里程碑与持续记录

`STATUS.md` 始终包含当前阶段、当前 case、最近完成、下一步、strong-win 数量、累计 GPU-hours、阻塞和恢复指令。每完成一个 run 更新账本；每 4–8 个 case 更新搜索总结；发现机制变化或性能进步立即写里程碑。

| ID | 完成条件 | 必须保存 |
|---|---|---|
| M0 | 环境/路径/源码冻结、U/UC/S 能力核对 | 软件/拓扑清单、受保护源码 hash、权限白名单 |
| M1 | 真实小长度正确性及测量口径通过 | 数值误差、token/FLOPs 审计、all-rank 显存样例 |
| M2 | 种子矩阵完成、正负例全入账 | case 表、失败/无收益原因、下一轮假设 |
| M3 | A 阶段冻结；十例达标或明确未达标 | 已确认优势数量、重复统计、覆盖缺口、B 负例名单 |
| M4 | 单机和多机非优势各形成因果证据 | 原始 traces、计时边界、可证伪假设及归因边界 |
| M5.x | 每次可复现调优进步或重大诊断修正 | 前后代码 commit、指标、正确性、回归、代价 |
| M6 | 最终至少十例验证及负例闭环 | 优势/非优势清单、复现命令、适用边界、未解决问题 |

每个 `milestones/M5-xxx.md` 使用固定模板：

```text
时间 / 阶段 / milestone_id：
问题与原始 case_id：
修改前 commit / 修改后 commit：
假设与证据（含推翻过的解释）：
本次唯一主要修改：
完整训练前后：step、TFLOPs/s/GPU、tokens/s、all-rank 显存：
U/UC/S 对照、重复次数、区间和噪声：
正确性与回归结果：
代价、失败案例、适用范围：
采纳 / 拒绝 / 待验证：
原始日志、profile、manifest 的路径与 hash：
下一步：
```

记录进步不只限于“更快”：纠正错误 FLOPs、排除错误根因、恢复正确性、发现收益只来自布局合并，也属于必须留痕的重要认识变化。不得把失败尝试从历史中抹掉。

## 10. 运行卫生与异常处理

- 同一批 GPU 不并行跑互相干扰的性能实验。只清理由本任务启动并记录 PID 的进程；多机一端失败时终止本次对应 workers，避免其他 rank 永久等待。
- 每个 run 设有限超时，阈值根据 pilot 的最慢合法步和编译耗时预先确定并保存。相同条件出现确定性 OOM/错误后不无限重试；一次复核仍失败就入账并换候选。
- 记录异常退出、NCCL timeout、预热不收敛、热降频/其他 GPU 负载。排除受干扰 run 时必须保留它并说明规则；补跑新 run，不能覆盖。
- 实验前检查磁盘容量；profile 只覆盖有诊断价值的短窗口。可压缩已完成证据，保留 checksum 和索引。
- 不下载/使用无授权数据，不修改目录外资源，不为满足十例调用未授权硬件。不要把本计划扩展成自动发布、外发消息或后台定时任务。

## 11. 最终交付清单

1. 至少 10 个 `STRONG_WIN`，每例附 U/UC/S 完整训练结果、至少五组重复、独立复核、TFLOPs/s/GPU、全局显存、正确性证据和一条可复现命令入口。
2. 全部非优势和失败清单，包括 OOM/timeout/invalid/unsupported，不仅是选中的几个故事。
3. 单机与多机非优势的 profiling 报告，区分已验证根因、推断、未知；给出可行解决办法和已失败办法。
4. 调优代码的本地 Git 提交、每项进步的里程碑，以及最终的无 profiler 训练复核。代码提交不混入大数据和日志。
5. 最终决策建议：什么条件用 S、什么条件只用 UC、什么条件回退 U；用真实实验边界支持，不能仅凭 CP 越小越好。
6. 若不足十例或资源/正确性关卡未过，明确写 `TARGET_NOT_MET`，给出已完成数量与剩余可执行工作；不能以模块 benchmark 或历史粗略 profile 补数。

**执行顺序：M0 → M1 → A 阶段搜索及全量记录 → M3 冻结 → B 阶段单机/多机归因与调优 → 冻结新实现、按同一标准复核场景 → M6。**
