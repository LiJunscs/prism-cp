---
title: Dynamic CP 下 GDN 层异构的 8 卡与 16 卡测试用例探索计划
date: 2026-09-19
status: ready_for_agent
type: project_execution_contract
reference_commit: de0628bdfeb77a9899b7e3f19c7b1cb8d8182dde
implementation: local_working_tree_dynamic_gdn_cp_patch
execution_status: CPU 调度预演已完成，GPU 测试未启动
---

# Agent 执行计划：8/16 卡 Dynamic CP 优势场景探索

## 1. 本轮目标与边界

在 **Dynamic CP 始终开启**、原生 `DefaultDynamicCPScheduler` 保持不变的条件下，分别在单机 8 卡和两机各 8 卡上，验证 GDN 按完整序列拆分父 CP 组是否能提高完整训练吞吐，并确定适用的数据分布、子组规模、显存约束和通信拓扑。

本轮先冻结模型/路由/调度/优化器实现，仅探索输入和既有配置。先完成准备、正确性、粗筛、重复确认和归因；之后再提出下一轮代码优化。不要边改实现边把不同版本拼成一个性能表。

成功交付是完成两种规模的规定队列、保留所有结果并给出有证据的适用范围，**不预设必须有多少个胜例**。零收益也可以是有效结论。本文用于后续 agent 执行；本次只完成了 CPU 元数据探索，没有启动 GPU 任务。

旧固定 CP 用例计划仅作背景。本轮不要求强制固定运行时父 CP，也不使用现有 `--gdn-preserve-parent-contiguous` 开关，因为它目前与 Dynamic CP 不兼容。

## 2. 先理解为什么这样构造

DCP 会自行把短序列安排到 CP1。只增加 packing 序列数量，可能完全不触发 GDN 拆分。另一方面，无上限地追求 CP1 可能把最长序列留在单卡，使 GDN、投影、FFN 和残差的整段执行都受其拖累。

本轮采用两条主线：

1. **直接支持原始“尽可能多 DP”策略的用例。** 每个 CP2 父 pack 为 `[17K,7K,5K,3K]`。父组每卡16K，拆成 CP1 后两卡分别17K和15K；最长卡仅增加6.25%，且全部序列都参与切换。
2. **限制 GDN 每卡 token 的用例。** 在8卡上构造总128K、最长65K的父 pack，在16卡上构造总256K、最长129K的父 pack。最长序列略高于半个最大父组的容量，原生 DCP 因而选择全组；将 GDN 上限设为17K时，分别选择CP4和CP8，两个子组近乎均衡。

第二类尤其适合测试两机16卡：如果父CP16跨两个节点，而两个CP8子组恰好各在一个节点，连续GDN段的内部CP通信可变成节点内通信。**边界重分布以及原来的DP×CP梯度/优化器通信仍可能跨节点**，不能据此直接声称整步通信减少多少。

这些是可解释的机制用例，不代表真实数据必然如此。本计划另含抖动、广分布、无切换和不均衡对照，检查收益是否仅存在于人为阈值附近。

## 3. 三种策略及公平性

| 代号 | 开关 | GDN 行为 |
|---|---|---|
| U | `--dynamic-context-parallel`，不开GDN新开关 | 与GQA使用同一运行时父CP |
| S-max | U + `--gdn-dynamic-context-parallel`，不设child token上限 | 当前原始策略，尽量增加非空独立组 |
| S-cap | S-max + `--packed-sequence-cp-max-tokens N` | 最小可行child CP，但每卡token不得超过N；不满足则保留父组 |

S-cap 使用已有的简单容量约束，不新增成本预测调度器。每个正例都必须报告 S-max；不能只留下更快的 S-cap，将其当成原始策略的结果。

U/S-max/S-cap 必须共享输入样本、长度、padding、prompt/loss mask、位置、全局独立样本数、原生DCP计划、初始化权重、训练步数、模型、精度、优化器、重计算和物理卡映射。S-cap改变的是GDN执行方式，不得改变父packing或有效训练量。按样本/原始token/监督token核对每个optimizer step，不把GBS误当作packed microbatch数。

预演确认 S-max 和 S-cap 的路由及组划分完全相同时，可预先登记只计时 S-max；记录等价原因，不能事后按快慢删模式。控制用例只需 U/S-max。

当前 U→S 的差异包含连续 GDN 层间保留布局，以及缩小CP两个因素。没有动态父组的 parent-contiguous 对照时，只能声称“组合方案收益”；不能把全部收益归给更小CP。如果后续需要拆开贡献，单独立项增加该对照，并重新过正确性关卡。

## 4. 已预演通过的8卡矩阵

拓扑主设定为1机8卡；父调度器每卡上限B=16K。K=1024，长度为next-token shift后的真实输入token数。`×r`表示整组长度列表重复r次组成一个global batch，不是重复同一条样本内容。

| ID | 独立长度列表（K） | GBS | 总真实token/step | 原生GQA父组 | S-max | S-cap / 上限 | S-cap最长卡token÷父组每卡token |
|---|---|---:|---:|---|---|---|---:|
| D8-P1 | `[17,7,5,3] ×4` | 16 | 128K | 4个CP2 | CP1 | CP1 /18K，与S-max相同 | 1.0625 |
| D8-P2 | `[65,31,16,8,8]` | 5 | 128K | 1个CP8 | CP2 | CP4 /17K | 1.015625 |
| D8-P3 | `[33,15,8,8] ×2` | 8 | 128K | 2个CP4 | CP1 | CP2 /17K | 1.03125 |
| D8-P4 | `[65] + [7] ×9` | 10 | 128K | 1个CP8 | CP1 | CP4 /17K | 1.015625 |
| D8-C1 | `[16] ×8` | 8 | 128K | 8个CP1 | 不切换 | 不必重复计时 | 1 |
| D8-C2 | `[128]` | 1 | 128K | 1个CP8，单序列 | 不切换 | 不必重复计时 | 1 |

优先顺序：P1 → P2 → P3 → P4，同时完成C1/C2。P4用于验证序列数超过GPU数；其S-max最长卡负载为父组每卡负载的4.0625倍，应作为失衡对照，不能当作优先胜例假设。

## 5. 已预演通过的16卡矩阵

主设定为2机各8卡，global ranks 0–7在节点0、8–15在节点1，实际映射必须验证。B=16K，其他定义同上。

| ID | 独立长度列表（K） | GBS | 总真实token/step | 原生GQA父组 | S-max | S-cap / 上限 | S-cap最长卡token÷父组每卡token |
|---|---|---:|---:|---|---|---|---:|
| D16-P1 | `[17,7,5,3] ×8` | 32 | 256K | 8个CP2 | CP1 | CP1 /18K，与S-max相同 | 1.0625 |
| D16-P2 | `[129,63,32,16,8,8]` | 6 | 256K | 1个CP16 | CP4 | CP8 /17K | 1.0078125 |
| D16-P3 | `[65,31,16,8,8] ×2` | 10 | 256K | 2个CP8 | CP2 | CP4 /17K | 1.015625 |
| D16-P4 | `[129] + [7] ×17 + [4,4]` | 20 | 256K | 1个CP16 | CP1 | CP8 /17K | 1.0078125 |
| D16-C1 | `[16] ×16` | 16 | 256K | 16个CP1 | 不切换 | 不必重复计时 | 1 |
| D16-C2 | `[256]` | 1 | 256K | 1个CP16，单序列 | 不切换 | 不必重复计时 | 1 |

优先顺序：P2 → P1 → P3 → P4，同时完成C1/C2。P2/P4检验跨节点父组收缩到节点内子组；P3的父CP8本身已经位于节点内，是必要的拓扑对照。P1检验多父组并行执行，不能说它消除了原本并不存在的GDN跨节点CP通信。

P4的S-max最长卡负载比为8.0625，需先做小模型容量检查；OOM时记录，不删除该结果，也不拿S-cap的结果冒充S-max。

两张表的token负载比只是规划指标，不是时间、显存倍数或预测加速比。8卡128K与16卡256K属于每卡工作量大致匹配的弱扩展测试，不能把它们直接当作固定工作量的强扩展加速比。

## 6. CPU搜索产物与扩展队列

交付文件：

- `analysis/dynamic_gdn_cp_campaign/campaign.json`：12个命名用例、两种GDN策略的完整父/子计划、528条扩展候选及源文件SHA256。
- `analysis/dynamic_gdn_cp_campaign/cases.csv`：上述表格对应的机器可读指标。
- `analysis/dynamic_gdn_cp_campaign/summary.json`：搜索规模与筛选计数。
- `analysis/dynamic_gdn_cp_campaign/validation.json`：54次命令dry-run（含16卡两个node rank）及基础用例几何校验，均已通过；未启动GPU。
- `analysis/dynamic_gdn_cp_campaign/results_template.csv`：实验结果表模板，状态均为`NOT_RUN`，不预填实测性能。
- `scripts/build_dynamic_gdn_cp_campaign.py`：复用指定Megatron checkout的真实纯调度函数，重新生成以上结果，不启动GPU。

528条候选包含8/16卡、基础长度/B/GDN上限同步缩放0.5×/1×/2×、长度±5%/±15%抖动，以及广分布合成长度。随机种子20260919。候选ID的不同policy/seed不是独立性能胜例。

预先登记的几何粗筛条件：参与拆分的真实token占比≥75%，且所有父pack中最大的“child每卡token÷parent每卡token”≤1.15。8卡、16卡各有49条策略候选通过，共98条。**本次广分布合成对照没有通过该联合筛选的候选**；不能据此称方案对一般SFT长度分布普遍有效，也不能把未通过者全部删除。

Agent先复核12个种子；然后每种卡数各选最多6个新增数据配方进入GPU粗筛：

1. 2个通过几何筛选的抖动配方，避免与种子完全重复。
2. 2个不同长度规模的配方，覆盖B=8K或32K。
3. 2个广分布/低覆盖率对照，不要求通过筛选。

同一数据配方中 U、S-max、S-cap仍公平匹配。首轮这些配方在运行前写入`selected_cases.json`，避免根据某一次GPU耗时反复挑选输入。后续扩展需先写清新的问题和预算，再执行。

可额外测试连续optimizer steps轮换预先冻结的不同长度配方，覆盖运行时父CP变化。它需要增加明确的数据manifest/fixture；当前周期fixture的稳定分布不能被描述为已经完成了真实在线混合分布测试。优先使用脱敏的真实长度清单、合成token，不需要复制真实训练文本。

## 7. 模型与层模式

| 档位 | 模型配置 | 用途 |
|---|---|---|
| M0 | 现有启动器：5层，hidden512，GDN/GQA=`[0,1,1,1,0]` | 全矩阵启动、路由、数值与容量smoke，不用它代表生产规模 |
| M1 | 16层，hidden2048，FFN8192，GQA heads16/query groups4/head dim128，GDN key/value heads16、dim128，`[1,1,1,0]×4` | 主要端到端性能筛选 |
| M2 | 32层，hidden4096，FFN16384，GQA heads32/query groups8/head dim128，GDN key heads16/value heads32、dim128，`[1,1,1,0]×8` | 仅对M1中最有信息量的每卡数最多2个配方复核；先检查容量 |

全部使用BF16、TP1/PP1/EP1、dense、chunkwise GDN、dropout0、原生distributed optimizer和per-token loss。开始采用full/uniform/每次1层重计算；同一case的所有模式保持一致。若探索不重算，创建独立case并对所有模式一致修改。

当前`run_dynamic_gdn_cp_native.sh`固定5层、8 attention heads以及指定GDN参数，**仅能直接启动M0**。准备阶段允许新增/扩展可审阅的参数化启动器来表达M1/M2，不能只设置未实现的`NUM_LAYERS`等环境变量并假定生效。展开命令必须验证层数、head数、层型列表长度和比例；准备通过后冻结启动器。

连续GDN段长是第二阶段的机制维度：可对M1增加`[1×12,0×4]`这种同层数、同GDN/GQA计数的集中排列，测试边界成本的摊销上限。它是不同结构的合成模型，只能在各自结构内比较U/S；不能将其较快直接解释为原模型的优化收益。

## 8. Agent分阶段执行

### P0：核对代码、环境与拓扑

1. 记录真实commit、dirty diff、核心文件与启动器SHA256。参考SHA不包含本地尚未提交的新实现，必须确认`gdn_dynamic_context_parallel`代码及process group注入已经同步；仅checkout参考SHA不够。
2. 在服务器用随附脚本重新预演选定配方，记录源hash；确认父pack、child CP、样本覆盖率和本表一致。不一致先解释，不继续沿用旧预期。
3. 记录每卡型号、显存、节点、GPU UUID、rank映射、驱动/CUDA/PyTorch/TE/FLA/NCCL版本与互联拓扑。对16卡P2/P4核对每个CP8 child的所有ranks同属一个节点。
4. 不在forward中建新组，不更改DP×CP状态组。核对单个pack所有rank的split决策一致；确认init后CP1/2/4/8/16等所需组可用。
5. 固定样本manifest、样本内容生成规则、初始化checkpoint、随机种子、测量区间和模式运行顺序。8卡和16卡必须分别落盘完整记录。

产物：`source_manifest.json`、`environment.json`、`rank_topology.json`、`selected_cases.json`、展开后的命令。准备阶段如必须修bug，修复后重新冻结并重跑受影响检查；不要将修复前后的性能放进同一组配对数据。

### P1：正确性和容量，先于性能

已有四进程CPU/Gloo测试可先跑，但它不覆盖8/16卡真实CUDA内核。已有`test_dynamic_packed_sequence_cp.py`目前固定四进程，**不能把torchrun进程数改为8/16就当成有效测试**。

分别在真实8卡和16卡上用M0跑本表全部用例，U/S-max/S-cap每个不同计划至少5个完整optimizer steps。对前1–3步核对：原始及监督token总数、loss分子/分母、全局梯度、参数更新；同时验证重计算路径以及有padding的小型附加配方。路由往返、序列边界和token/loss mask索引要求精确一致；浮点归约结果允许预先登记的精度容差。

浮点门槛应在看性能前，用同配置U→U重复以及小规模高精度参考建立。可将原生BF16的loss相对差1e-3、全局梯度/参数更新向量相对L2差1e-2作为初始排错上限；这些不是通用等价证明。保存原始差值、逐层定位和近零分母处理，若超过上限暂停该用例并分析，不因出现性能收益而放宽门槛。只对齐loss或参数范数不足以证明更新一致。

确认没有NaN、跳过更新、样本丢失/重复、跨序列标签连接、通信超时；所有rank报告峰值显存。OOM、数值不一致、未触发split分别记为不同状态。

### P2：固定实现的粗筛

对通过P1的种子进入M1；先8卡P1/P2，再16卡P2/P1，其余随后。每个模式先做2次配对运行，每次70步：前20步warmup，步骤21–60为测量窗口，其余用于收尾。保存/评估不得落入测量窗口，所有模式固定同一规则。编译若仍未稳定，在看最终快慢前统一调整warmup并重跑配对。

性能运行关闭`MCORE_PACKED_CP_NVTX`和`MCORE_DYNAMIC_GDN_CP_LOG`。单独诊断run开启日志确认`parent_cp/gdn_cp/child_tokens`，不能将其时间与关闭日志的baseline比较。Nsight/profile另跑，不与无profile结果混用。

首先报告完整optimizer step时间、有效原始tokens/s、监督tokens/s和每卡峰值allocated/reserved显存。训练窗口的墙钟计时应在warmup后及窗口结束时对所有rank完成同步，窗口内不额外逐步同步；读取原生计时日志交叉核对。保存所有rank/两个节点的日志，不能只凭节点0的局部GPU事件认定全局加速。

TFLOPs/s/GPU是辅助项：若现有估算不支持可变长packing，应标记未校准，不用固定seq_length的平方替代各独立序列平方和。不给没有依据的精确算力利用率。

粗筛提名：两次配对均改善、step时间中位数至少改善3%，且没有正确性或容量问题。它只是复核入口，不是最终胜例。无收益、反向收益、没有split和OOM全部保留。

### P3：重复确认与分布验证

对每种卡数最多2个领先配方及1个最有解释价值的负例，进行至少5次独立配对运行，交替U/S顺序；使用单独冻结的确认seed/长度manifest，不能只复用挑选胜者的数据。

正式“可复现性能优势”的初始标准：正确性通过、完整step时间配对speedup中位数≥1.05，且5次配对均>1.00。报告每次结果、范围及run级不确定性，不将同一run的相关steps充作独立重复；若波动大或区间跨1，增至10次或标注收益不稳定。没有满足标准时如实记录。

8卡和16卡各自汇总，不强制两类都出现正收益。上述几何筛选和GPU筛选存在选择偏差，确认集上的结论仍只适用于相应模型/拓扑/分布，不宣称覆盖所有SFT。

### P4：归因后再提优化

对每种卡数至少一个正例（如有）和一个负例采集profile，区分：GQA、GDN、FFN/投影、parent↔child边界、原生梯度/参数通信、planner CPU同步和数据准备。

重点回答：

- S-max是否因为最长卡token增加而抵消通信收益？S-cap能否纠正？
- D16-P2/P4是否实际把GDN内部跨节点CP通信移到了节点内？边界交换是否又吃掉收益？
- 连续GDN段是否足够长，能够摊销两次布局/归属转换？
- 数据预演能拆，但原生训练是否被其他模块/状态通信主导？
- 无切换控制里规划开销是否可测？真实/广分布数据里的拆分覆盖率是否太低？

本轮输出下一步建议，不默认修改scheduler或kernel。只有解释清楚瓶颈后，再另开实现优化阶段。

## 9. 可直接启动的M0命令

从包含本研究`scripts/`的目录启动。下列路径仅为示例，替换为已同步新代码的真实路径；16卡两节点必须使用相同模型/输入/模式，并具有各自正确的node rank。保存checkpoint时使用两节点均可访问、布局一致的run目录。

8卡D8-P1，U与S-max：

```bash
MEGATRON_ROOT=/workspace/repos/Megatron-LM \
NNODES=1 NPROC_PER_NODE=8 MODE=uniform \
LENGTHS_JSON='[17408,7168,5120,3072]' GLOBAL_BATCH_SIZE=16 \
MAX_TOKENS_PER_RANK=16384 TRAIN_ITERS=5 \
OUT=outputs/dynamic_gdn_campaign/D8-P1/U-smoke \
bash scripts/run_dynamic_gdn_cp_native.sh

MEGATRON_ROOT=/workspace/repos/Megatron-LM \
NNODES=1 NPROC_PER_NODE=8 MODE=split \
LENGTHS_JSON='[17408,7168,5120,3072]' GLOBAL_BATCH_SIZE=16 \
MAX_TOKENS_PER_RANK=16384 TRAIN_ITERS=5 \
OUT=outputs/dynamic_gdn_campaign/D8-P1/Smax-smoke \
bash scripts/run_dynamic_gdn_cp_native.sh
```

运行S-max前确认没有继承`GDN_MAX_TOKENS_PER_RANK`环境变量。若不确定，先`unset GDN_MAX_TOKENS_PER_RANK`。加`DRY_RUN=1`可先审阅完整命令。

16卡D16-P2的S-cap，在**两个节点分别执行**，节点0设置`NODE_RANK=0`、节点1设置`NODE_RANK=1`，其余相同：

```bash
MEGATRON_ROOT=/workspace/repos/Megatron-LM \
NNODES=2 NPROC_PER_NODE=8 NODE_RANK=0 MASTER_ADDR=<节点0可达地址> \
MODE=split GDN_MAX_TOKENS_PER_RANK=17408 \
LENGTHS_JSON='[132096,64512,32768,16384,8192,8192]' GLOBAL_BATCH_SIZE=6 \
MAX_TOKENS_PER_RANK=16384 TRAIN_ITERS=5 \
OUT=outputs/dynamic_gdn_campaign/D16-P2/Scap-smoke \
bash scripts/run_dynamic_gdn_cp_native.sh
```

同一D16-P2的U改为`MODE=uniform`并移除GDN上限；S-max保留`MODE=split`但移除上限。三者使用不同OUT，每一配对共用同一个起始checkpoint/确定性初始化。16卡P2的S-max预期CP16→4，S-cap预期CP16→8，不能混淆。

8卡D8-P2的S-cap参数为：`NPROC_PER_NODE=8`、`LENGTHS_JSON='[66560,31744,16384,8192,8192]'`、`GLOBAL_BATCH_SIZE=5`、`GDN_MAX_TOKENS_PER_RANK=17408`。其余用例以`campaign.json`的原始整数为准，避免手动K换算错误。

## 10. 产物与交接完成条件

推荐每个实验目录包含：展开命令、输入manifest、父/子调度计划、源码hash、环境与拓扑、所有rank日志、数值差异、step原始计时、每卡显存和必要的profile。用唯一run ID连接所有证据。

总表至少记录`case_id/world/topology/model/pattern/policy/seed/GBS/real_tokens/supervised_tokens/parent_cp_hist/child_cp_hist/split_token_fraction/max_token_ratio/recompute/source_hash/correctness/status/step_ms/tokens_s/peak_memory_per_rank/speedup/evidence_path`。未知值留空并写原因，不能用预演数代替测量数。

交接必须分别给出8卡、16卡的：已完成/未完成矩阵，S-max和S-cap相对U的收益分布，显存代价，无收益/失败清单，真实拆分覆盖率，拓扑验证与profile结论。若缺16卡资源，完成8卡部分并明确16卡仍未验证，不能用8卡结果外推填表。

完成本轮基础矩阵与受限扩展、确认/归因后即可提交探索报告；不为凑胜例无限追加实验。最优策略可能是按简单容量上限选择CP，也可能是在某些分布保留父组，应由实测决定。
