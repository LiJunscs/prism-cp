# 知识库主题：GDN/GQA 3:1 层块异构 CP——优化实施与实验交接计划

日期：2026-09-19。状态：**可交给实现 agent；本计划未运行新的 GPU 实验，未修改 Megatron 生产代码。**

本地研究目录：`E:/LLM论文/mine/heterogeneous_cp`。本地实现目录：`E:/Project/Pythonproject/Megatron-LM`。在服务器执行时，用真实 checkout 替换这两个根目录，保留相对路径。

## 0. 给接手 agent 的任务摘要

保持现有 **GDN/GDN/GDN/GQA** 模型顺序、原生 Dynamic CP 的 parent packing、全局训练数据与参数/梯度/优化器通信域，完成以下闭环：

1. 冻结当前实现，补齐真实训练启动器和证据记录，复现现有 U/S 差异。
2. 将 GDN 子组规划所需 metadata 前移至 CPU，复用不可变几何计划及安全的执行元数据。
3. 增加动态父组下的 parent-contiguous block 对照，并复用已有等价 converter；验证所有权相同的情形能够避开通用重排成本。
4. 用现有 deferred gradient dispatch 验证 GQA 与梯度归约竞争；只有证据支持时才做受限的 GQA 窗口避让。
5. 对少量合法 child 方案与 parent 方案做整步成本选择，允许放弃切换；在未用于选策略的长度分布上验证。

**第一轮不做：** CSA/HCA/DSA 适配、改变层比例/顺序、重写 GDN scan、任意大小 communicator、逐层 CP 搜索、修改 parent packing 算法、PP/TP/MoE 扩展、重构模型状态分片或自定义两级 optimizer collective。

文中“state mesh”专指模型参数、梯度和优化器的通信/分片域；不要与 GDN recurrent state 混淆。GDN 是线性递推注意力，GQA 是 softmax grouped-query attention，研究对象是二者的混合模型。

## 1. 当前判断与已经完成的工作

### 1.1 为什么继续集中在 3:1

连续三个 GDN 层可以共用一次进入、一次退出的布局切换，较短的 GQA 间隔使这一单元适合重复优化。但这是**固定架构下的工程机会，不是性能保证**：3:1 的层数比例不等于 3:1 的训练时间比例，层顺序也必须确实形成连续段。

对于 `[GDN,GDN,GDN,GQA]×6`，有 6 个连续 GDN 段，通常对应 12 次前向边界转换及其反向逆路由。当前执行器已经合并连续 GDN，不能再以“从逐层切换改为三层一次”作为新增优化。最终以实际 layout/padding/fallback 与 trace 计数为准。

当前建议的研究问题是：

> 在固定模型状态通信域和真实 sequence 依赖下，怎样让连续 GDN 段选择较小的激活 CP 域，并通过计划复用、等价布局路径和受控通信调度，把局部通信节省转化为完整训练步收益？策略能否识别不值得切换的 pack？

### 1.2 已有证据及其限制

| 事实 | 实施含义 |
|---|---|
| 文件名含 `128k_64k` 的 trace 实际为 32K+32K、24 层、18 GDN+6 GQA、world16、S child8 | 按真实配置命名，不能继续当作 128K scaling 点 |
| iter10–16：UC 926.757 ms，S 942.743 ms，约慢 1.72% | 差距较小；不能用“至少 5% 加速”作为唯一筛选门槛 |
| 标记步 GDN 前向 AG 累计 22.681→2.270 ms，完整 GDN 相关 AG 69.910→26.288 ms | 局部通信收益已经存在，优先查收益为何未兑现 |
| rank8 的设备周期分解差额：非 NCCL 活动 +1.885、仅 NCCL +4.004、无活动 +10.027 ms | 不将 kernel 累计相加；不把 idle 直接归因于 Python 或 `.item()` |
| 这份 UC 开启 parent-contiguous，layout collective 次数与 S 相同 | 当前固定 trace 的收益不能再归因于新消除转换；它也不是下文动态 U-native |
| 256K/8 层 trace 后段有明显 GPU7 拖尾，且仅一节点 | 先排除设备/拓扑/共跑混杂，不用它拟合成本模型 |

上述时间来自已有一次运行的指定窗口，不是重复实验的稳定效应量。原始口径见 [Nsight 审计](E:/LLM论文/mine/heterogeneous_cp/analysis/current_nsys_audit_20260919/REPORT.md) 和 [256K 复核](E:/LLM论文/mine/heterogeneous_cp/analysis/nsys_reaudit_20260919/REPORT_256K.md)。

### 1.3 不要重复实现或错误归功

当前参考 HEAD 为 `de0628bdfeb77a9899b7e3f19c7b1cb8d8182dde`；必须另外记录工作区 patch，HEAD 不能代表完整实现。

- 已有完整 sequence 的等大小嵌套 child group、最小可行 CP 选择、GDN 每卡 token cap、单序列/不可行 fallback。
- 已有连续 GDN block、ExecutionBlock IR、route index 操作及部分缓存；不重新建设一套执行器。
- 静态路径已有 `parent_contiguous_equivalent` converter；动态路径目前显式排除它。
- `CellPlan` 已有 sidecar，但当前路径仍会从 CUDA lengths 重建计划并验证，不能称 CPU 预规划已完成。
- 已有 `defer_grad_reduce_dispatch`。真实 drain 在 `DistributedDataParallel.finish_grad_sync`；不要调用不存在的 `flush_deferred_grad_sync` 接口。
- 前一轮 quickwins 已复用 `plan.source_tokens`，无 padding 时直接生成零 mask；不要重新提交同一改动。
- 当前用户改动有 8 个文件在暂存区，quickwins 有 3 个文件未暂存。本轮交接以完整工作区为起点，不以 `git diff` 单独代表全部补丁。

修改前保存 staged/unstaged 两份二进制可恢复 patch、状态清单与 hash；保留原文件，按独立 patch 或隔离分支组织新修改。不要 `reset --hard`、覆盖 checkout、清理不相关未跟踪文件或自动推送。用户允许暂存，但不要求再次把不相关文件全部暂存。

## 2. 固定的语义与执行契约

以下条件适用于所有实现和性能实验：

1. TP1/PP1/EP1，dense GDN/GQA，THD，chunkwise GDN，零 dropout，无 FP8/FP4、CUDA Graph、额外 pack 尾部 padding；保留现有逐序列 padding 支持。只验证当前支持的无重计算或 full/uniform/每次一层重计算。
2. GQA 沿用当前 microbatch 的真实 runtime parent group。child 必须来自预建 group，成员及顺序是 parent 的合法嵌套子集；执行中不创建 communicator。
3. 一个 sequence 的历史状态不能因“换 CP”被切成多个相互独立样本。每个非空 child 组至少有一条完整 sequence；不使用空子组，也不跨 parent pack 借样本。
4. 参数、梯度、distributed optimizer 分片仍由原来的 DP×CP 状态组维护。改变 activation CP 不减少应参与全局梯度的样本，不新增一遍 child 平均，不改变 loss scaling。
5. 比较时相同 global optimizer step 必须包含相同 sample ID、有效 token、监督 token、prompt/loss mask、position 和权重初态。按原生 per-token loss 做全局归一化；不能将局部 mean 再等权平均冒充 token 加权目标。
6. 不同 child 下每卡 token 可能不均；必须记录最大 rank 的 token、内存和完成时间。cap 是子组选择约束，**不保证 parent fallback 的显存可行**，也不是完整模型显存预算的替代品。
7. forward 保存的 group、metadata、route 与 backward/recompute 一一对应；多次 forward 尚未 backward 时不得被后续 microbatch 的计划覆盖。
8. 生产模块采用显式 `ProcessGroupCollection`/group 传递，不新增全局 `parallel_state.get_*_group()` 读取。实现前读服务器 `AGENTS.md`、相关测试与格式化技能。

## 3. 能快速吸收的已有工作

| 工作 | 本轮吸收的机制 | 对应任务与边界 |
|---|---|---|
| [DCP](https://arxiv.org/html/2510.10620#S6.SS1) | 数据侧规划、计划与执行分离、预取覆盖规划耗时 | A1；不替换现有 TE/FLA 为整套 DCP executor |
| [DCP §7.5](https://arxiv.org/html/2510.10620#S7.SS5) | 通信总时长下降不一定改善因果训练整步，重叠结构很重要 | A3；这是相似现象的先例，不是本地根因证明 |
| [FlexSP](https://arxiv.org/abs/2412.01523) | 以执行时间与容量约束选择策略 | A4；先用本硬件测量表，不照搬 Ulysses 的成本系数 |
| [LASP-2H](https://arxiv.org/html/2502.07563v1#S3.SS5) | 混合线性/softmax 层具有不同通信对象 | 保持 layer-family 边界；该工作不证明本方案 child degree 选择的收益 |
| [FLA CP 实现说明](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/cp/README.md) | 递推状态通信、数值精度、前后向接口约束 | A1 的 context 审计及正确性；先读服务器实际版本，不直接缓存可变状态或移植 GLA 分片 |
| [Hydraulis](https://arxiv.org/abs/2412.07894) | 序列分配与执行时间联合考虑 | A4 的负载特征；本轮不修改 parent packing，联合重规划留作后续 |

更广的来源和限制见 [相关工作报告](E:/LLM论文/mine/heterogeneous_cp/analysis/dynamic_cp_literature_20260919/REPORT.md)。本轮贡献不能仅写成“不同层用不同 CP”，应以净收益、正确回退和完整训练证据为中心。

## 4. 基线必须拆开命名

| 名称 | 运行行为 | 用途/状态 |
|---|---|---|
| U-native | 原生 Dynamic CP，不开 GDN 子组功能 | 当前已有；主基线，不修改 parent scheduler |
| S-max | 原生 parent + 当前最小可行 GDN child，无 token cap | 当前已有；保留原始策略，不只报告更快的 cap 版本 |
| S-cap | S-max + 固定 GDN 每卡 physical-token cap | 当前已有；主要性能候选 |
| U-block | 相同 runtime parent CP，连续 GDN 保持 parent-contiguous；不拆成 child | A2 新增的动态对照；隔离“保持布局”与“缩小 CP” |
| S-cost | A4 在 U-native、U-block、少量可行 child 中选策略 | 待实现；默认不在线试错，不逐层切换 |

**当前 `--gdn-preserve-parent-contiguous` 与 Dynamic CP 不兼容，不能把两个已有 flag 强行拼起来当 U-block。** A2 要显式实现并验证 runtime-parent 版本，保留原限制直到新路径通过测试。

至少报告：`U-native→U-block`（连续布局贡献）、`U-block→S-cap`（含实际差异的 child 贡献）、`U-native→最终方案`（整体收益）。如果 A1 的公共优化也加速 U，必须对 U 应用同样优化，不能把公共收益全部记为 HCP 优势。

同一配置不同路径若几何和执行严格等价，可在**计时前**登记别名并省略重复；不能根据快慢事后合并。fallback 到 parent 也要标注执行模式：当前 S fallback 并不自动获得 U-block 的连续布局行为。

统一 CP8/DP2 可作为后续“parent 是否过大”的外部基线，但不替换原生 Dynamic CP 主对照；须单独核对同一全局工作量、归一化与状态分片。如果改变 packing、状态布局或 microbatch 调度，标为完整系统比较，不声称只测了 child CP。

## 5. 实现任务与独立验收

### A0：冻结版本，补齐训练与测量入口

**先做，其他任务依赖它。** 主要修改研究目录的 `scripts/run_dynamic_gdn_cp_native.sh` 或新增薄封装，不能用拼接任意参数造成重复 flag。

当前启动器硬编码 5 层 `[0,1,1,1,0]`，只改 `HIDDEN` 不会变成 M1；也未显式开启旧 Nsight 配置中的 `overlap-grad-reduce/overlap-param-gather`，默认还开 NVTX/决策日志。接手 agent 必须先解决这些差异。

要求：

- 显式支持模型层数、完整 layer pattern、GQA/GDN head 参数、FFN、重计算、两种 overlap 开关、bucket 配置、defer 开关；输出 resolved config，不将本文的新模式名伪装成已存在 CLI。
- 干净输出三个 preset：M0 正确性、M1 粗筛、M2 复现/确认。支持 dry-run，检查 pattern 长度和 head 维度。
- 正式性能关闭 `MCORE_PACKED_CP_NVTX`、`MCORE_DYNAMIC_GDN_CP_LOG`；内存/参数范数诊断放在单独 run 或固定窗口外，避免每步额外同步。U/S 日志粒度相同。
- 记录源码 HEAD、完整 staged/unstaged hash、脚本 hash、依赖版本及加载路径、模型参数数目、GPU UUID/node/rank、NVLink/跨节点 group 成员、bucket 分区与通信 dtype。
- 不改变训练的参数预取与 gradient bucket 行为来模拟 defer。先确认服务器实际开关是否能让 bucket 指纹保持一致。
- 保存所有 rank 日志、完整命令、运行退出状态、timeout/OOM 和输入 manifest。唯一 run_id 关联全部文件。

**验收：** dry-run 配置正确；M0 的 U/S 能运行完整 optimizer step；命令区分性能和诊断；相同配对的 batch、bucket、模型状态组一致。没有 16 卡时先完成 8 卡，不虚构多机结果。

### A1：CPU metadata 前移与不可变计划复用

**目标假设：** 消除执行器关键路径上的重复长度回读、计划构建及小对象创建；不预先承诺拿回 trace 中的 10 ms idle。

代码入口，均在 Megatron checkout：

- `megatron/core/packed_sequence_cp.py`：`dynamic_plan_from_packed_seq_params`、`plan_from_packed_seq_params`、`child_packed_seq_params`。
- `megatron/core/packed_seq_params.py`：现有 metadata/CellPlan 接口。
- `megatron/core/datasets/data_schedule_utils.py` 的 `_pack_sequences`、`build_packed_microbatches`，以及 `megatron/core/datasets/data_schedule.py` 中 THD packing→训练输入构造链；mock 数据位于 `megatron/training/datasets/varlen_dataset.py`，但不能只在 mock 里实现。已有 CPU 视图可能只含 padded length，必须保留真实 length 来源，不能混用。
- `megatron/core/transformer/transformer_block.py`：`_packed_sequence_cp_forward`。
- GDN 的 `common.py`、`gdn.py` 及服务器实际安装 FLA 的 CP-context 构造代码。

分成两个 patch：

**A1a：权威 CPU 几何。** 在数据已经完成 packing 后保存真实/物理长度、sequence 顺序和 padding 规则，生成不可变 sidecar。数据 worker 只生成可序列化 CPU 数据，不访问 CUDA、不传 process group；在训练主进程绑定预建 runtime group、构建一次验证过的 plan。保留原生 parent packing 算法和结果。消除 dynamic 路径“先构造 plan、再从 CUDA 回读重新构造并比对”的重复工作。没有 sidecar 的老调用者保留一次安全验证路径；不能简单置 `validated=True` 绕过输入检查。

**A1b：安全的执行缓存。** 缓存不可变 route、child lengths/mask 的只读 tensor、block plan，必要时缓存 converter 的静态路由。返回 per-invocation metadata wrapper，或证明对象在 forward/backward 完全只读后才共享整个对象。geometry cache 与本次 batch 的 sample identity 分开：同形状样本可共用几何，但不能错误共用样本、位置内容或 loss mask。

建议逻辑键覆盖：ordered real/physical lengths、layout、parent/child membership 与顺序、rank、device、相关 tensor dtype、token cap、模型 pattern/重计算相关路由契约、geometry schema/version。具体实现可拆分几何键与设备键；不可只用 token 总数、CP 大小或对象地址。

禁止缓存 activation、recurrent state、可变 FLA scratch、autograd context；FLA context 只有在安装版本明确分离静态 metadata 后才复用静态部分。设备 tensor 按需物化；跨 batch cache 有容量上限，报告命中率、冷启动/热启动时间和内存，不让长度抖动导致无限增长。防止 GPU metadata 原地修改后继续命中旧计划。

**正确性验收：** CP4/2/1、多序列、有/无 padding、相同总 token 不同顺序、相同 geometry 不同 sample ID、不同 runtime parent；并行保留两次 forward 后反序 backward，重计算开/关、输入/参数梯度及 optimizer 更新对齐。非法 Q/KV metadata、额外 tail padding、过期 sidecar 明确拒绝。CPU/Gloo 通过后仍必须做真实 CUDA GDN/TE/DDP 验证。

**机制验收：** CPU sidecar 和 plan 准备完成后，split 与 fallback 的模型执行热路径中，planner-owned CUDA→CPU scalar/list 读取为 **0**；每个 geometry cache miss 只构造一次计划。不能只做到“不随 GDN 层数增长”，现有 per-microbatch cache 已部分满足这一弱条件。数据准备侧的同步、跨 rank metadata 通信及完整 step 开销另计，不能将 `.cpu()` 移到另一个计时区间就声称消除成本。GPU timeline 不出现新同步或额外大复制。公共 GDN 校验优化同时测 U 与 S。即使 step 无收益，也交付已测开销与 `NO_STEP_GAIN`，不将整条研究路线判死。

### A2：动态 U-block 对照与等价 converter

**依赖 A0；与 A1 可分别实现，但合并时串行核验。** 不重写 ExecutionBlock compiler，只补齐运行时策略与 group/metadata 契约。

入口：`megatron/core/transformer/transformer_block.py` 中 `_packed_sequence_cp_forward`、现有 parent-contiguous 执行路径；`megatron/core/packed_sequence_cp.py` 的 `parent_contiguous_equivalent`；现有 `CpPartitionModeConverter`、`prebuild_thd_cp_partition_routes`；`transformer_config.py` 的兼容性校验。

**A2a：U-block。** 在动态 parent group 内，让连续 GDN 使用 parent-contiguous metadata 和 parent CP，GQA 恢复原 parent zigzag view。使用独立、默认关闭的实验策略接口；所有 runtime-parent、反向捕获和 checkpoint 逻辑都按调用绑定。不仅放开旧 flag 的 assert。

**A2b：动态 equivalent path。** 当 plan 明确证明 child 的全体 token ownership 与 row order 等于 parent-contiguous 时，复用现有 parent zigzag↔contiguous converter。核验 `cp_group`、TP/CP 辅助 group、padding、route cache、metadata copy 和反向使用的是本次 runtime parent，而非最大配置组。

当前 `megatron/core/context_parallel_layout/conversion.py` 的 `CpPartitionModeConverter.convert()` 会修改其 `packed_seq_params.cp_partition_mode`。因此不可跨未完成 forward 共享同一个可变 converter/metadata；默认只共享静态 route，每次调用创建独立 wrapper，不能仅因形状相同就复用对象。

这条路径**仍可能有通信**：zigzag→contiguous 通常仍需转换。它仅避免更昂贵的通用 route/pack/unpack。只有源/目标的物理所有权、行序及依赖都一致时才 identity；不能把 `parent_contiguous_equivalent` 当成“无需交换”的同义词。

**验收：** dynamic CP4→2、CP2→1、不同 parent 成员、均衡/不均衡、真实长度不同而 padded 相同、非等价 case、backward 与 full recompute；generic 与 converter 数值一致。记录实际边界次数、跨 rank bytes、本地 pack/unpack 和 max-rank span。等价路径不可增加通信次数；非等价路径正确保留 generic fallback。

A1b 负责缓存接口、A2 负责路由选择；同一个文件由整合 agent 串行合并，不能并发覆盖。每个 patch 独立正确性与性能记录，之后才能累积比较。

### A3：固定 bucket 的通信竞争实验；按证据决定局部 guard

**先用已有 hook，不新增通信调度框架。** 入口：`megatron/core/distributed/param_and_grad_buffer.py`、`distributed_data_parallel.py` 的 ready/dispatch/finish 路径及 `finish_grad_sync`。

在一个明确有 parent16→child8 的代表 case 上，做固定 bucket、相同 optimizer 的 2×2：

| | eager dispatch | all-backward deferred dispatch |
|---|---|---|
| U-block（若未完成先 U-native 并注明） | U/eager | U/defer |
| S-cap | S/eager | S/defer |

记录完整 step、GQA backward 完成 span、GDN span、bucket ready/dispatch/finish、RS drain、optimizer 开始与结束、参数 AG 首消费者等待。正式 step 用无 profiler run，机制用单独多节点 trace。

交互量定义：`I=(T_S,defer−T_U,defer)−(T_S,eager−T_U,eager)`。I<0 只说明该干预使 S 的相对差距改善，不能单凭它确定硬件因果；同时检查 GQA、RS tail 和 rank arrival。若改用 U-native/U-block，整张 2×2 必须一致。

**推进规则：**

- 若 defer 缩短 GQA/暴露竞争，但将时间搬到 optimizer tail，即使整步无收益，在发现可移动的通信窗口、最晚完成约束允许且下述跨 communicator 安全条件成立后，允许一个有预算的局部 guard 原型。
- 若 GQA span、等待与相对差距均未出现可重复变化，停止本轮调度分支，继续 A1/A2/A4，不上复杂 arbiter。
- 若 defer 直接有稳定整步收益，先把它作为固定策略候选；仍对 U/S 同时调优，不把共同收益全部归因于异构。

局部原型最多两个预登记策略：在确定的 GQA backward 窗口暂缓**新** RS dispatch、退出窗口按原顺序排空；以及一个固定的 in-flight bucket 上限。必须保持同一状态组中各 rank 的 collective 顺序一致，动态 parent 进度不同也不能各自选择不同队列顺序。只暂停 dispatch，不阻塞 ready hook，不假设 stream priority 能抢占已运行 NCCL。

prototype 前先解决：事件/工作句柄生命周期、最后 microbatch、no_sync 累积、ready count、unused parameter、已有 CUDA stream 依赖、最终 drain 与 checkpoint 重算重复回调。不能把 barrier 插入每个 block 来“保证一致”后仍声称轻量优化。

**跨 communicator 安全条件：** state 组内 bucket 顺序一致仍不充分。不同 rank 若分别先发 RS 后发 GQA、先发 GQA 后发 RS，可能在 CP/state communicator 之间形成循环等待。实现前写出通信发起、stream event、wait 的偏序，覆盖不同 runtime parent/child 与 recompute；用全部 rank 的 debug launch/wait trace 验证无环。若无法建立一致的安全顺序，只保留已有 all-deferred，或使用经验证的全 state 组一致的保守静态释放点；不得按每个 rank 的局部 GQA 状态自由控制跨组发起顺序。

**正确性：** eager/defer/guard 的 main-grad、Adam 更新和 save/resume 对齐；至少 100 step 动态 pack smoke 无 pending queue 泄漏或 collective mismatch。**性能：** GQA 缩短须大于新增 tail/其他路径损失；只减少重叠面积不算成功。

### A4：小候选成本选择器，允许 parent 回退

**在 A1/A2 版本稳定、A3 调度状态冻结后进行。** 第一版不联合修改 parent scheduler、不跨 pack 调配样本、不搜索每一层。一个 pack 的所有同类 GDN 段先使用同一方案；整步使用固定的状态通信策略。

候选最多四个执行方案：U-native、U-block、S-max 的合法 child、S-cap 的合法 child；去重后可能更少。使用已预建 group 和既有 deterministic assignment。候选不能只由“最小 child”组成，必须包含 parent 的实际执行模式。

先实现离线 `enumerate → validate → measure → select`：

1. 几何可行：完整序列覆盖、非空组、padding、合法嵌套与 shape。
2. 容量可行：实际 all-rank peak allocated/reserved 加预登记余量；不把 token cap 当精确显存预测。OOM 单独记录，不作为一个很大的伪造耗时。
3. 成本采样：固定模型、GPU 拓扑、bucket、重计算、dtype，测完整训练步。可用三层 GDN+邻接 GQA 的阶段 span 作解释，不能孤立 kernel 相加作为真值。
4. 建立少量 profile bins/查表，特征至少包括 parent/child CP、节点范围、seq 数、每卡最大 physical token、负载比、padding、转换类别、GQA 的各序列 L² 特征、模型和调度版本。GDN+MLP 的近线性 token 成本不能用统一 L²/CP 代替。
5. 选择 predicted step 最低的可行候选；预测优势小于测量噪声/模型误差余量时选择已测最稳健的 parent。未覆盖 geometry/topology 或 profile 过期时保守 fallback。

采样不得作为隐式在线探索混入正式训练。运行时以 CPU metadata 决策并缓存，parent 内所有 rank 一致执行；需要传播 decision 时显式核算开销，禁止每层建组或额外同步。若本轮只支持固定 cell/pattern 的查表，就如实声明，不能写成通用在线优化器。

报告 `regret=T_selected/T_best_measured_feasible−1`、fallback 原因/比例、错误切换率、规划开销和显存。oracle 必须来自留出 case 上相同候选集合的完整测量；未测齐候选则标为 partial oracle，不编造“最优”。正式测试集不得反过来调阈值。

**验收：** 正例能选择有效 child；原生 CP1、单长序列、严重失衡和未覆盖配置能回退；holdout 不劣于其公平 parent 对照超过预设容忍区间。选择 parent 也有执行/规划开销，必须实测，不能默认为 0。

## 6. 最小实验矩阵

### 6.1 模型档位

| 档位 | 配置 | 用途 |
|---|---|---|
| M0 | 现有 5 层、hidden512、`[0,1,1,1,0]` | 快速真实 kernel/路由/优化器正确性；不是 3:1 性能证据 |
| M1 | 16 层，hidden2048，FFN8192；GQA heads16、query groups4、dim128；GDN key/value heads16、dim128；`[1,1,1,0]×4` | 主筛选，固定模型结构 |
| M2-R24 | 24 层，hidden4096，FFN16384；GQA heads32、query groups8、dim128；GDN key16/value32、dim128；`[1,1,1,0]×6` | 对齐当前 64K/24 层实验尺度，仍需从原日志确认全部 resolved 参数 |

M1 与 M2 都保留完整 embedding、FFN、输出层、loss、backward、梯度同步、Adam/distributed optimizer。M2 如果 OOM，记录该配置的容量结论；不要只改 S 的重计算或偷偷改层数。若需较小模型，对整个配对新建 case ID。

统一 BF16、TP1/PP1、相同完整重计算策略。主性能配置显式开启与目标训练一致的 gradient reduce/parameter gather overlap，并固定 bucket；不带 overlap 的结果只能作为另一已标注配置。

### 6.2 数据种子

复用 `analysis/dynamic_gdn_cp_campaign/campaign.json` 和 `scripts/build_dynamic_gdn_cp_campaign.py`，在服务器实际源码重新预演。下表预期来自已有 CPU 预演，不是 GPU 结果。K=1024；长度为 next-token shift 后有效输入长度，`×r` 表示 r 组不同样本，不是同一内容重复。

| ID | W / 长度列表（K） | GBS / 有效总 token | 预期原生 parent | 预期 S-max / S-cap | 目的 |
|---|---|---|---|---|---|
| D8-P1 | 8；`[17,7,5,3]×4` | 16 /128K | 4 个 CP2 | CP1 /CP1，cap18K | 轻微失衡，CP2→1 |
| D8-P2 | 8；`[65,31,16,8,8]` | 5 /128K | CP8 | CP2 /CP4，cap17K | cap 避免过度下沉 |
| D8-C1 | 8；`[16]×8` | 8 /128K | 8 个 CP1 | 不拆 | 无收益时的管理开销 |
| D8-C2 | 8；`[128]` | 1 /128K | CP8 | 不拆 | 单长序列正确 fallback |
| D16-P2 | 16；`[129,63,32,16,8,8]` | 6 /256K | CP16 | CP4 /CP8，cap17K | 跨节点→节点内 child |
| D16-P3 | 16；`[65,31,16,8,8]×2` | 10 /256K | 2 个 CP8 | CP2 /CP4，cap17K | parent 已节点内的拓扑对照 |

parent 每卡上限 B=16K。须验证真实 rank/node membership 后才能把 child8 称为节点内。当前脚本会按 world 相关倍数做 physical padding；记录实际 padded lengths 和原生 scheduler 的所有 pack，不能只记录表中理想几何。

核心矩阵暂不增加旧计划 D8/16-P4 的极端 S-max、528 条扩展全集或全 CP sweep；这些保留为负例/后续预算。相同 token/卡的 8 卡与 16 卡表属于弱扩展比较，不写成固定工作量强扩展。

### 6.3 分轮执行，避免笛卡尔积

**R0 正确性与基线冻结。** M0 先做全部六个 case 的 U/S 启动，覆盖不同子组与 fallback；集成 A1/A2 后补新模式正确性，不做性能结论。小型 FP32 参考校验与真实 CUDA 测试见第 7 节。

**R1 原始候选粗筛。** M1 在 D8-P1/P2、D16-P2/P3 比 U-native、S-max、S-cap；严格等价者预先去重。C1/C2 只比较 U-native 与启用 S 后的 fallback。每个 distinct 模式先 2 次独立配对。已有硬件不够时按 8 卡优先执行，不能把多机任务标为完成。

**R2 单机制消融。** 每个实现 patch 只先在 D8-P2、D16-P2 和 D8-C1 上测 A/B；A1 对 U 与 S 同时应用。A2 用 U-native/U-block/S-cap 分解。若 fastpath 在上述数据均不触发，从 CPU campaign 预先选择一个满足等价证明的 case，明确标记构造用例。不要从计时结果倒选快路径输入。

**R3 通信干预。** 在 D16-P2 上做 A3 的 2×2；若其模型形状不能复现旧 trace 的 state 通信主导，再用 M2-R24 的实际 32K+32K 固定 parent16/child8 诊断 case。两套 packing/runtime 分开报告，不能混算交互量。原 static case 的 UC 使用其原 parent-contiguous 行为，不标成 U-native。

**R4 确认与泛化。** 最多保留一个 8 卡代表、一个 16 卡代表及一个信息量最大的负例，在 M1 做 5 次独立配对；主要正例再做 M2-R24 的公平确认。长度分布 holdout 至少包括 2 个预先冻结的 ±5%/±15% 扰动配方和 1 个广分布/低拆分覆盖配方，不按几何筛选把负例排除。可复用旧 campaign 的独立 seed。测试动态父组跨 optimizer step 变化，所有策略使用同一轮换顺序。

**R5 S-cost。** 只在稳定版本上建立训练集成本表，冻结阈值，在 R4 holdout 上比较 parent、S-max、S-cap、S-cost。对 oracle 候选缺测的 case 明确记录，不能拿训练集 winner 当泛化结论。

原始 R1 与修改后的 R2 版本分栏保存；最后在选定的同一最终版本重测公平基线。不要让 U 用旧版本而 S 用新版本。

## 7. 正确性、计时与接受标准

### 7.1 正确性优先且不能只比 loss

- CPU 路由测试应覆盖 token 一一覆盖、序列顺序、逆路由梯度、padding、预算 fallback、不同 runtime parent。现有 CPU/Gloo 隔离测试可复用，但不是完整 CUDA 训练的替代品。
- 测试入口优先扩展 `tests/unit_tests/test_dynamic_packed_sequence_cp.py`、`test_packed_sequence_cp.py`、`test_context_parallel_layout.py`、`test_attention_execution_plan.py` 和 `tests/unit_tests/ssm/gated_delta_net/test_gdn_parallel.py`；不要复制一套只验证 toy 的旁路测试来替代真实实现。
- 小型可控参考：route 的整数标识要求严格一致；toy FP32 输出/梯度按预登记数值容差比较。真实 BF16 GDN/TE 先做 U 对 U 重复及已知等价路径对照，保存误差分布，再冻结合理容差；不得看到失败后放宽。
- 同步比较 loss、有效监督 token、grad norm、选定及完整可行规模的 main-grad、参数更新；另做关闭 grad clipping 或记录裁剪系数的诊断，避免 clipping 隐藏差异。不能只以“两个 loss 很接近”判定正确。
- 至少 2 个 microbatch 的梯度累积、multiple forward/reverse backward、full recompute/no recompute、小规模 save/resume；调度 guard 另做 100 step 无挂起测试。
- mixed 与 U 的改变可能导致浮点规约顺序差异，不要求真实 BF16 bitwise 一致，但应有可解释误差且不超过冻结的精度预算。正确性失败的性能数全部标为无效。

### 7.2 计时协议

每次建议 70 step：1–20 warmup，21–60 measurement，61–70 收尾；编译未稳定时，在查看最终快慢前为该组配对统一增加 warmup 并重跑。保存/评估在测量窗口外，数据源、同步方式与日志一致。

正式吞吐来自无 profiler 的完整训练。measurement 窗口前后对所有 rank 完成必要同步，窗口内不额外逐步 barrier；记录窗口全局 wall time，原生 step 日志用于交叉核验和分位数。profile 另跑，双节点同时覆盖全部 rank，核对时间对齐；CUDA kernel 用 `globalPid + correlationId` 关联，不能跨进程只按 correlationId 合并。

每个 run 保存 step 原始序列，但独立重复单位是 run/pair，不是其中 40 个相关 step。交错 U/S 顺序（AB、BA 或预登记随机顺序），相同初态和数据清单。GPU 频率/功耗/温度/其他进程及慢 rank 单独记录；异常按事先规则整对重跑，同时保留原失败尝试。

主要指标：完整 step、input tokens/s、supervised tokens/s、all-rank max allocated/reserved。辅助指标：GDN/GQA/FFN span、layout bytes/span、CPU planner 时间、cache hit/miss、DtoH/同步、bucket 等待和 optimizer tail、split-token fraction、max child/parent token ratio、rank arrival spread。不将未校准的 packed FLOPs 估算当主要结论。

### 7.3 决策阈值

本题现有差距约 1.7%，所以 **5% 不是进入后续实验的硬门槛**。

先在每种拓扑的代表 case 做 3 组同策略独立重复，默认以 `median(abs(T_U,a/T_U,b−1))` 估计 run 级相对噪声，将实用收益阈值预登记为 `epsilon=max(1%, 2×该噪声估计)`。方法与阈值写入分析脚本，在看优化快慢前冻结；样本少时不能声称精准估计。这个规则是工程筛选标准，不是显著性检验。

- 粗筛 2 对只用于提名：改善达到 epsilon，或虽小但提供明确机制/负例信息的 case，可以进入确认；不要据此宣布成功。
- 确认至少 5 对，报告每对、配对 speedup 的中位数与区间。bootstrap 按 pair 重采样，5 对区间很不稳定；若临界或噪声较大，最多增至 10 对，否则标为 `INCONCLUSIVE`。
- 可接受优化：正确性通过、无显存预算违规、配对中位收益超过 epsilon，并且不确定性支持正收益；≥5% 可描述为较强收益，但不是预设目标。
- 热路径执行器优化的 C1/C2 若出现可重复超过 epsilon 的回退，先修缓存/分支管理开销，或让策略绕开该路径。
- 若显存更高，仅在同一预登记预算下仍可行时比较性能。低 CP OOM 的点不充当吞吐基线；最终比较“预算内最快的已测可行方案”。

对于 A3，all-deferred 无 step 收益本身**不是**否定局部 guard 的充分条件；必须结合 GQA 节省是否被尾部抵消。对于 A4，选择不切换不是失败，但若所有 case 都回退且未解决负面区间，不能宣称异构策略取得端到端优势。

## 8. 任务分派、依赖与止损

| agent 角色 | 负责 | 允许并行的边界 | 交付 |
|---|---|---|---|
| 集成/实验 agent | A0、所有版本冻结、R0–R5 编排、最终合并 | 独占启动器和结果汇总 | manifest、命令、状态表、总报告 |
| metadata agent | A1a/A1b、对应测试 | 与 A3 的只读归因可并行；不与 A2 同时改 block 文件 | 独立 patch、cache 契约、正确性/开销对照 |
| layout agent | A2a/A2b、对应测试 | 先在隔离补丁工作，依赖 A1 接口后串行整合 | U-block、equivalence 证据、转换分解 |
| 通信 agent | A3 的已有 defer 验证；有证据再 guard | 不自行改 bucket/state mesh；不抢占同一 GPU 同时跑基准 | 2×2、timeline、Go/No-Go、可选局部 patch |
| 策略 agent | A4，R5 分析 | 等稳定测量 schema/训练集；前期只写离线枚举与分析 | 候选集、冻结成本表、holdout regret |

建议执行图：

```text
A0 + R0
  → R1（现有实现的受限矩阵）
  → A1a → A1b → A2a → A2b → R2
  → A3 已有 defer 2×2（可提前在冻结基线做一次，最终版本再复核）
       ├─ 无竞争证据：停止调度分支
       └─ 有 GQA 节省/差距交互：最多两个局部策略
  → R4 确认与 holdout
  → A4/R5 少候选选择器
```

每条机制独立 patch、独立数值验证；合并后再测组合。不同 agent 不得同时在相同 GPU 上做性能测试。无需完成旧计划的大矩阵才能开始局部实现；也无需先完成 HFSDP 才能检验当前主线。

止损与转向：

- 正确性/collective 顺序/保存恢复失败：停止该实现分支的性能结论，修复后重跑。
- metadata/fastpath 没有整步收益：保留机制数据，不扩大为通用重构；转查 GQA/state 等待。
- 局部 guard 最多两种策略仍只搬移通信而不改善 step：停止 guard，不继续引入通用 CommArbiter。
- 仅人工阈值配方有收益，广分布大多不拆或负收益：收紧适用范围，优化 conservative fallback；不通过改变样本权重制造覆盖率。
- 多机 GPU 拖尾随物理卡或共跑模式变化：先诊断映射/设备/通信争用，不把不稳定样本加入 selector 成本表。
- 若最佳 parent 已占优，明确报告该区域 HCP 不必要。不能为了证明 idea 删掉 U-block 或其他同预算可行强基线。

## 9. 交付格式与最短启动指令

建议输出目录：`outputs/gdn_gqa_hcp_3to1/<campaign_id>/`。

```text
STATUS.md
source_manifest.json
selected_cases.json
resolved_models.json
patches/
correctness/
runs/<run_id>/{command,config,metadata,logs,timings,memory,exit_status}
traces/
results.csv
reports/{metadata,layout,defer,selector,final}.md
```

`results.csv` 至少包含：case/model/pattern/version/patch hash/topology/bucket fingerprint/policy/parent与child直方图/sequence与physical manifest hash/global样本数/input与supervised tokens/重计算/overlap/defer/route class/cache统计/正确性状态/run_id/pair_id/step时间/峰值显存/退出状态/证据路径。未知项为空并附原因，禁止将预演数据填作实测。

可直接交给 agent 的指令：

> 请执行本计划 A0→R0/R1→A1/A2→R2，保留 GDN:GQA=3:1、原生 Dynamic CP parent scheduler 和 DP×CP 模型状态组。先读取服务器实际源码与本地完整补丁，保护已有 staged/unstaged 改动。补齐可配置的真实训练启动器，不使用其默认 5 层和默认日志设置代表 M1 性能。每个机制独立补丁及数值验证，记录所有无收益、OOM 和 timeout。A3 先使用已有 defer hook；只有阶段证据支持才实现最多两个局部调度策略。稳定后执行 R4，再做 A4/R5。不得改层顺序、单独降低某个模式的训练工作量、重写 optimizer 或扩展稀疏注意力来完成本任务。GPU 资源不足时完成可做部分并准确标记待验证项；不输出未经运行的加速结论。

现有 M0 脚本和命令可从 [初版说明](E:/LLM论文/mine/heterogeneous_cp/progress/GDN_LAYER_DYNAMIC_CP_INITIAL_20260919.md) 复用；M1/M2、U-block/S-cost 和新增测量配置必须先由 A0/A2/A4 实现，不是假定已有的 CLI。

## 10. 与旧计划的关系及后续候选

本计划是当前用户收敛到 GDN/GQA 后的**本轮执行优先级**，不是删除旧路线。参考：

- [旧 8/16 卡探索计划](E:/LLM论文/mine/heterogeneous_cp/progress/AGENT_DYNAMIC_GDN_CP_8GPU_16GPU_EXPLORATION_PLAN_20260919.md)：复用 CPU campaign 和数据配方；大矩阵后置，修订 3%/5% 作为唯一门槛的做法。
- [旧 block/state-plane 计划](E:/LLM论文/mine/heterogeneous_cp/progress/AGENT_BLOCK_HCP_PERSISTENT_CELLS_AND_STATE_PLANE_PLAN_20260919.md)：block IR/defer 已存在；不重做 A1/A2/A4，也不把 all-deferred 无整步收益当作无条件否定局部 guard。
- [已有 quickwins 验证](E:/Project/Pythonproject/Megatron-LM/outputs/dynamic_cp_quickwins_validation/README.md)：保留，但不能替代真实 CUDA/DDP 验证。

第一轮之后，仅按实测瓶颈选择一个方向：

1. **GQA 通信拓扑。** 若仍由跨节点 GQA 主导，确认实际 TE/THD/runtime-group 支持后，测试节点内 A2A + 节点间 P2P；当前 query groups8 限制纯 A2A16，不能直接打开。对公平 U/S 同时应用。
2. **parent packing 与 child assignment 联合成本。** 若长度分布导致拆分率低/失衡，才扩展原生 packing-aware scheduler；这是新实验阶段，不能与本轮“固定 parent”归因混写。
3. **GDN compact-state/prefix 优化。** 若单条长序列不能拆且 GDN 仍在关键路径，可参考 FLA/LASP 类方法独立研究；需保持 GDN 的状态转移代数与前后向精度，不把 GLA 的分片方案直接复制到 GDN。

本轮不承诺具体加速倍数。交付成功是给出可复现的优化、适用区间和回退规则；只有完整训练对照支持时，才把它提升为性能结论。
