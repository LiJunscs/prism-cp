# 知识库主题：Heterogeneous CP 中面向空间批次槽不足的时序网格折叠发明专利工作范围

> 日期：2026-09-03  
> 任务类型：专利披露分析（disclosure analysis）  
> 状态：**不完整草案，供发明人与专利代理师评审，不构成可专利性、侵权或授权保证**  
> 证据基础：当前 `1f1b/` 实现、transition/runtime 代码、正确性测试、性能记录与总 Roadmap；申请前仍需冻结版本和正式检索。

## 1. 申请类型建议

建议申请 **中国发明专利**，不建议以实用新型作为主申请。

原因是本方案的核心是一种由多 GPU 计算设备执行的分布式神经网络训练方法，包括时序宽度计算、逻辑微批次聚合、序列分片重映射和伴随反向传播。国家知识产权局明确说明，实用新型只保护具有确定形状或构造的实体产品，不能保护方法；发明则可以保护产品、方法或其改进。[国家知识产权局：实用新型的保护对象](https://www.cnipa.gov.cn/art/2020/6/8/art_1750_107431.html)

本方案应当把算法步骤与 GPU 集群内部的数据布局、显存占用、设备间通信和训练执行效率结合起来描述。CNIPA 关于人工智能发明的审查说明明确认可：当算法与计算机系统内部结构存在特定技术关联，并改进硬件执行效率时，可以构成技术方案。[CNIPA《专利审查指南》修改解读](https://www.cnipa.gov.cn/art/2024/1/18/art_2199_189877.html)

## 2. 建议首案名称

首选：

> 一种异构上下文并行训练的时序网格折叠方法、系统、电子设备及存储介质

更聚焦的备选名称：

> 一种空间批次槽不足条件下的上下文并行布局转换方法及系统

“TMF”可以在说明书中作为英文简称，但不建议只用项目内部简称构成专利名称。

## 3. 一句话发明概念

针对长序列模型训练中单个执行时刻的独立样本数量不足以支持层组间 DP/CP 布局转换的问题，根据目标上下文并行布局计算时序聚合宽度，从同一优化器更新步内聚合多个带逻辑标识的独立微批次，执行可逆的序列分片重映射及其伴随反向转换，从而在不复制训练样本且不改变逻辑批次语义的条件下，获得目标层组的训练输出和对应梯度。

## 4. 首案应保护的闭合技术链

### 4.1 输入

- 一个由多个计算设备构成的阶段内物理网格；
- 待训练神经网络的层组计划；
- 每个层组的上下文并行度及对应数据并行度；
- 同一优化器更新步内依次到达的多个独立逻辑微批次；
- 每个逻辑微批次对应的逻辑样本标识和序列分片。

### 4.2 必要操作

1. 解析源布局和目标布局，并根据物理设备数、逻辑微批次大小及目标上下文并行度计算所需的时序聚合宽度。
2. 在同一优化器更新步内收集达到所述宽度的、内容互异的逻辑微批次，并保持逻辑样本标识。
3. 将逻辑微批次沿批次维打包，根据源 CP 组与目标 CP 组的设备映射重分布序列分片，形成目标的低 CP/高 DP 或其他 DP/CP 布局。
4. 在目标布局下执行一个或多个连续的目标层。
5. 离开目标层组时执行逆向网格重映射和拆分，将输出恢复为规范布局及原逻辑样本顺序。
6. 反向传播时执行与正向 scatter/gather 配对的伴随布局转换，使输入梯度返回到相应逻辑微批次和源序列分片。
7. 按逻辑样本而非物理副本统计训练输出、损失和梯度，使每条逻辑样本在一个优化器更新步内只被计入一次。

### 4.3 具体技术输出

经异构上下文并行层组计算、且恢复了逻辑样本与序列分片对应关系的：

- 模型训练输出；
- 各逻辑微批次的输入梯度；
- 目标层组的参数梯度。

不能只把最终节点写成泛化的“处理结果”或“优化结果”。

## 5. 建议保留的核心公式

设阶段内物理设备数为 `W`，第 `l` 个层组的上下文并行度为 `C_l`，有效数据并行度为 `D_l=W/C_l`，逻辑微批次大小为 `b`，则该层组需要的最小时序聚合宽度为：

\[
K_l=\frac{D_l}{\gcd(D_l,b)}.
\]

对于包含多个布局的执行计划，全局时序宽度为：

\[
K=\operatorname{lcm}_{l}(K_l),\qquad M\bmod K=0,
\]

其中 `M` 是同一优化器更新步内的逻辑微批次数。该公式分别由 `1f1b/plan.py:90-99` 和 `1f1b/plan.py:198-205` 明确实现，是首案中最适合形成限定和回退位置的内容。

## 6. 可以快速形成的保护层级

### 独立保护链

建议独立项覆盖第 4 节的完整方法链，但不要把以下内容限定死：

- 模型必须是 GDN 或 Transformer；
- CP 必须是 4→1→4；
- 必须使用 1F1B；
- 必须使用 Triton、NCCL 或 Megatron-LM；
- 必须是单机八卡；
- exact sequence length、hidden size、层数或具体 GPU 型号。

### 第一组回退位置

- 使用上述 GCD/LCM 公式计算时序宽度；
- 只允许具有整数包含关系的嵌套 CP degree；
- 当 `M mod K != 0` 时在执行前拒绝非法计划；
- 每个 `TemporalPayload` 携带一个或多个逻辑样本标识；
- 目标层组由多个连续低 CP 层组成；
- stage 入口和出口恢复为 temporal width 1 的规范布局；
- 当 `K=1` 时旁路 temporal cohort 路径。

### 第二组实现回退位置

- 正向 scatter 的 backward 使用对应 gather，正向 gather 的 backward 使用对应 scatter；
- 融合发送缓冲区打包和接收端最终布局落位；
- 使用预建立的多组上下文并行通信组；
- 在 cohort 粒度的流水线调度中进行前向、反向及 stage 间 P2P；
- 对输出宽度、逻辑标识顺序、输入梯度存在性和有限性进行运行时校验。

上述内容分别已有 `1f1b/tmf_adapter.py`、`utils.py`、`transition_kernels.py`、`1f1b/scheduler.py` 和测试代码支持。

## 7. 首案明确排除的内容

为了尽快形成支持充分、边界清晰的第一案，暂不将以下内容写成必要特征：

- ContextFlow IR 和多表示 HCP 抽象；
- CSA/HCA/QSA 专用 CP backend；
- 自动性能模型或 planner；
- FSDP 参数驻留和 AG/RS 调度；
- streaming/interleaved TMF；
- CP 与 MoE expert communication overlap；
- 多节点 topology-aware placement；
- 128K CP1 可运行性。

这些可以在完成实现和实验后形成后续专利，或在不影响首案支持性的前提下作为说明书中的展望，但不应抢先进入首案核心权利要求。

## 8. 与相邻技术的初步差异

不能把下列单点分别作为发明核心：

- 跨 mesh 的 tensor reshard；已有工作系统讨论不同设备 mesh 间布局转换。[On Optimizing the Communication of Model Parallelism](https://arxiv.org/abs/2211.05322)
- 把多个 microbatch 组成 group/chunk；已有流水线和动态 batch 技术覆盖一般聚合。
- 每个 microbatch 或 iteration 选择不同 CP；Dynamic CP 已有类似能力。
- shared-rank 多逻辑网格和普通 fan-in/fan-out；NVIDIA colocated 类工作会构成重要对比。

首案需要坚持的组合差异是：

> 在单个空间执行时刻缺少独立 batch slots 时，从同一 optimizer step 的时间序列中取得互异逻辑样本；依据 DP/CP 整除关系确定最小聚合宽度；在模型内部层组边界执行可逆 CP remesh；通过逻辑标识和伴随 backward 恢复样本及梯度对应关系，并保持 logical batch accounting。

初步检索已经发现一般 microbatch-group distributed training 和 device-mesh partition schedule 专利，例如 [US20220283820A1](https://patents.justia.com/patent/20220283820) 和 [WO2025120238A1](https://patents.google.com/patent/WO2025120238A1/en)。它们并不自动否定本方案，但必须在正式撰写前执行权利要求逐项对比。本文件不作新颖性或创造性结论。

## 9. 最快形成代理师交底包所需工作

### 已有，可立即使用

- `temporal_width`、全局 LCM 和非法配置门禁；
- 带 logical IDs 的 temporal payload；
- pack/remesh/split 和 identity restoration；
- scatter/gather 配对的 autograd backward；
- macro-cohort 1F1B 原型；
- output/loss/gradient calibrated correctness；
- 32K/64K 性能与 128K OOM 边界记录。

### 必须尽快补齐

1. 冻结一份代码和实验目录快照，记录 Git commit、dirty diff 和文件哈希。
2. 画两张图：主方法流程图、CP4/CP1 示例设备映射图。
3. 用一页表格说明 TMF 与 repeat、普通 gradient accumulation、普通 fan-in/fan-out、dynamic CP 和 cross-mesh reshard 的逐特征区别。
4. 确认发明人、权属、最早公开日期和已公开材料。
5. 由代理师执行正式专利检索，并根据检索结果确定独立项是否保留 GCD/LCM、MBS=1、同一 optimizer step 和 repeated internal boundary 等限定。

不需要等待 ContextFlow、QSA backend、FSDP 或顶会规模实验完成。对首案而言，现有代码已经足以支持发明概念和核心执行链；当前主要缺口是证据冻结、公开日期、权属与 prior-art 对比，而不是继续优化 cohort 粒度。

## 10. 发明人待确认事项

- [TO CONFIRM: 申请国家是否以中国为主？]
- [TO CONFIRM: 发明人名单及每个人对 temporal-width、routing、autograd、scheduler 和实验的具体贡献是什么？]
- [TO CONFIRM: 当前代码、roadmap、实验结果或汇报是否已经公开；最早公开日期是什么？]
- [TO CONFIRM: 申请权属于个人、学校还是合作单位；是否存在项目或劳动合同约定？]
- [TO CONFIRM: loss normalization、RNG、packed-sequence metadata 是否已有可复现实现，还是仅在设计文档中？]

在这些问题得到确认前，本方案可以进入技术交底撰写，但不能标记为 filing-ready。
