# 知识库主题：混合注意力模型的 Block 级异构 CP、持久 Sample Cell 与独立分层参数通信平面实施计划

- 日期：2026-09-19
- 状态：`READY_FOR_AGENT`
- 文档类型：实现与实验执行合同
- 本地参考仓库：`E:/Project/Pythonproject/Megatron-LM`
- 本地参考提交：`77b9d342d1e10043f4bd8c383ed7f3a578a1cecc`
- 服务器目标仓库：`/workspace/repos/Megatron-LM`
- 当前执行状态：只完成代码、trace 和方案审计；本文没有声称任何新增 GPU 实验结果

## 0. 任务结论与范围

保留 GDN 与 GQA 的异构 CP，但把它明确限制为**自然连续 block 级**，不做逐层动态搜索：

```text
重复 motif：[GDN, GDN, GDN] block → [GQA] block

GDN block：sample-cell CP，例如 CP8
GQA block：parent CP，例如 CP16
参数/梯度/optimizer：固定 state mesh，整个训练过程不随 block 改变
```

本计划采用三平面设计：

1. **数据归属平面**：每个 packed sample 在一个 microbatch 的 forward、recompute 和 backward 期间拥有不变的逻辑 home cell。
2. **激活执行平面**：层类型只选择执行 view 和算法；GDN 使用 child-cell view，GQA 使用 parent view。物理激活允许在 block 边界变换，但不能重新决定 sample 属于哪个 cell。
3. **模型状态平面**：参数、梯度和 optimizer shard 独立于激活 CP。第一阶段保持 flat parent RS16/AG16；只有因果实验成立后，才评估固定的分层 state mesh。

该设计不否定 block 边界转换。第一版的 persistent ownership 指**逻辑 home-cell 持久**，不承诺 GQA 能直接消费长期 CP8 物理布局。标准 TE CP16 GQA 仍需要 parent execution view。

## 1. 已确认的现状，禁止重复包装为新贡献

当前代码已经不是“每个 GDN 层都切换一次”。

- `TransformerBlock._packed_sequence_cp_forward()` 只在 `use_child != in_child` 时转换；连续 GDN 层共享 child layout，离开连续 GDN run 才恢复 parent layout：`megatron/core/transformer/transformer_block.py:827-894`。
- 对 `[GDN,GDN,GDN,GQA]`，当前行为已经是 GDN run 入口一次、出口一次，而不是三层各做一次。
- `[1,1,1,0]*4` 的边界次数测试已经存在：`tests/unit_tests/test_packed_sequence_cp.py:366-438`。
- child layer 只替换 activation `cp/tp_cp`，`dp_cp` 仍沿用 parent：`transformer_block.py:430-449`。
- `PackedSequenceCPPlan` 已缓存一次 microbatch 的 assignment 和 route：`megatron/core/packed_sequence_cp.py:82-197`。

因此第一项工作是把散落的隐式机制整理成 first-class block IR，并证明行为不变；不是再实现一次 run coalescing。

## 2. 必须守住的语义不变量

### 2.1 Sample ownership

- `sample_id → home_cell_id` 在一个 logical microbatch 的 forward、activation recompute 和 backward 中不变。
- 相同长度但不同 sample ID 不能错误命中旧 plan。
- GQA 的 parent view 是临时执行 view；执行后仍返回相同 home cell，不能把 parent rank 暂时持有数据解释成 sample owner 已改变。
- 不等长 sample 可以有不等 token 数，但 planner 必须显式记录 imbalance，不能在层内重新贪心换 cell。

### 2.2 Block execution

- block 是最大连续相同 execution policy 的层段，不是单层。
- 初始只支持完整 TransformerLayer 跟随 block layout：attention、残差、MLP 一起执行，避免层内新增第二套布局语义。
- 未来 KDA、DSA、Mamba 或其他变体通过 policy registry 映射到 execution domain，executor 不允许硬编码 `if GDN`。
- 模型层顺序和 3:1 比例保持真实架构，不允许为了减少边界而重排层。

### 2.3 State mesh

- `S_param`、`S_grad`、`S_opt` 在模型初始化时确定，训练中不随当前 block 的 CP 改变。
- 第一阶段：`S_param = S_grad = S_opt = parent DP×CP16`。
- loss 使用 logical-token SUM，并在固定 parent 域按 global valid-token count 归一化。
- child cell 不能拥有独立 optimizer 副本；child-local partial gradient 必须最终覆盖整个 parent state mesh。

### 2.4 Collective ordering

- 所有 process group 在初始化阶段按 WORLD 一致顺序创建。
- 同一 communicator 上所有 rank 必须按同一 bucket 顺序 enqueue collective。
- sample 数、长度、cell assignment 或局部 ready 时间不能改变 collective 调用序列。
- 延迟 RS 只改变 dispatch 时间，不改变 bucket 内容、group、reduce op、shard owner 或最终 drain 顺序。

## 3. 目标软件抽象

首版只增加必要的三个 IR，不构建大型通用调度器。

### 3.1 `CellPlan`

一次 packed microbatch 唯一的数据归属计划：

```python
@dataclass(frozen=True)
class CellPlan:
    plan_id: str
    sample_ids: tuple[int, ...]
    sample_lengths: tuple[int, ...]
    padded_lengths: tuple[int, ...]
    home_cell_by_sample: tuple[int, ...]
    ranks_by_cell: tuple[tuple[int, ...], ...]
    local_tokens_by_rank: tuple[int, ...]
    parent_cp: int
    child_cp: int
    topology_signature: str
    forward_route_id: str
    inverse_route_id: str
```

实现原则：

- 复用当前 `PackedSequenceCPPlan.assignments/edges/local_route()`，不要重新写路由算法。
- scheduler 若能提供稳定 `sample_ids/cell_ids`，模型必须消费该预编译计划；不得在每个 GDN block 内重新运行贪心分配。
- cache key 至少覆盖 sample IDs、真实/物理长度、group identity、partition mode、device 和 metadata version。
- mutable transition buffer 不放进全局 cache；首版只缓存 immutable metadata 和 index tensors，避免 recompute/并发覆盖。

### 3.2 `ExecutionBlockPlan`

全模型或一个 PP stage 的连续 block 列表：

```python
@dataclass(frozen=True)
class ExecutionBlock:
    block_id: int
    layer_start: int
    layer_end: int          # exclusive
    family: str             # GDN, GQA, DSA, KDA, MAMBA...
    execution_domain: str   # HOME_CELL or PARENT
    cp_size: int
    partition_mode: str
    group_key: str
```

编译示例：

```text
layer pattern: [GDN,GDN,GDN,GQA,GDN,GDN,GDN,GQA]

block plan:
  B0 [0,3) GDN HOME_CELL CP8 contiguous
  B1 [3,4) GQA PARENT    CP16 zigzag
  B2 [4,7) GDN HOME_CELL CP8 contiguous
  B3 [7,8) GQA PARENT    CP16 zigzag
```

默认编译结果必须与当前 `use_child != in_child` 行为完全一致。

### 3.3 `ExecutionView`

`ExecutionView` 描述当前物理张量如何解释，但不改变 `CellPlan`：

```text
HOME_CELL view：child group、contiguous、cell-local cu_seqlens
PARENT view：parent group、zigzag、global packed cu_seqlens
```

状态机只有：

```text
HOME_CELL → HOME_CELL：无转换
PARENT    → PARENT：无转换
PARENT    → HOME_CELL：复用现有 forward route
HOME_CELL → PARENT：复用严格逆 route
```

每条边记录 `(step, microbatch, plan_id, block_id, direction, sequence_no)`，用于 trace 和 collective-order 审计。

## 4. 数据流

以两条 64K、parent CP16、两个 node-local CP8 cell 为例：

```text
batch scheduler
  └─ CellPlan: seqA→ranks[0..7], seqB→ranks[8..15]

parent input view
  └─ transition once
      └─ GDN block 0: three layers, HOME_CELL CP8
          └─ transition once
              └─ GQA block 1: PARENT CP16
                  └─ inverse of the same CellPlan
                      └─ GDN block 2: same home cells, no replanning
```

这里的“长期归属”意味着每次回到 HOME_CELL 时，seqA/seqB 返回原 cell；不意味着 CP16 GQA 不做物理 parent view。

### 4.1 可选研究扩展：GQA operator-local borrowed view

只有 block plan、RS 调度和固定 state plane 验证后，才允许实验：

- hidden/residual 的权威布局保持 HOME_CELL；
- GQA 仅把 Q/K/V/O 暂时物化成 parent CP16 operator view；
- core attention 结束后把 O 恢复到 owner layout，再做残差和 MLP；
- 连续 GQA layers 是否复用 parent view由独立代价模型决定。

这需要新的 GQA dataflow/adapter，标准 TE CP16 不能直接消费“前8卡只有seqA、后8卡只有seqB”的 cell layout。首版不得承诺零 remesh，也不得把 operator-local staging 改名后隐藏其字节量。

## 5. 参数同步平面：严格分阶段

### Phase S0：flat parent，保持现状

```text
Param AG: parent AG16
Grad RS:  parent RS16
Optimizer ownership: 每 rank 唯一 1/16 shard
Dispatch: bucket ready 后 eager
```

这是所有新机制的共同基线。

### Phase S1：先做延迟原 RS16，不改数学

目标是验证“RS 与 GQA critical path 冲突”是否真能改善整步，而不是直接实现复杂优先级。

第一版只做两个控制：

1. `eager_parent_rs`：现状。
2. `deferred_parent_rs_at_bwd_end`：保持原 bucketization、RS16、shard mapping 和 SUM/scale，只延迟 dispatch，完整 backward 后按固定 bucket 顺序 drain。

实现建议：

- 不把 `overlap_grad_reduce=False` 作为唯一 deferred 实验，因为该开关还会把 bucket size 设为 `None`，同时改变 bucketization。
- 正式实现增加独立的 `defer_grad_reduce_dispatch` 门控：最后一个 microbatch 仍保持 `is_last_microbatch=True`，parameter hooks 正常记录 ready count；bucket group 达到 golden ready count 后只进入有序 ready queue，不立即调用 `start_grad_sync()`。
- backward 结束后调用 `flush_deferred_grad_sync()`，严格按现有 `bucket_groups + expert_parallel_bucket_groups` 顺序发射，再正常 `finish_grad_sync()`。
- 不得把“最后一个 microbatch 也长期包进现有 `no_sync`”作为正式实现：当前 `register_grad_ready()` 在 `is_last_microbatch=False` 时不会学习首批 golden ready counts，可能使 `is_first_batch` 一直为真。只有在先用 eager step 建立 golden counts 后，才允许把 all-`no_sync` 用作一次性的零代码趋势诊断，并明确标成临时 ablation。
- 单独 sweep 16/32/64/128M bucket；不同时改变更多参数。

若全延迟只把时间等量搬到 optimizer tail，则停止开发在线 GQA guard。

### Phase S2：block-safe GQA guard

只有 S1 证明 contention 有净收益，才实现选择性 admission control：

- 自定义 autograd boundary marker 在 GQA backward 前后发布 deterministic phase；不依赖 CUDA stream priority。
- ready bucket 先进入有序 pending queue。
- GQA protected window 内不发可能跨越窗口的 parent RS；窗口后按全局一致 bucket order flush。
- 第一版使用静态 block boundary，不做预测型在线 scheduler。
- 后续可用 RS EWMA 做 deadline admission，但所有 rank 必须得出相同决策。
- backward/optimizer 前有强制 drain；reset 时 pending queue 必须为空。

候选落点：

- 普通 DDP hook：`megatron/core/distributed/distributed_data_parallel.py:500-542`。
- bucket ready/dispatch：`megatron/core/distributed/param_and_grad_buffer.py:913-935`。
- 强制 finalize：`megatron/core/distributed/finalize_model_grads.py:552-558`。

### Phase S3：使用现有 Megatron-FSDP/HFSDP 作为精确层次通信 oracle

不要先手写 RS8→RS2。仓库已有精确实现：

```text
配置：
  use_megatron_fsdp = true
  data_parallel_sharding_strategy = optim_grads_params
  num_distributed_optimizer_instances = 2
  outer_dp_sharding_strategy = optim

梯度：inner SUM-RS8 → outer SUM-RS2
参数：outer AG2 → inner AG8
最终 ownership：parent 范围内唯一 1/16 shard
```

关键实现：

- mesh 适配：`megatron/core/distributed/fsdp/mcore_fsdp_adapter.py:370-455`。
- inner RS：`megatron/core/distributed/fsdp/src/megatron_fsdp/param_and_grad_buffer.py:4176-4255`。
- outer RS：同文件 `4273-4345`。
- outer AG 后 inner AG：同文件 `4681-4713`。
- logical rank/shard mapping：`megatron/core/distributed/fsdp/src/megatron_fsdp/utils.py:729-770`。
- 已有组合测试：`tests/unit_tests/distributed/mfsdp_v1/test_mcore_fully_sharded_data_parallel.py:1040-1057`。

HFSDP 是正确性和性能 oracle，不是普通 DDP 的无扰动小补丁。它强制 inner full shard，会改变 FSDP unit、parameter AG 和 buffer 生命周期。实验必须把它单列，不能把差值全部归因于“两级 RS”。

### Phase S4：outer-only defer

只有 HFSDP 正确、有效且 outer RS 仍与 GQA 冲突时才修改：

- inner RS 正常执行并记录 completion event；
- outer RS 进入 deterministic queue；
- GQA protected window 后或 backward 末尾 flush；
- 新增类似 `INNER_REDUCED` 的 bucket 状态；
- outer stream 必须等待 inner event；
- outer 完成前禁止 free/reset helper grad buffer。

### 明确禁止把普通 multi-instance DDP 当成精确 hierarchy

普通 `num_distributed_optimizer_instances=2` 的语义是：

```text
inner RS8 → outer AR2
每个 cell 保留同一 1/8 optimizer shard
optimizer state 复制两份
```

它可以作为通信拓扑诊断，但不能称为 parent 1/16 ownership，也不能作为最终独立参数平面。若使用，报告中必须写成 `RS8+AR2 duplicated-state control`。

## 6. 代码实施阶段

### P0：冻结、审计和离线分析（无 GPU 也可推进）

任务：

- 保存当前 commit、dirty diff、环境和已有 trace hash。
- 用现有 S/UC sqlite 生成 block 级报告：GDN/GQA fwd/bwd、RS overlap、optimizer tail、transition、device idle。
- 明确旧 trace 只有节点0；跨节点 arrival skew 只能标为推断。
- 给所有新增指标写公式，区分 host span、GPU union、kernel residency 和关键路径。

交付：

```text
analysis/block_hcp_trace_20260919/
  analyze.py
  metrics.json
  REPORT.md
```

### P1：显式 block IR，行为必须不变

建议新增：

- `megatron/core/attention_execution_plan.py`：纯函数和 frozen dataclass。
- `compile_execution_blocks(pattern, family_policy)`：把 per-layer pattern 编译成最大连续 spans。

修改：

- `transformer_block.py:364-410`：初始化时编译并验证 block plan。
- `transformer_block.py:827-894`：按 block span 执行，不再每层判断是否换 view。
- 保留当前 converter、route、collective 次序和 group。

必须测试：

- `[GDN,GDN,GDN,GQA]*4` 精确编译为8个 span。
- current S 与 block-plan 版 output、input grad、param grad、optimizer update 一致。
- forward/backward transition 次数、peer bytes、collective sequence 完全一致。
- full recompute 和无 recompute 都通过。

Go 条件：无 profiler 完整训练性能不得回退超过1%；否则先修等价重构，不进入 P2。

### P2：把 CellPlan 提升为唯一长期逻辑归属

修改：

- scheduler 输出 `sample_ids + cell_ids + cell_cp`。
- `PackedSeqParams` 或明确的 batch sidecar 携带 immutable plan identity。
- `plan_from_packed_seq_params()` 优先消费预编译 CellPlan，禁止每个 block 自行分配。
- trace 加 `plan_id/cell_id/block_id/execution_view`。

相关入口：

- `megatron/core/datasets/data_schedule.py:146-218,335-365,838-851`。
- `megatron/core/datasets/data_schedule_utils.py:358-539`。
- `megatron/core/packed_sequence_cp.py:82-223`。

首版不改变物理通信量；其目标是建立正确、可扩展、可审计的归属基础。若只完成元数据显式化，不得声称性能收益。

### P3：flat RS16 调度因果实验

先完成全延迟 control，再决定是否实现 GQA guard。保持：

- 相同 state mesh；
- 相同 bucket size；
- 相同 reduce op 和 scaling；
- 相同 optimizer shard；
- 相同样本与 block plan。

### P4：HFSDP exact hierarchy oracle

先使用现有实现，不改 collective：

1. 关闭复杂 prefetch/overlap，过正确性和 checkpoint。
2. 默认 FSDP unit 跑一组。
3. 再评估 unit 对齐自然 superblock，而不是逐层 state ownership。
4. 逐项打开 AG prefetch、grad overlap、outer defer。

必须增加一条16-rank确定性代数测试：每个rank用可识别值填充梯度，对比 HFSDP 每rank的1/16 shard与一次 parent RS16 的结果；比较时应用现有 `(outer,inner)→(inner,outer)` logical-rank 映射，禁止直接按物理rank下标比较。checkpoint只接受 fully-reshardable、fully-sharded-model-space 或 `fsdp_dtensor` 等可重分片格式；legacy optimizer-state格式首版 fail-fast。

### P5：可选 GQA operator-local view

只实现一个 motif：

```text
GQA → 3×GDN → GQA
```

覆盖 forward、backward、recompute、残差和 MLP。必须分别记录：

- 完整 hidden/residual transition bytes；
- Q/K/V/O staging bytes；
- 总跨 cell bytes；
- exposed transition span；
- GQA critical path；
- peak memory。

若只是把同量通信搬到 GQA 内部，判定失败。

## 7. 最小实验矩阵

用户明确排除 `CP8DP2` 对 `CP16(Virtual-DP2,CP8)` 的显然对比。本计划不运行它。

固定 parent CP16、同一逻辑 batch、同一模型、同一 loss、同一 optimizer 和同一硬件：

| ID | Activation / ownership | Grad dispatch | State plane | 目的 |
|---|---|---|---|---|
| B0 | UC：全部 parent CP16，连续 GDN 仅保持 parent contiguous | eager | flat AG16/RS16 | parent control |
| B1 | 当前 S：GDN block CP8、GQA block CP16，block remesh | eager | flat AG16/RS16 | block异构净效应 |
| B2 | 与 B1 完全相同 | all-deferred-at-bwd-end | flat AG16/RS16 | 最低成本 contention 因果 control |
| B3 | block plan + logical persistent CellPlan | eager | flat AG16/RS16 | 等价重构与归属显式化；预期不靠此提速 |
| B4 | 与 B3 相同 | GQA block-safe guard | flat AG16/RS16 | optimizer-aware调度 |
| B5 | 与 B3 相同 | eager，再单独测 outer defer | HFSDP RS8→RS2 / AG2→AG8 | exact hierarchy oracle |
| B6 | 可选 operator-local borrowed view | guard | 先 flat，后 HFSDP | 物理persistent原型 |

严格归因：

- `B1−B0`：block级异构 CP 的真实净效应。
- `B2−B1`：完全延迟 parent RS 是否只是移动 tail。
- `B3−B1`：重构必须接近0；有大差值先查实现不等价。
- `B4−B3`：选择性 GQA guard。
- `B5−B3`：HFSDP 整体效应；不能写成纯 hierarchy 效应。
- `B6−B4`：operator-local ownership dataflow 的净效应。

额外负控：

- B0 + all-deferred：判断收益是否是通用 FSDP/DDP scheduling，而非 HCP 特有。
- 普通 `RS8+AR2 duplicated-state`：只作拓扑诊断，不进入最终方案。

## 8. 工作负载阶梯

### 快速机制验证

- 8层：`[GDN,GDN,GDN,GQA]×2`。
- 2×64K，总128K，parent CP16，两个 node-local CP8 cell。
- 先做3组交替配对，10步 warmup、20步 measurement。
- profile 必须在编译和 communicator 初始化后，只采1–2个稳态 step。

### 端到端验证

- 24层：`[GDN,GDN,GDN,GQA]×6`。
- 数据形态：
  1. 64K+64K：等长 fast path；
  2. 60K+68K：不均衡压力，必须使用通用 route，不得套等长 fast path；
  3. 4×32K：相同总 token、更低 GQA 计算密度；
  4. 8×16K：进一步暴露 GQA 通信不可隐藏区间。
- 这些 case 不改变 GDN:GQA 比例，不改变有效 tokens 或 loss 语义。

### 不适用/fallback

- 单条128K：双 cell persistent 路径禁用；回退 parent CP16。
- cell 数不能被独立序列合法填满：回退 B0/B1 合法路径，不复制样本凑数。
- 单节点：hierarchy 可能无收益，planner 应选择 flat state plane。

## 9. 正确性门禁

任一失败立即停止性能实验：

- sample ID、token ID、position、cu_seqlens、padding、loss mask、owner cell 精确一致。
- route round-trip 与 adjoint：`<T(x),y> == <x,T*(y)>`。
- FP32 debug `max_abs ≤ 1e-5`。
- BF16误差不超过两次 UC 自重复噪声的2倍。
- gradient cosine `≥0.999`。
- optimizer step 后重建 full parameters 的 relative L2 `≤1e-3`。
- global valid-token count只统计一次 logical token。
- same-topology save/reload 后下一步 loss、grad norm、参数一致。
- HFSDP 还必须从 fully-reshardable checkpoint 加载；legacy optimizer-state 格式首版不支持时 fail-fast。
- 人为延迟任一 rank 时不死锁，collective ledger 在所有 rank 一致。

缩放规则：

```text
local loss numerator = SUM(valid token losses)
global numerator/count = parent state domain reduction
gradient collective = SUM
normalization = exactly once by global valid-token count
```

禁止 cell-local mean 后再平均，也禁止 inner/outer 两级重复除 world size。

## 10. 性能指标与判定

每个 run 必须记录：

- end-to-end step mean/median/p90/p95；
- input tokens/s、effective TFLOPs/s/GPU；
- GDN/GQA block fwd/bwd max-rank span；
- 第一个与后续 GQA backward 分别统计；
- transition pack/collective/unpack；
- bucket ready→dispatch→NCCL start/end→drain；
- `RS∩GQA` 和 `RS∩GDN` 的时间并集；
- optimizer tail；
- rank arrival spread；
- kernel residency与关键路径分开；
- all-rank max allocated/reserved；
- parent/cross-node bytes、cell-local bytes。

多节点 profile 必须同时采集两节点全部 rank。只有节点0时，跨节点到达偏斜只能标为假设。

### 快速筛选

- 3组交替配对，方向必须3/3一致。
- 收益必须超过 `max(1%, 2×paired_MAD/median)`。
- 不满足则不扩大实现。

### 正式晋级

- 至少5组独立交替配对；warmup≥10，measurement≥30。
- run级 bootstrap 95% CI 下界大于0。
- median step收益≥2%才继续扩展。
- ≥5%才登记为强端到端收益。

### GQA guard Go/No-Go

- `RS∩GQA` interval union 减少≥50%。
- max-rank GQA backward span 降低≥5%。
- optimizer tail 增量小于 GQA 关键路径节省。
- 总 step 不允许回退>1%。

若 overlap 下降但 step 不变，只能说明通信被搬走，不能认定成功。

### HFSDP hierarchy Go/No-Go

- 先用真实 bucket 尺寸 microbenchmark：flat、hierarchical、与 GQA P2P 并发、arrival skew 四类。
- 跨节点 exposed time 至少降低10%，才进入完整训练。
- B5 相对相应 flat state baseline median收益≥2%。
- peak allocated 增幅≤5%。
- 参数 AG/RS 次数、shard owner 和 checkpoint 全部可审计。

### Operator-local borrowed view Go/No-Go

- 完整 hidden/residual transition bytes 或 exposed span降低≥50%。
- Q/K/V/O staging单独核算。
- 总跨 cell bytes或GQA关键路径不能等量回升。
- peak allocated增幅≤5%。

## 11. Agent 任务分解与依赖

### A0：只读基线冻结

允许：读取代码、trace、日志，生成 repo 内 manifest/report。

交付：commit/diff/env/topology/trace hash、当前功能表、已知限制。

完成条件：服务器源码与本地参考提交的差异已明确，不能只记录短 SHA。

### A1：block compiler 与纯单元测试

依赖：A0。

允许修改：planner 新文件、对应 unit tests、最小 config plumbing。

禁止：改 collective、kernel、loss、optimizer、训练 schedule。

完成条件：pattern compiler、非法 policy、PP stage slice、未来 family registry 测试通过。

### A2：block executor 等价重构

依赖：A1。

允许修改：`TransformerBlock` packed split loop 与标签。

禁止：更换 route、group、bucket或buffer。

完成条件：B1 与旧 S 的数值、transition count/bytes、collective order一致；无 profiler性能回退≤1%。

### A3：CellPlan/data identity

依赖：A2。

允许修改：packing scheduler、PackedSeqParams sidecar、plan cache和trace labels。

禁止：物理 persistent GQA、动态建组、跨 step cache sample tensors。

完成条件：同ID可复用、异ID不误复用、recompute一致、60K+68K合法fallback。

### A4：flat RS16 deferred control

依赖：A2；可与A3部分并行，但不得混在同一性能commit。

允许修改：grad ready queue、独立 dispatch hold/drain 和实验 flag；最后microbatch仍须正常学习ready counts。

禁止：改变bucketization、group、reduce op、shard mapping。

完成条件：2/4-rank toy 中 eager/deferred 的最终 main-grad 一致；两microbatch只同步一次；unused/conditional parameter遵守golden-ready逻辑；100-step无hang；pending queue可审计；全部路径最终drain。

### A5：GQA block-safe guard

依赖：A4 的 end-to-end Go。

完成条件：满足第10节 guard 门槛，否则标记 `NO_GO_GQA_GUARD` 并停止复杂化。

### A6：HFSDP oracle

依赖：A2正确，A0确认环境支持。

允许：使用现有 Megatron-FSDP配置、补正确性和checkpoint测试；先不改collective。

禁止：把普通 multi-instance DDP 结果写成 exact hierarchy。

完成条件：RS8→RS2/AG2→AG8 trace、logical-rank重排后的唯一1/16 shard、16-rank代数测试、post-step/checkpoint parity，以及60K+68K下per-token normalization正确。

### A7：outer-only defer

依赖：A6正确且有效、outer RS仍有冲突。

禁止：inner event未完成就启动outer；outer完成前释放helper buffer。

### A8：operator-local borrowed view

依赖：A3、A5；长期任务。

只先实现单 motif，不直接推广全模型。失败也要保留流量、显存和关键路径证据。

## 12. Agent 操作边界

### 必须做

- 修改前读取服务器真实源码、AGENTS.md和相关skills。
- 每个机制单独commit/patch、单独正确性、单独性能对照。
- 保存全部负例、OOM、timeout和无收益结果。
- 使用显式传入的 `ProcessGroupCollection`/process group；遵守 MCore 不在生产模块新增全局 `parallel_state.get_*_group()`读取的规则。
- 只在仓库内写日志、脚本、profile和结果。
- profile run与正式性能run分开。

### 绝对不能做

- 不做 `CP8DP2` 与 `CP16 Virtual-DP2/CP8` 的对比。
- 不把当前 run coalescing 写成新实现成果。
- 不做逐层 CP 搜索或每层动态建组。
- 不改变 3:1 层比例或层顺序制造收益。
- 不把 child group 变成独立 optimizer 副本。
- 不遗漏跨 cell gradient reduction。
- 不重复 loss/gradient scaling。
- 不把普通 `RS8+AR2` 宣称为精确 `RS8→RS2`。
- 不把 HFSDP整体差值宣称为纯 hierarchy 收益。
- 不凭 NCCL kernel residency 和节点0 trace下关键路径结论。
- 不只跑attention microbenchmark后宣称端到端收益。
- 不使用破坏性 git reset、清理他人文件、修改目录外依赖或推送远端。

## 13. 停止条件

出现下列任一情况，停止对应分支而不是继续堆复杂度：

1. block IR 等价重构仍有>1%稳定回退；先修基础实现。
2. all-deferred RS16 没有缩短整步，只增加optimizer tail；停止GQA-aware在线调度。
3. GQA guard减少overlap但不减少step；标记communication displacement。
4. HFSDP oracle的参数AG、full-shard或显存税吞掉层次通信收益；不移植到普通DDP。
5. operator-local view总跨cell字节或GQA critical path不降；停止物理persistent路线。
6. 60K+68K或低packing-density case反复回退；planner必须fallback，不扩大默认启用范围。
7. 任一正确性、save/resume或collective-order测试失败；性能数字全部作废。

## 14. 第一轮建议执行顺序

```text
A0 服务器/trace冻结
→ A1 block compiler
→ A2 等价executor
→ A4 deferred原RS16因果实验
   ├─ No-Go：保留block HCP，停止复杂optimizer调度
   └─ Go：A5 GQA guard
→ A3 CellPlan身份显式化
→ A6 现有HFSDP exact hierarchy oracle
   ├─ No-Go：保留flat state plane
   └─ Go：A7 outer-only defer
→ A8 operator-local borrowed view（长期、单motif）
```

第一轮最有信息量、成本最低的分叉点是 A4：保持原 RS16 数学不变，只改变 dispatch 时机。如果它不能把 GQA 路径节省转化为整步收益，就没有理由马上实现复杂 CommArbiter 或自定义层次 collective。

## 15. 预期交付目录

```text
outputs/block_hcp_state_plane/
  STATUS.md
  source_manifest.json
  experiments.csv
  attempts.jsonl
  correctness/
  traces/
  runs/<run_id>/
  reports/
    block_ir_equivalence.md
    deferred_rs16_causality.md
    gqa_guard.md
    hfsdp_hierarchy_oracle.md
    operator_local_view.md
```

每个报告必须包含：假设、可推翻观测、唯一主要修改、完整训练指标、正确性、显存、原始证据路径、接受/拒绝决定和下一步。不得只保存成功结果。

## 16. 最终研究定位

本方向不应描述为“给每层选择一个最优CP”。更准确的定位是：

> 在 3:1 等重复混合注意力架构中，以自然 block 为执行单元，保持 sample 的逻辑 home-cell 与模型状态 ownership 稳定，仅在 block policy 要求时物化不同 activation view；同时用 optimizer-aware 的固定参数通信平面，避免异构 block 产生的进度偏斜在 GQA critical path 上放大。

这保留了混合模型最有价值的结构异构性，同时把研究重点从局部算子提速提升到 block dataflow、sample identity、collective scheduling 和固定 state ownership 的端到端协同。
