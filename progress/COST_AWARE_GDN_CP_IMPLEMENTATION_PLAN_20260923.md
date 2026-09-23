# Cost-aware Dynamic GDN CP 实现计划（2026-09-23）

## 结论与执行边界

本轮直接完成两步：

1. **冻结当前实测基线。** 三次独立运行已经足以作为当前工程稳定性门槛，不再以“五次复核”为实现前置条件。冻结文件是 `analysis/cost_aware_gdn_cp_20260923/frozen_baseline.json`。
2. **实现离线 profiler 与 cost-aware child-CP selector。** 第一版只在已经预建的 CP1/2/4/8/16 嵌套组中选择，候选必须命中离线完整训练步 profile 且满足实测显存预算；未覆盖、收益低于余量或有 OOM 记录时回退 parent CP。没有在线试错、每层建组或任意整数 communicator。

这两步解决当前最直接的问题：旧 selector 只选“最小可行 child CP”，把通信域越小误当成整步越快，无法表达 12 层 CP4 OOM、CP8 稳定获益、均匀输入无收益等事实。

## 冻结数据判断

固定环境为 world16、2×8 H800 80GB、hidden 4096、无 activation recompute，正式窗口取 step 11–40 的 run 内中位数。

| 场景 | 对照 | 三次结果 | 冻结判断 |
|---|---|---|---|
| 8 层强偏斜 `[132096,64512,32768,16384,8192,8192]` | DU→DS/CP4 | 吞吐 +14.57%、+20.50%、+18.06%；峰值 41.1→59.0 GiB | 明确速度 winner，但有显存代价 |
| 同场景 | DU→DC/CP8 | -3.03%、+2.66%、+5.61%；峰值 41.1→40.6 GiB | 显存安全，收益较弱且有一次负值 |
| 12 层强偏斜 | DU→DC/CP8 | +3.57%、+5.45%、+5.42%；峰值 59.1→58.2 GiB | 3/3 正收益，作为 12 层首选 |
| 12 层强偏斜 | DS/CP4 | 首迭代约 79 GiB OOM | 明确容量反例，禁止作为可行候选 |
| 12 层均匀 32×8192 | DU→DS/CP1 | +0.33%、-0.31% | 无稳定收益；当前仅 2 次，保守回退即可，不追加运行阻塞实现 |
| 12 层中等偏斜 | DS/CP2 | fused CE 前后 OOM | 下游 loss/临时 buffer 会决定容量，token cap 不是显存模型 |

正确性证据目前包括显示 lm loss 一致和 grad norm 最大差 0.01–0.03；完整参数更新 parity 仍是后续真实 CUDA 验收项，不能由选择器单元测试替代。

## 第一步：不可追加基线

### 交付

- 锁定 JSON 记录每个候选的模型签名、拓扑、长度、parent/child CP、run 级 step 中位数、峰值显存和 OOM。
- `minimum_successful_runs=3`，并显式记录 `additional_repeats_required=false`。
- 只有代码/配置/拓扑改变、新结果与当前方向矛盾，或需要论文级置信区间时才重跑；不能因为没到 5 次自动追加。

### 验收

- 离线 profile builder 拒绝未锁定基线。
- 成功点少于 3 次不会进入可切换候选，但 OOM 点即使只有一次也会作为禁止证据保留。

## 第二步：离线 profiler + cost-aware selector

### 数据流

```text
冻结的完整 step / 显存 / OOM
        ↓ offline compile
按模型与拓扑隔离的 empirical profile
        ↓ runtime CPU metadata lookup
parent + 已预建 child CP 候选
        ↓ geometry coverage + memory gate + speedup margin
child winner 或 parent fallback
```

### Profile 特征与规则

每个 parent/child 候选记录：sequence 数、总 padded token、最大每卡 token、线性负载比、padding 比例、最长序列占比、长度平方集中度、跨 rank 路由比例。真值是完整训练步中位数与 all-rank 峰值显存，不用孤立 kernel 时间拼成 step。

首版使用保守邻近 profile bin，不做小样本全局回归：

- parent 点没有 3 次成功覆盖：直接回退 parent，不使用 child。
- child 点没有覆盖或有 OOM：排除该 child。
- `predicted_peak + reserve > memory_limit`：排除该 child。
- 预测收益低于预登记 margin：选择 parent。
- 多个 child 都可行：选实测完整 step 最小者；相同预测时偏向较大 CP，避免无必要的深度收缩。
- profile 内容哈希进入 microbatch cache key，替换 profile 后不会复用旧决策。

### 代码落点

- `megatron/core/packed_sequence_cp_cost_model.py`：profile schema、覆盖匹配、显存/收益门控、确定性决策。
- `megatron/core/packed_sequence_cp.py`：枚举所有几何可行的预建 child，生成候选特征；未配置 profile 时保持原 smallest-feasible 行为。
- `megatron/core/transformer/transformer_config.py`：新增 `--packed-sequence-cp-cost-profile`。
- `megatron/core/transformer/transformer_block.py`：构造时加载一次 profile，forward 只做 CPU 查表与已有 plan 缓存。
- `tools/build_packed_sequence_cp_cost_profile.py`：从冻结 JSON 编译模型/拓扑专属 profile。

### 启用方式

```bash
python tools/build_packed_sequence_cp_cost_profile.py \
  --input /path/to/frozen_baseline.json \
  --output /path/to/h800_12l_profile.json \
  --profile-id h800-12l-v1 \
  --model-signature gdn-gqa-3to1-h4096-12l-no-recompute-bf16 \
  --topology-signature h800-80gb-2x8-world16 \
  --minimum-samples 3 \
  --min-predicted-speedup 0.02 \
  --memory-limit-mib 81920 \
  --memory-reserve-mib 2048
```

训练端在原有 `--gdn-dynamic-context-parallel` 基础上增加：

```bash
--packed-sequence-cp-cost-profile /path/to/h800_12l_profile.json
```

## 后续优化路径

### CP32/CP64 覆盖扩展（2026-09-23）

首版的 CP1/2/4/8/16 范围是已有 world16 实验的**实测覆盖**，不是执行器或预建嵌套组的上限。当前扩展允许 parent CP32、CP64 和其已预建的 2 的幂 child；profile 仍按精确 parent CP、模型签名和拓扑分开编译，绝不把 CP16 的步时或显存外推到 CP32/64。

- 训练时先确认 parent profile 有足够成功运行，再检查该 parent 下哪些 child 有足够成功样本。无覆盖直接保留 parent，不生成所有候选的 all-to-all 路由。
- 对有覆盖的 child，先用 CPU 序列长度和预建组几何计算最大每卡 token、负载比及跨 rank 路由比例；仅给最终 winner 构建完整路由。最长序列的 `padded_length / child_cp` 已超过 token cap 时提前排除。
- 接受特征范围内的任一 OOM 记录否决该 child，即使相邻成功点更近。CP32/64 的容量边界须由完整 step 峰值显存验证。
- 冻结新基线时对 CP32、CP64 分别测单条超长序列、双长序列、强偏斜多序列、近均匀多序列和接近每卡 token cap 的 pack；每个 parent 保留原生 parent CP 对照，对可行 child 记录至少 3 次独立完整训练步和 all-rank 峰值显存，对 OOM 保留单次禁止证据。跨节点布局分别编译 topology profile。
- CPU 单测验证 CP32/64 的轻量特征与实际路由一致、带 profile 可选择 child、无 parent 覆盖回退。真实多机运行仍需验证 NCCL collective 顺序、数值 parity、峰值显存和端到端收益。

这一步扩展的是固定 parent pack 内的 child 调度。极端输入若在 parent 打包阶段已出现严重跨 DP 负载偏斜，需要下面第三步的 parent/child 联合调度，不能仅凭 child 切换消除。

### 第三步：parent packing 与 child assignment 联合调度

当前只在固定 parent pack 内选 child。下一步把 parent scheduler 的二次 GQA 工作量、GDN/FFN 线性 token 工作量、显存和拓扑纳入同一小候选搜索，同时保留原生 parent 方案。优先使用 beam search/动态规划而非大而慢的通用 MILP，并以 holdout regret 验证。该方向对应 [FlexSP](https://arxiv.org/abs/2412.01523) 的成本驱动异构 SP、[Hydraulis](https://arxiv.org/abs/2412.07894) 的数据分配与并行策略联合优化，以及最新 [NSP](https://arxiv.org/abs/2609.22755) 的嵌套组与树形路由；只吸收适用于当前 GDN/GQA 执行域的部分。

### 第四步：通信计算掩盖与空闲网卡利用

先用全 rank timeline 确定 parent↔child route、GQA CP、梯度 RS/参数 AG 的 ready/finish 窗口，再实施：

- 将连续 GDN block 的父子布局转换切成可流水的 chunk，尝试让 intra-node route 与 GDN 计算重叠。
- 区分 NVLink 域与 IB/RoCE 域；child8 在节点内执行时，若跨节点 NIC 空闲，可预取下一 parent pack 的只读 metadata/允许提前的通信，但不得改变 collective 顺序或跨 microbatch 梯度语义。
- 只在依赖图证明无环后加入固定窗口的通信优先级/dispatch guard；已运行的 NCCL 不假设可抢占。
- 用“关键路径缩短量”而非通信总量下降作为验收，避免 DCP 已观察到的“通信变少但重叠变差”。

### 第五步：下游容量与 loss 路径

中等偏斜 CP2 在 fused CE 额外申请 2 GiB 时 OOM，说明 selector 必须考虑 GDN 之后的峰值。下一阶段记录 block exit、logits、CE、backward 各阶段峰值，优先减少大 logits/临时 float32 buffer 或采用兼容的 chunked/linear CE，再重新开放被容量门控的候选。

### 第六步：更细 CP 与分层通信

只有 CP1/2/4/8/16 的 profile 已稳定且候选间存在明显空档时，才增加任意整数/非幂次 communicator。届时需要初始化期统一建组、拓扑合法性和 collective 顺序证明；本轮不扩大通信组生命周期风险。

## 验收矩阵

- 单元测试：legacy selector 不变；成本模型能因显存拒绝更快的 CP1、选择 CP2；收益低于 margin 或 profile 未覆盖时回退 parent。
- 离线工具：8 层强偏斜产出 parent/CP4/CP8 三个成功点；12 层产出 parent/CP8 成功点并保留 CP4 OOM；两次均匀和一次 moderate parent 不进入成功 profile。
- 真实训练：先 dry-run 打印 `selector=profile` 与 reason，再做数值 parity；不要求为“补足 5 次”重复旧基线。
- 性能：只有代码或 profile 改变后的新候选需要 3 次独立配对；未覆盖输入的 parent fallback 管理开销必须低于冻结噪声阈值。
