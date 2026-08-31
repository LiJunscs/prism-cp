# 知识库主题：TMF Spatial-Repeat Correctness 初步运行分析与修复建议

> 日期：2026-08-30  
> 分析对象：`agent_run/20260830T081917Z_tmf_repeat_fixed`  
> 状态：2026-08-31 远程复测与对照诊断完成；当前验证边界内 correctness 关闭，允许进入性能 profiling  
> 环境补充：研究者确认执行前已恢复目标 Megatron-LM commit。因此本文不把 Megatron 分支错误列为本次失败根因；但现有 metadata 未记录该 commit，下一轮仍需补齐。

## 1. 结论

当前结果包含三个不同层次的问题，必须分开处理：

1. **bundle/repeat 的主要阻塞项是 input-gradient 判据，不是 output、replica 或 reduced parameter gradient。**
   - bundle/repeat 所有 output 均 PASS；
   - repeat physical replica consistency 全部精确 PASS（max error 为 0）；
   - 日志中的 `grad_buffer PASS` 才是 `overall` 实际采用的参数梯度判据；
   - 后续打印的 `elementwise=FAIL` 只是附加诊断，报告将其误写成 parameter-gradient FAIL。
2. **GA=1 没有显示出新的 repeat 结构性失败。** GA=1 的三组 input-gradient max-abs 恰好都是 GA=4 对应值的 4 倍，完全匹配 loss 中的 `1 / accumulation_steps` 缩放。当前绝对 `atol=0.05` 因而对 GA=1 人为严格 4 倍。
3. **legacy 深栈确有独立的明显偏差。** 它的 parameter-gradient global relative L2 为 0.228、cosine 为 0.974，远大于 bundle/repeat 的约 0.004–0.006 与 0.99998 以上。该问题应做逐层定位，不能归因给 repeat。

因此建议保持 performance gate 关闭，但优先修正/增强 correctness harness，并运行 transition-only 与单层 CP parity 测试；在这些测试失败前，不建议先修改 `_RepeatGatherSequenceForCPTransition` 或 `_RepeatScatterSequenceForCPTransition`。

## 2. 日志证据的正确解读

### 2.1 Bundle/repeat 已通过的部分

| case | output | reduced parameter gradient | replica consistency |
|---|---|---|---|
| bundle roundtrip, GA=4 | PASS | PASS（global rel L2 0.00358，cos 0.999992） | n/a |
| repeat roundtrip, GA=4 | PASS | PASS（0.00402，0.999991） | exact PASS |
| repeat terminal, GA=4 | PASS | PASS（0.00450，0.999988） | exact PASS |
| repeat initial, GA=4 | PASS | PASS（0.00491，0.999985） | exact PASS |
| repeat roundtrip, GA=1 | PASS | PASS（0.00409，0.999990） | exact PASS |
| repeat terminal, GA=1 | PASS | PASS（0.00440，0.999989） | exact PASS |
| repeat initial, GA=1 | PASS | PASS（0.00587，0.999980） | exact PASS |

`compare_states()` 对 `grad_buffer` 使用 `gradient_norm_difference()` 的结果作为 pass/fail。日志中的第二行 `elementwise=FAIL` 不参与该 tensor 的最终判定。最直接的证据是 T2a `initial_replication`：它打印 `elementwise=FAIL`，但 `grad_buffer PASS` 且 `overall PASS`。

因此 `NEEDS_RESEARCHER.md` 中“reduced param grad FAIL”的表述以及 CSV 中仅突出 `param_grad_elementwise_pass=false` 的做法会误导审计。参数梯度目前表现为 BF16 下约 0.4%–0.6% 的全局相对误差，并非 gate 失败来源。

### 2.2 GA=1 的失败是尺度相关的

| repeat case | GA=4 input-grad max-abs | GA=1 input-grad max-abs | 比值 |
|---|---:|---:|---:|
| roundtrip | 0.125 | 0.5 | 4.0 |
| terminal replication | 0.25 | 1.0 | 4.0 |
| initial replication | 0.0625 | 0.25 | 4.0 |

`_temporal_loss()` 将 loss 除以 `accumulation_steps * replication_factor`。T2a 的 GA=4 会把每个 logical microbatch 的 input gradient 缩小为 GA=1 的四分之一，而 `tensor_difference()` 始终使用固定 `atol=0.05`。这导致同一类 BF16 差异在 GA=4 更容易 PASS，在 GA=1 更容易 FAIL。

这也解释了 T2a `initial_replication` PASS、T2b 同 case FAIL，无需假设 GA=1 下 physical leaf 梯度求和突然出错。

此外，旧运行 `20260822T100108Z_tmf_validation_profile` 已出现 bundle fixed input-gradient max-abs 1.0、repeat input-gradient sum max-abs 1.5，并在预先设定的相对判据下 PASS。该历史结果进一步说明固定 0.05 的逐元素绝对门限没有针对当前 BF16 gradient scale 校准。

### 2.3 Bundle 与 repeat 同时失败，降低了 repeat P2P 为根因的可能性

T2a bundle roundtrip 也出现 `out PASS + input_grad FAIL + grad_buffer PASS`。bundle 不使用新 repeat P2P/autograd 类，因此共同失败更像是：

- CP4 与 CP1 FlashAttention backward 的 BF16 数值路径差异；
- input-gradient 判据不具备尺度不变性；
- 或 fixed/mixed kernel parity 本身存在问题。

仅根据现有日志，不能把该模式定位到 repeat gather/scatter 的反向缩放。

### 2.4 Legacy 是另一类问题

Legacy 结果为：

- output max-abs 1.3789；
- parameter-gradient global relative L2 0.2282；
- cosine 0.97396；
- normalized max 0.2241。

当前命令使用 `attn_cp=1, gdn_cp=4`。随机 2A+6G schedule 为：

```text
A-G-G-G-G-G-A-G
```

因此 mixed path 首层 Attention 使用 CP1，而 fixed reference 使用 CP4；其后连续五个 CP4 GDN 可能放大首层 Attention 的小偏差。应逐层采集 canonical activation 与 input-gradient，定位误差首次出现和增长位置。该深栈结果不能用于判断 repeat transition，因为 legacy 使用 redistribution，且 bundle/repeat 的三层显式 case 数值明显更接近。

## 3. 当前测试覆盖还缺少目标方向

性能 Config A 是：

```text
Attention CP4, GDN CP1, fixed CP4
```

本次 T2 使用的却是：

```text
Attention CP1, GDN CP4, fixed CP4
```

显式三层虽然仍覆盖 CP4→CP1→CP4 两个 transition 方向，但它主要比较 **Attention CP4 与 Attention CP1**，没有比较目标配置所需的 **GDN CP4 reference 与 GDN CP1 mixed**。

下一轮应把 `attn_cp=4, gdn_cp=1` 作为正式 gate；当前反向配置保留为额外的 Attention-CP stress test，不能替代目标配置。

## 4. 修复建议

### P0：先修 correctness harness

1. 为 output 和 input gradient 增加与 parameter gradient 同类的尺度不变指标：
   - global relative L2；
   - cosine similarity；
   - normalized max error；
   - max-abs/max-rel 继续保留为诊断，不单独决定 BF16 gradient PASS。
2. 对 input gradient 报告 `accumulation_steps`，并额外报告乘回 accumulation steps 后的 per-sample gradient 误差，使 GA=1 与 GA=4 可比较。
3. 预先固定 norm/cosine 阈值。可先用单层 CP parity 的实测噪声作为基线，再冻结最终阈值；不要直接把 `atol` 调大到 1.0。
4. 修正参数梯度字段：
   - `param_grad_norm_pass`：作为正式 gate；
   - `param_grad_elementwise_pass`：仅作诊断；
   - `overall_pass` 必须与代码实际采用的 norm 判据一致。
5. correctness 模式的 `BenchmarkGradBuffer` 建议使用 FP32 main-grad/all-reduce buffer，减少 BF16 累加与 collective 顺序噪声。性能 benchmark 是否使用 FP32 buffer应单独决定，避免改变性能口径。
6. 记录每个比较 tensor 的 reference norm、actual norm 与 max magnitude，否则 max-abs 无法解释。

### P0：增加 transition-only 代数测试

在不构建 Attention/GDN 的情况下，对 CP4↔CP1 repeat primitive 运行：

1. `gather_repeat -> scatter_mean` forward roundtrip；
2. 任意一致 replicas 的 `scatter_mean -> gather_repeat`；
3. gather 的 adjoint dot-product test；
4. scatter-mean 的 adjoint dot-product test；
5. terminal loss normalization；
6. initially replicated physical leaves 求和恢复 logical input gradient；
7. FP32 与 BF16、多个 seeds、contiguous/non-contiguous inputs。

transition-only FP32 应接近精确。只有该层失败时，才直接修改 repeat P2P/autograd。

### P0：补齐目标配置与逐层隔离

按以下顺序运行，每一步通过后再进入下一步：

1. identity/linear-only CP4↔CP1；
2. 单 Attention：CP4 reference vs CP1；
3. 单 GDN：CP4 reference vs CP1；
4. 三层目标路径：Attention(CP4)→GDN(CP1)→Attention(CP4)；
5. 三层反向 stress path：GDN(CP4)→Attention(CP1)→GDN(CP4)；
6. 8-layer schedule，并逐层记录误差增长。

该矩阵可以区分 transition 错误、FlashAttention CP backward 数值差异、GDN headwise CP 数值差异和深栈误差放大。

### P1：锁定 GDN headwise-only 底层源码

1. 目标 Megatron-LM 版本的 GDN CP 只有 headwise 实现，且 `TransformerConfig` 不存在 `linear_cp_mode` 字段；`build_gdn_te()` 不传入该参数。
2. correctness oracle 验证 headwise CP4 与 CP1 的 parity。
3. metadata 至少记录：
   - 实际导入的 `megatron.__file__`；
   - Megatron-LM commit、status 与 patch；
   - Transformer Engine、FlashAttention/FLA 版本；
   - 三个 wrapper 脚本 SHA。

研究者已确认本轮执行前恢复了目标 Megatron-LM commit，因此该项是证据链补强，不是对本轮环境的否定。

## 5. 建议的最小复测 gate

第一阶段只跑小而可诊断的 correctness，不跑性能：

1. repeat transition-only suite：全部 PASS；
2. target direction `attn_cp=4, gdn_cp=1`：
   - GBS=2/MBS=1/GA=1 repeat 三个 case；
   - GBS=8/MBS=1 对应 full all case；
3. reverse direction `attn_cp=1, gdn_cp=4`：作为诊断，不替代 target gate；
4. 每个 case 输出 output/input/param 的 norm、cosine、normalized max；
5. 两次独立 launch。

判定分支：

- transition-only FAIL：修 `utils.py` repeat P2P/autograd；
- transition-only PASS、单 Attention FAIL：定位 TE/Flash CP kernel parity 或阈值；
- Attention PASS、单 GDN FAIL：定位 GDN headwise CP layout/kernel；
- 单层全部 PASS、深栈 FAIL：做逐层误差增长与训练相关容差评估；
- 上述均 PASS：再开放 T3 performance。

## 6. 本轮状态建议

当前应记录为：

> **Performance No-Go；Repeat correctness inconclusive under the current input-gradient criterion.**  
> Repeat output、replica consistency 与 reduced parameter-gradient norm 均通过；固定绝对阈值导致 GA-dependent input-gradient gate，需要 transition-only 与 norm-based input-gradient 复测。Legacy deep-stack parity 另行定位。

不建议写成“repeat autograd 已证伪”，也不建议直接放宽 atol 后宣布 PASS。

## 7. 2026-08-30 第一轮修改记录

- `utils.py`：按目标 Megatron-LM 的 headwise-only API 构造 GDN，不传入该版本不存在的 `linear_cp_mode`；
- `test_correctness.py`：默认目标配置改为 Attention CP4 / GDN CP1 / fixed CP4；
- `test_correctness.py`：增加不构建模型 kernel 的 repeat transition algebra gate，覆盖 roundtrip、terminal replication、initial replication 及 input-gradient；
- `test_correctness.py`：input gradient 正式 gate 改为 global relative L2，逐元素 allclose 保留为诊断；
- `test_correctness.py`：output 增加 norm/cosine/normalized-max 诊断；
- `test_correctness.py`：参数梯度日志明确标注 `gate=max_param_rel_l2` 与 `elementwise_diagnostic`；
- 下一步：在恢复后的远程 Megatron-LM headwise 环境执行目标方向 T2a/T2b，各两次独立 launch。

## 8. 2026-08-30 transition-only 复测报错与第二轮修复

复测日志的时序是：

1. 第 1–8 行为第一次 repeat gather 的八个 rank P2P 配对；
2. 第 9 行 rank 0 输出 `repeat replica consistency: PASS`；
3. 第 10–17 行为 backward 触发的第二次 repeat gather；
4. 随后 rank 5 在 `dist.batch_isend_irecv()` 报 `Cuda failure 700 'an illegal memory access was encountered'`。

这条 rank 0 PASS 日志将两次 P2P 调用分开，不是 rank 0 越过未完成的同一轮 collective。它证明前向 gather 已完成，故障发生在 `(tensor.sum() / replication_factor).backward()` 的 repeat-scatter adjoint。`sum()` 的上游梯度可以是 expanded/zero-stride view；旧实现将它作为同一个 send buffer 直接交给 NCCL，与 CUDA 700 的触发位置一致。

第二轮修复：

- `_repeat_gather_sequence_p2p()` 在构造 P2P ops 前执行 `seq_local = seq_local.contiguous()`；
- `_repeat_scatter_sequence_p2p()` 同样先物化 contiguous input，覆盖另一个 adjoint 方向；
- 删除临时的全 rank `p2p_ops` 打印，避免 stdout 合并顺序干扰诊断；
- `build_gdn_te()` 保持与目标 Megatron-LM headwise-only API 一致，不传入 `linear_cp_mode`。

这项修复需要远程 8-GPU 重跑 transition-only gate 验证；本地静态检查不能宣称 NCCL 故障已在 GPU 上消失。

## 9. 2026-08-30 contiguous 修复后的最新结果解读

最新日志显示 CUDA 700 已未再出现，repeat transition algebra 的 roundtrip、terminal replication 和 initial replication 全部精确 PASS。完整 launch 中 legacy redistribution、bundle 和 repeat 也都给出 overall PASS。

日志中的大数值需按字段语义区分：

- `reference_l2` / `actual_l2` 是整个巨型 tensor 的 L2 norm，会随元素数量和 sum-loss 尺度增长，不是误差；
- `max_abs` 是未归一化的单元素最大绝对差，参数梯度本身较大时，`1024` 仍可能只是约 1% 的局部尺度；
- `max_rel` 的实现是 `abs_error / max(abs(reference), atol)`，它会被 reference 接近零的元素放大，因此 `2560` 不等于整体梯度相差 2560 倍；
- input-gradient 的正式 gate 是 `global_rel_l2 <= rtol`；
- parameter-gradient 的正式 gate 是每个参数 tensor 的 relative L2，再取 `max_param_rel_l2 <= rtol`；
- `elementwise_diagnostic=FAIL` 是诊断告警，不参与 input/parameter-gradient 的当前 overall gate。

本次所有正式归一化梯度误差都低于 1%：

| Case | input global relative L2 | worst parameter relative L2 |
| --- | ---: | ---: |
| redistribution | — | 0.6437% |
| bundle roundtrip | 0.2849% | 0.5927% |
| repeat roundtrip | 0.2820% | 0.6956% |
| repeat terminal replication | 0.0086% | 0.6232% |
| repeat initial replication | 0.0605% | 0.6685% |

当前 `rtol=5e-2` 对 norm gate 表示 5% 阈值，所以这些值通过与代码判据一致。同时，结果只能称为 **norm-based BF16 parity PASS**；由于 elementwise 诊断仍失败，不能声称 bitwise 或 elementwise parity。参数梯度缓冲区当前使用 BF16，不同 CP 路径的累加/collective 顺序会放大这类量化误差。

建议在最终冻结阈值前补两项诊断：统计 elementwise 超限元素比例及最大误差位置的 reference/actual 值；用 FP32 correctness grad buffer 重跑一次，区分 BF16 累加噪声与 transition 语义误差。

## 10. 2026-08-30 严格 correctness gate 修改

第 9 节的 PASS 是旧 norm-only gate 下的 baseline，不再作为最终 correctness 结论。新版修改如下：

- output 的默认逐元素阈值从 `atol=rtol=5e-2` 收紧为 `1e-2`；
- correctness 参数梯度缓冲区默认使用 FP32 累加和 all-reduce，性能 benchmark 的默认 dtype 不变；
- input-gradient 正式 gate 改为同时满足：
  - global relative L2 `<=5e-3`；
  - cosine `>=0.9999`；
  - normalized max `<=1e-2`；
  - max scaled error `<=10`，防止极少量巨大局部误差被 L2 稀释；
  - 逐元素超限比例 `<=1e-4`；
  - shape 一致且 non-finite 数量为 0。
- parameter-gradient 正式 gate 改为同时满足：
  - 最差单参数 relative L2 `<=1e-2`；
  - global relative L2 `<=1e-2`；
  - cosine `>=0.9999`；
  - 最差单参数 normalized max `<=2e-2`；
  - max scaled error `<=10`；
  - 逐元素超限比例 `<=1e-4`；
  - shape 一致且 non-finite 数量为 0。
- 日志新增 `max_scaled_error`、`violation_count`、`violation_fraction`、`nonfinite_count` 和每个 formal check 的 PASS/FAIL；
- 所有 formal metrics 都在 rank 间取 worst-case extrema，overall 仍使用 all-rank MIN 合并。

上述阈值比旧 `rtol=5%` 明显更严，且防止大参数 tensor 的 L2 汇总掩盖稀疏局部异常。新版需从 T2a 开始重跑；旧日志不能代替新 gate 结果。

## 11. 2026-08-31 对照诊断后的最终关闭结论

严格 harness 的 same-layout replay 证明，即使布局、参数和输入均相同，BF16 backward/WGRAD 仍会产生与跨布局比较同量级的逐元素差异。FP32 shadow reduction 能稳定降低 global relative L2，但不能将误差归零，说明误差同时来自本地反向 kernel 的非确定性和 BF16 collective 顺序。grad-buffer audit 同时确认参数绑定正常、hook 每轮触发一次、WORLD all-reduce 每轮执行一次，排除了重复规约和绑定失效。

据此，最终日常回归门控调整为：

- output：保留 `atol=rtol=1e-2` 的 elementwise allclose；
- input-gradient：global relative L2、cosine、normalized max、finite、shape；
- parameter-gradient：max per-parameter/global relative L2、cosine、max per-parameter normalized max、finite、shape；
- max-abs/max-rel 继续报告，但逐元素 gradient fail fraction 与 max-scaled error 不再作为硬门控；
- 删除 same-layout replay、FP32 shadow snapshot、逐参数 hook/norm audit 及其详细日志。

repeat transition algebra 三类路径精确 PASS，模型层 output 与冻结的梯度范数门控均通过。因此当前结论更新为：

> **Benchmark forward/backward parity verified within the frozen BF16 gates; performance profiling may proceed.**

该结论仍不覆盖 optimizer update、长程 loss trajectory、checkpoint/recompute、pipeline parallelism 或未测试的 CP 组合。
