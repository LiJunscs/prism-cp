# 知识库主题：异构 DP–CP 研究对话续档——梯度正确性、profiling 重构与 TMF 前置验证

> 归档日期：2026-08-16  
> 性质：对 `CONVERSATION_ARCHIVE_20260816.md` 的增量续档；旧文件保留为历史证据。  
> 当前状态：代码已在本地重构并通过静态检查，但本轮新增路径尚未在 GPU/NCCL 环境执行。

## 1. 已冻结的前序实验结论

前序 8×H800、`seq=65536`、global batch 8、2 Attention + 6 GDN 的三 seed 复测支持以下筛选级结论：

| 配置 | Attention CP | GDN CP | fixed CP | 跨 schedule speedup 中位数 |
| ---- | -----------: | -----: | -------: | -------------------------: |
| A    |            4 |      1 |        4 |                    1.2481× |
| B    |            8 |      2 |        8 |                    1.0688× |
| C    |            8 |      4 |        8 |                    1.0230× |

- 三个 layer schedule 中配置排序始终为 `A > B > C`；
- 降低 GDN CP 的幅度足够大时，模块效率收益可以覆盖 DP–CP reshard 成本；
- 已测 mixed 配置的 worst-rank peak allocated 未高于对应 fixed baseline；
- transition 数量与收益负相关只是相关性观察，尚未形成因果归因；
- 这些结果证明静态异构 DP–CP 的性能动机，但没有验证 TMF、收敛性或完整训练系统。

旧 `res_v2.txt` 中更大的 speedup 只保留为历史动机。其运行口径、代码版本和 direction table 状态不完整，不能与修复后的三 seed 结果合并。

## 2. 研究问题的最新边界

### 2.1 DPCP 所处的并行层次

TP、PP 和必要的模型并行首先保证模型可以训练；DP–CP 是在这些基础布局确定后，对长序列模块进行的运行期调优。DP–CP 不应被表述为用 CP 替代 TP，也不应把 CP 转换为 TP 作为主要方案。

真实训练会启用 PP，单个 PP stage 预计约 8 层，并具有规则而非随机的模块 pattern：

- Qwen3.5-Next 类结构可用 `3 GDN + 1 Attention` 周期近似；
- DSv4 后续考虑 `CSA + HCA` pattern，但其 CP 实现仍在开发，不属于当前自动实验范围。

因此，当前随机 2A+6GDN stack 是机制验证工具；后续论文实验需要补充规则 pattern 和 PP stage 语义。

### 2.2 当前近期优先级

在进入 TMF 前，必须依次关闭：

1. 参数梯度同步与 fixed/mixed 数值等价；
2. Attention、GDN 单层准确 profiling；
3. transition forward/backward/round-trip 独立开销；
4. 端到端收益的可解释分解。

TMF 不是当前阶段的默认答案，更不是唯一可能机制。它只针对 nominal microbatch 很小、空间 DP–CP 转换受 batch 整除约束时的 temporal bundling。若普通空间转换已经合法且收益足够，静态 heterogeneous DP–CP 本身就是更简单的方案。

## 3. 本轮梯度同步实现

为了模拟 Megatron Core DDP 的参数梯度路径，benchmark 公共工具中加入连续梯度缓冲：

1. 收集 Attention/GDN stack 的全部可训练参数并按对象去重；
2. 申请元素数等于全部参数元素数之和的连续 `grad_data`；
3. 为每个参数建立对应的 `param.main_grad` view；
4. 保留参数的 `AccumulateGrad` 节点并注册 hook；
5. hook 将 `param.grad` 累加到 `param.main_grad`，随后清空 `param.grad`；
6. 每次 backward 前整块清零；
7. backward 后对整个 buffer 执行一次 WORLD all-reduce，并保持此前除以 world size 的平均语义；
8. plan 结束后移除 hook，避免 fixed/mixed 顺序运行时保留旧模型和显存。

这条路径解决的是 benchmark 中参数梯度规约的实现问题。它尚未经过服务器 GPU 数值验证，也不等价于完整 optimizer、loss scaling 和长期训练语义验证。

## 4. 正确性验证的新入口

正确性逻辑已从性能 benchmark 中拆出到 `heterogeneous_cp/test_correctness.py`：

- fixed/mixed 构建前重置相同模型随机种子；
- 从统一的最大 CP 布局生成输入；
- 两种方案的最终输出都转换回统一 CP 布局；
- 使用相同的 `out.sum()` 目标执行 backward；
- 比较 canonical output 和 WORLD all-reduce 后的连续 `grad_buffer`；
- 分块转 FP32 计算误差，避免为大 buffer 创建完整 FP32 临时副本；
- 汇总所有 rank 的最大绝对误差、最大相对误差和 PASS/FAIL；
- 任一比较失败时脚本以失败状态退出。

默认 BF16 容差为 `atol=rtol=5e-2`。该阈值是当前工程起点，服务器运行必须保存具体误差，不能只保存 PASS 标签，也不能为获得 PASS 自动放宽阈值。

## 5. profiling 代码结构重组

当前 `heterogeneous_cp` 的职责划分为：

| 文件                   | 单一职责                                                                         |
| ---------------------- | -------------------------------------------------------------------------------- |
| `utils.py`             | 公共参数、进程组、模块构造、transition、计时、layer plan、连续 grad buffer       |
| `bench_attn_gdn.py`    | 指定 CP 下 Attention/GDN 单层 profiling                                          |
| `bench_dp_cp.py`       | 固定 world size 与 global batch 下，单层 DP–CP 组合 profiling                    |
| `bench_fixed_mixed.py` | fixed/mixed stack 端到端 profiling，以及每个 transition 的独立 fwd/bwd/both 开销 |
| `test_correctness.py`  | fixed/mixed output 和规约后参数梯度正确性                                        |

四个入口都通过 `utils.parse_benchmark_args()` 接收公共参数，并允许各自或外部调用者传入 `extra_args_provider`。

## 6. 本轮静态验证

已完成：

- 六个相关 Python 文件通过 `py_compile`；
- `git diff --check` 通过；
- 公共参数与各入口 `extra_args_provider` 引用关系完成静态核对；
- 未覆盖用户已有 CSV、实验日志或旧归档。

未完成：

- 当前本地解释器没有 PyTorch，无法执行 CLI import/help；
- 当前环境没有 GPU/NCCL，无法执行 distributed correctness；
- 没有验证新的 transition 独立 profiler 是否在所有合法 CP pair 上完成；
- 没有验证连续 grad buffer 在 TE Attention/GDN 真实 backward 中的数值和显存行为。

因此当前代码状态只能写为：**结构与语法就绪，GPU 数值与性能验证待服务器 Agent 完成。**

## 7. 下一阶段 Go/No-Go

进入 TMF 原型前至少满足：

1. config A/B/C 的小规模 fixed/mixed output 与 grad buffer 均通过；
2. 单层 profiling 的日志明确只包含一个目标层，而不是完整 8-layer stack；
3. `bench_dp_cp.py` 与 `bench_attn_gdn.py` 的 batch/DP 口径在报告中明确区分；
4. `bench_fixed_mixed.py` 每个实际 layout boundary 都输出一条 isolated transition profile；
5. 端到端 benchmark 的梯度 all-reduce 计入 backward，而单层 DP–CP frontier 继续明确标记 `dp_grad_sync=false`；
6. 所有失败保留原始日志和首个错误，不由 Agent 修改 transition 或模型实现。

若正确性失败，先由研究者定位，不进入 TMF。若正确性通过但 transition 成本足以吞噬 A/B 的收益，则需要重新评估静态异构方案和 TMF 的必要性。若正确性与成本均通过，再由研究者实现 temporal packing、checkpoint 和 PP 集成。

## 8. 证据索引

- 历史阶段归档：`heterogeneous_cp/CONVERSATION_ARCHIVE_20260816.md`；
- 三 seed 报告：`heterogeneous_cp/20260815T120648Z_b0226ace2_seed_sweep/`；
- 当前研究笔记：`.codex/heterogeneous_dp_cp_research_notes.md`；
- TMF 重新评估：`E:/.codex/my_knowledge/temporal_mesh_folding_reassessment_and_roadmap.md`；
- 本轮服务器执行计划：`heterogeneous_cp/AGENT_SIMPLE_GPU_TASKS_20260816.md`。
