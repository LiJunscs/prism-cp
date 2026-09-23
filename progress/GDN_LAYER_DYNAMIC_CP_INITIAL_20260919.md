# GDN 层异构 Dynamic CP 初版

日期：2026-09-19。实现基于本地 Megatron-LM `de0628bdfeb77a9899b7e3f19c7b1cb8d8182dde` 的工作区；未提交、未运行 GPU 训练。

## 已实现的行为

GQA 沿用 `DefaultDynamicCPScheduler` 为当前 packed microbatch 选择的父 CP 组。进入连续 GDN 层时，将该 pack 内的完整独立序列分配给父组内更多等大小子组；GDN 结束后恢复父组的 token 顺序。连续 GDN 层之间不重复交换。

子组选择从已创建的 CP1、CP2、CP4 等组开始，按 CP 从小到大尝试。要求每个子组至少一个完整序列；按 padded length 从长到短贪心分配，长度相同时按序列序号确定顺序。可选的每卡 token 上限不满足时尝试更大的子组；全部不满足时保留父 CP。单序列父组、父 CP1 也保留原执行方式。

这里增加的是 GDN 的独立序列执行组数量。参数复制、梯度归约、优化器分片仍使用原来的 DP×CP 状态组，不重新创建 DDP 或优化器，也不额外平均一次梯度。现有 Dynamic CP 的 per-token loss 归一化继续使用。

新增配置：

- `--dynamic-context-parallel`：已有的 Megatron 数据侧调度。
- `--gdn-dynamic-context-parallel`：新增 GDN 层内动态子组选择。
- `--min-dynamic-context-parallel-size 1`：初版要求，以复用预创建的 CP1 子组。
- `--packed-sequence-cp-max-tokens N`：可选，限制**拆分后的 GDN 每卡 padded token 数**；不限制保留父组的 fallback。

未设置 GDN token 上限时，按用户要求尽可能增加独立组。它不保证显存最优或运行更快：长短序列分到 CP1 后，最长序列可能成为耗时和显存瓶颈。

## 多序列支持与范围

`MockPackedCPCaseDataset` 已从双序列循环扩展为任意非空长度循环，支持 Dynamic CP，并按最大 DP×CP 大小补齐物理长度；padding 和 prompt 不计入 loss。真实变长数据也可使用相同模型路径，新增功能不依赖 mock recipe。

模型初版沿用已有 packed CP 的限制：TP1/PP1/EP1、dense GDN/GQA、chunkwise GDN、THD、零 dropout、无 FP8/FP4、无 CUDA Graph、无额外 pack 尾部 padding。支持原有每序列物理 padding。支持不重计算和 full/uniform/每次一层的重计算。

初版使用等大小嵌套子组，不跨不同父 pack 借用样本，不拆分单个序列为独立 DP 样本，不使用空子组，也没有新增性能预测器。对于 CP4 中仅三个序列，初版可使用两个 CP2 组，不能组成三个不同大小的执行组。

## 代码位置

Megatron-LM 中修改：

- `megatron/core/packed_sequence_cp.py`：动态选择、元数据缓存、组成员验证、可选决策日志；修正非均衡子组的 `local_tokens_by_rank` 展开顺序。
- `megatron/core/parallel_state.py`、`process_groups_config.py`：将已有动态组通过显式 process group collection 传入。
- `megatron/core/transformer/transformer_block.py`：复用连续 GDN block 执行器，使用每个 microbatch 的真实父组。
- `megatron/core/ssm/gated_delta_net/gdn.py`：按当前输入元数据选择布局，避免把已经连续排列的子组数据再次当作 zigzag 转换。
- `megatron/core/transformer/transformer_config.py`：新开关和兼容性检查。
- `megatron/training/datasets/varlen_dataset.py`：多序列 fixture。
- `tests/unit_tests/test_dynamic_packed_sequence_cp.py`：新增语义测试及完整 block 测试。

本研究目录中新增：`scripts/run_dynamic_gdn_cp_native.sh`、`scripts/preview_dynamic_gdn_cp.py`。旧的双序列启动脚本保留。

## 已验证与未验证

本机使用已有 Python 3.10 / PyTorch 1.13.1，通过隔离加载真实路由模块完成四进程 CPU/Gloo 验证，不替换真实 all-to-all 或其 autograd：

1. CP4/CP2/CP1 父组、多于 GPU 数量的独立序列、不同物理 padding；子组选择、预算回退、样本完整覆盖。
2. 真实通信后的 token 顺序、往返恢复、输入梯度及元数据缓存失效。
3. 隔离加载实际 `_packed_sequence_cp_forward` 方法，配合有跨 token 依赖的参考层：多个不同父组的 forward 同时保留，再反序 backward；对比重计算开/关、参数梯度和 SGD 更新。
4. 多序列 mock 数据循环、物理 padding、loss mask 和确定性。
5. 启动脚本语法、uniform/split 命令预演及 Python 语法检查。

完整 TransformerBlock 构造测试已编写，但因本机完整 MCore 依赖不满足而跳过。实际 GDN/FA/TE CUDA kernel、原生 DDP/分布式优化器和端到端吞吐尚未验证。CPU 参考层的 SGD 对齐不能代替这些验证。

本机测试日志：Megatron-LM 的 `outputs/dynamic_cp_validation/gloo_results.txt`。

服务器建议先运行：

```bash
MCORE_CP_TEST_CPU=1 torchrun --nproc_per_node=4 -m unittest \
  tests.unit_tests.test_dynamic_packed_sequence_cp -v
```

完整 MCore CPU 测试需要兼容该仓库的 Python/PyTorch 依赖。随后再运行下述真实训练对照。

## 可手动运行的场景

以下结果由当前 checkout 的实际 DCP 调度函数和新增 GDN 选择函数进行元数据预演得到，不是实测性能。K 表示 1024 tokens；父组每卡预算为 16K。

| 场景 | GPU / 全局样本数 | 长度循环 | 预演行为 | 用途 |
|---|---|---|---|---|
| A 默认 | 4 / 8 | 24K,16K,8K,8K，重复两轮 | 两个父 pack 为 `[24K,8K]`，GQA CP2 → GDN CP1 | 首个功能/性能对照，观察 24K vs 8K 负载差异 |
| B 多子组 | 4 / 4 | 48K,8K,4K,4K | 一个 GQA CP4 pack → 四个 GDN CP1 组 | 验证多个独立组；长短不均，不能预设有加速 |
| C 预算限制 | 4 / 4 | 同 B，GDN 上限24K | CP1 超预算，选择 CP2；每卡 token 为24K,24K,8K,8K | 验证自动选择较大子组 |
| D 无切换对照 | 4 / 8 | 32K,16K,8K,8K，重复两轮 | GQA 已分成单序列 CP2 和短序列 CP1，GDN不继续拆 | 验证调度覆盖边界和额外规划开销 |

预演文件在 `analysis/dynamic_gdn_cp/preview_*.json`。可修改输入后先预演：

```bash
python scripts/preview_dynamic_gdn_cp.py \
  --megatron-root /path/to/Megatron-LM \
  --world-size 4 --lengths 24576 16384 8192 8192 --repeat 2 \
  --max-tokens-per-rank 16384
```

在本研究目录中运行 A 的原生训练对照；`MEGATRON_ROOT` 指向包含本次修改的服务器 checkout：

```bash
MEGATRON_ROOT=/path/to/Megatron-LM MODE=uniform OUT=outputs/dcp_A_uniform \
  bash scripts/run_dynamic_gdn_cp_native.sh

MEGATRON_ROOT=/path/to/Megatron-LM MODE=split OUT=outputs/dcp_A_split \
  bash scripts/run_dynamic_gdn_cp_native.sh
```

先加 `DRY_RUN=1` 可仅打印命令。两个模式使用完全相同的原生 DCP packing 和输入；只在 split 模式启用 GDN 子组。

运行 B 或 C 时，在两条命令前都增加：

```bash
LENGTHS_JSON='[49152,8192,4096,4096]' GLOBAL_BATCH_SIZE=4
```

C 的 split 命令另外增加 `GDN_MAX_TOKENS_PER_RANK=24576`。改变场景必须使用新的 `OUT`，避免混入已有 checkpoint。

运行 D 时在两条命令前都增加 `LENGTHS_JSON='[32768,16384,8192,8192]'`。更多序列可直接替换 JSON 数组，并让 global batch 包含完整的长度循环。多节点通过 `NNODES/NPROC_PER_NODE/NODE_RANK/MASTER_ADDR` 指定。

## 首轮应该记录什么

先确认 split 日志中的 `[dynamic-gdn-cp]` 行实际出现 `gdn_cp < parent_cp`，否则没有测试到异构切换。日志开关为 `MCORE_DYNAMIC_GDN_CP_LOG=1`，启动脚本默认打开，正式计时可关闭。

对照同一 global step 的 loss、grad norm、参数范数，以及预热后的 iteration time、tokens/s 和各卡峰值显存。若出现数值差异，先复现和定位，不把吞吐差异解释为有效加速。进一步用 NVTX 观察 GDN 时间、GQA时间、边界 all-to-all 和梯度通信。

优先期待收益的是：GDN 有连续多个层、父 pack 确实含多个独立序列、子组负载相对均衡，并且减少 GDN 内部 CP 通信足以抵消边界交换。单序列占主导、DCP 已经用 CP1、或某条长序列下沉到 CP1 后成为瓶颈时，初版可能没有收益，甚至变慢。下一步优化应由这些实测结果决定。
