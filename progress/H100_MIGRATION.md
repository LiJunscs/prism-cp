# H100 迁移与 Git 保留范围

本次只精简与当前 idea 有关的内容。用户已授权按明确清单分别创建两个仓库的本地迁移提交，不执行 push。GPU 仍按用户要求暂停。旧 H800 实验移作证据；H100 使用新的输出目录和同硬件对照。

配套 Megatron-LM 版本为 `ce909a8ca5d9e1d7c1ada33e7ec6d1adc7e08312`（分支 `prism-cp`）；prism-cp 的本次迁移提交位于 `main`。新集群应一起迁移这两个版本，不能只复制 prism 的调度器。

## 保留内容

新集群 Codex 从根目录 `AGENTS.md` 进入 [CODEX_HANDOFF.md](CODEX_HANDOFF.md)。必须保留 `bench_data/` 的节点内 CP1/2/4/8、固定 batch 和显存测量表，以及 `analysis/` 的两份 GDN/CSA/HCA 扩展性 CSV。上轮误删的这些小表已恢复，99 条拟合观测与原始 CSV 一致；二维模型、求解器、执行器和关联数据均纳入明确路径清单。

prism-cp 保留当前原生 N/二维执行依赖、独立二维求解、固定/可变容量分析、相关 CPU 测试、三个研究文档和必要实验证据。历史逐 rank 更新合并为 `steps_by_rank.json.gz`，保留全部字段，可由 `results_io.py` 复核。1888 条记录保留；224Ki 的时间、显存和训练合同与旧摘要逐项一致，5 份 N 原函数计划与 4 份二维布局核对通过。

删除约 909MiB 旧产物，包括 profiling sqlite、大型重复 replay/config、成功日志、快照、旧 sweep 和过时原型。当前训练路径退休了 affinity、旧 reverse、旧 online/arch/runtime-capacity 实验分支；当前 idea 的在线 N/二维选择仍待实现。旧场景生成器已被显式长度 manifest 替代。

Megatron-LM 当前 10 个变更文件都服务于保留路径：

- `megatron/core/mixed_context_parallel.py`：混合 CP 的 GQA/GDN、tokenwise 合并和连续 GDN 执行。
- `megatron/core/datasets/data_schedule.py`、`megatron/core/packed_seq_params.py`：混合批元数据及组批接入。
- `megatron/core/transformer/attention.py`、`megatron/core/ssm/gated_delta_net/gdn.py`、`megatron/core/extensions/transformer_engine.py`：对应算子分派。
- `megatron/core/context_parallel_layout/utils.py`、`megatron/core/transformer/transformer_block.py`、`megatron/core/transformer/transformer_config.py`：动态 parent CP 与连续 GDN 布局。
- `tests/unit_tests/test_packed_sequence_cp.py`：对应配置合同。

本次没有改写 Megatron 计算内核，也没有为减小仓库而删除上游核心代码。旧 child-CP 等已提交模块仍有核心引用，不把“本轮不用”当作“可以直接删除”。

## 提交范围

同目录 `PRISM_GIT_PATHS.txt`、`MEGATRON_GIT_PATHS.txt` 是本次整理的明确路径清单：prism 包含有效源码/证据以及已退休跟踪文件的删除；Megatron 只包含上述 10 项。它们只列路径，不执行暂存。提交前按清单核对 diff，避免 `git add -A` 或把工作区全部变更带入。

若决定暂存，分别在对应仓库使用：

```bash
# prism-cp
 git add --pathspec-from-file=progress/PRISM_GIT_PATHS.txt
# Megatron-LM
 git add --pathspec-from-file=../prism-cp/progress/MEGATRON_GIT_PATHS.txt
```

清单按本次迁移范围冻结，新结果不自动加入。Megatron 的上游 PR 要求 `-s -S`；本次为本地迁移提交，使用现有 Git 身份并保留 `Signed-off-by`，不创建上游 PR。

## 新集群路径与执行

两个仓库放在同一父目录；不相邻时显式设置 `MEGATRON_ROOT`。选择 H100 上已有的 Python 环境，入口不再写死旧 flagscale 路径、旧 IP 或 SSH 10710。所有节点需要相同源码路径和共享输出目录；SSH 端口默认为 22，可覆盖。

```bash
export PYTHON=/path/to/existing/environment/bin/python
export MEGATRON_ROOT=/shared/repos/Megatron-LM
cd /shared/repos/prism-cp

# 只做 CPU 接手检查：原始数据 → 模型 → 独立求解 → replay → N 原函数。
"$PYTHON" experiments/sequence_dynamic_cp/handoff_check.py

# 只做 CPU 准备：从保留输入/计划重建三输入八项矩阵。
"$PYTHON" experiments/sequence_dynamic_cp/random_batch_capacity_gpu.py prepare \
  --output /shared/experiment-output/h100_capacity_r1
```

用户明确恢复 GPU 后，逐项启动，地址来自新集群：

```bash
"$PYTHON" experiments/sequence_dynamic_cp/random_batch_capacity_gpu.py run L224_S4_N_K14 \
  --output /shared/experiment-output/h100_capacity_r1 \
  --hosts node0 node1 --master-addr node0 --ssh-port 22
```

每次只启动明确指定的 arm，不自动重试。共享 shell recipe 只保留 DUPC；二维由 joint hook 接管。原生与二维都保持完整训练和实际显存保护。

准备矩阵沿用 H800 的候选容量与计划，适合在新硬件建立初始对照；K24 最优和 H800 模型排序不直接视为 H100 结论。新的硬件/拓扑模型写入独立配置，保留旧 `joint_16gpu_measured.json` 解释 H800 结果。H100 上不混用旧 H800 基线时间计算收益。

## 本次检查

2026-10-16 的 `handoff_check.py` 已通过，提交前再次复核通过：73 个关键 prism 文件和 10 个 Megatron 文件存在；99 条拟合观测与原表一致，20 条早期扩展性记录完整；三个 response profile 可加载，224Ki 计划时间/显存与缓存一致；小规模独立求解及 replay 通过，5 份 N 与 4 份 joint 冻结计划复核通过。Git 清单中的 234 个保留文件均未被忽略，58 项为已跟踪旧文件的删除；开始暂存前两仓库 index 均为空。提交仅按清单暂存，66 个保留 Python 文件通过语法解析。

65 项相关 CPU 测试通过（60 项既有检查，5 项现行输入/packet 检查）；移动公共校验函数后另复测 18 项通过。原生计划与布局、压缩结果核对均通过。N 和 joint 两个入口完成 shell dry run，未执行 torchrun。临时目录成功重建 3 输入、8 项及 15 个 joint replay 步；没有 GPU 启动、环境安装、全文件 hash 或全工作区巡检。整理后完整训练执行路径尚未在 GPU 上复跑，应在恢复实验后用首个授权 arm 验证。
