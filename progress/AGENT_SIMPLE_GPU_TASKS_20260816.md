# 知识库主题：异构 DP–CP 近期服务器 Agent 简单 GPU 任务计划

> 用途：将本文全文交给服务器 Agent 执行。  
> 范围：只运行、记录、解析现有脚本；不设计新算法，不修改通信和模型实现。  
> 目标：验证本轮代码重构、连续梯度缓冲、fixed/mixed 正确性、单层 profiling 和 transition 独立开销。

## 1. Agent 角色与强制边界

你是实验执行 Agent，不是算法开发 Agent。

允许：

- 读取当前 checkout、`AGENTS.md` 和相关 skill；
- 运行静态检查、分布式正确性脚本和 benchmark；
- 在新的运行目录内创建编排脚本、日志、CSV 和 Markdown 报告；
- 修复你自己创建的日志解析脚本；
- 对互相独立的配置在单项失败后继续执行。

禁止：

- 修改 `heterogeneous_cp/utils.py`、`test_correctness.py` 或三个 benchmark；
- 修改 direction table、P2P mapping、autograd、Attention、GDN 或 Megatron Core；
- 放宽 `atol/rtol`、减少 rank、减少指定配置来制造通过；
- 实现 TMF、checkpoint、PP、planner 或 DSv4；
- 删除/覆盖已有实验目录、CSV 和日志；
- commit、push 或创建 PR；
- 杀死非本任务创建的 GPU 进程。

如果必须修改禁止区域才能继续，将原始错误写入 `NEEDS_RESEARCHER.md`，继续其他独立任务。

## 2. 执行前读取顺序

1. `AGENTS.md`；
2. `skills/mcore-testing/SKILL.md`；
3. `heterogeneous_cp/CONVERSATION_ARCHIVE_20260816_V2.md`；
4. `heterogeneous_cp/utils.py`；
5. `heterogeneous_cp/test_correctness.py`；
6. `heterogeneous_cp/bench_attn_gdn.py`；
7. `heterogeneous_cp/bench_dp_cp.py`；
8. `heterogeneous_cp/bench_fixed_mixed.py`。

只使用当前 checkout。记录 commit、branch 和 dirty files，不清理 worktree。

## 3. 运行目录与环境

创建：

```text
heterogeneous_cp/agent_runs/<UTC时间>_<短commit>_post_refactor/
├── metadata/
├── static/
├── correctness/
├── single_layer/
├── dp_cp_frontier/
├── fixed_mixed/
├── commands.tsv
├── summary.csv
├── NEEDS_RESEARCHER.md
└── REPORT.md
```

元数据至少包含：

- commit、branch、`git status --short`、最近 5 个 commit；
- `nvidia-smi`、GPU 数量/型号/显存/driver；
- Python、PyTorch、CUDA、NCCL、Transformer Engine 版本；
- hostname、容器标识；
- 非敏感的 `CUDA*`、`NCCL*`、`NVTE*`、`MEGATRON_BENCH*` 环境变量。

设置：

```bash
export PYTHONPATH="$PWD/heterogeneous_cp:$PWD:${PYTHONPATH:-}"
```

所有 GPU 任务使用 8 张独占 GPU，同一时间只运行一个任务。每条命令记录开始/结束 UTC、退出码、wall time 和日志路径。建议正确性单项超时 20 分钟，benchmark 单项超时 30 分钟。

## 4. Task A：结构与导入冒烟

### A1. 静态检查

```bash
python -m py_compile \
  heterogeneous_cp/utils.py \
  heterogeneous_cp/bench_attn_gdn.py \
  heterogeneous_cp/bench_dp_cp.py \
  heterogeneous_cp/bench_fixed_mixed.py \
  heterogeneous_cp/test_correctness.py

git diff --check -- heterogeneous_cp
```

### A2. 公共模块与 CLI

```bash
python - <<'PY'
import sys
sys.path.insert(0, "heterogeneous_cp")
import utils
PY

python heterogeneous_cp/bench_attn_gdn.py --help
python heterogeneous_cp/bench_dp_cp.py --help
python heterogeneous_cp/bench_fixed_mixed.py --help
python heterogeneous_cp/test_correctness.py --help
```

检查四个入口都包含公共参数 `--seq-length`、模型参数和各自额外参数。任一导入失败都保存 traceback；不要改代码。

### A3. 验收

- 六文件语法通过；
- 四个 CLI help 正常；
- 恰好看到 8 张目标 GPU，否则停止全部 GPU 任务。

## 5. Task B：fixed/mixed 输出与 grad buffer 正确性

固定轻量矩阵：

- `seq_length=4096`；
- global batch 8；
- hidden size 4096；
- 2 Attention + 2 GDN；
- layer seed 1234，对应 `A-G-A-G`，包含三个 layout boundary；
- Flash Attention；
- 默认 `atol=rtol=5e-2`，不得修改。

### B1. Config B：CP8 ↔ CP2

```bash
MEGATRON_BENCH_DEBUG_CP_TABLES=1 timeout 20m \
torchrun --nproc-per-node 8 heterogeneous_cp/test_correctness.py \
  --seq-length 4096 --global-batch-size 8 \
  --num-attention-layers 2 --num-gdn-layers 2 --layer-seed 1234 \
  --attn-cp-size 8 --gdn-cp-size 2 --fixed-cp-size 8 \
  --hidden-size 4096 --attention-backend flash
```

### B2. Config C：CP8 ↔ CP4

```bash
MEGATRON_BENCH_DEBUG_CP_TABLES=1 timeout 20m \
torchrun --nproc-per-node 8 heterogeneous_cp/test_correctness.py \
  --seq-length 4096 --global-batch-size 8 \
  --num-attention-layers 2 --num-gdn-layers 2 --layer-seed 1234 \
  --attn-cp-size 8 --gdn-cp-size 4 --fixed-cp-size 8 \
  --hidden-size 4096 --attention-backend flash
```

### B3. Config A：CP4 ↔ CP1，一般 nested mapping

当前 debug table 校验对 `world_size=8, max_cp=4` 有额外限制，因此该项显式关闭固定表，测试一般 nested-CP 路径：

```bash
MEGATRON_BENCH_DEBUG_CP_TABLES=0 timeout 20m \
torchrun --nproc-per-node 8 heterogeneous_cp/test_correctness.py \
  --seq-length 4096 --global-batch-size 8 \
  --num-attention-layers 2 --num-gdn-layers 2 --layer-seed 1234 \
  --attn-cp-size 4 --gdn-cp-size 1 --fixed-cp-size 4 \
  --hidden-size 4096 --attention-backend flash
```

### B4. 必须提取

每个配置保存：

```text
config,debug_tables,out_status,out_max_abs,out_max_rel,
grad_status,grad_max_abs,grad_max_rel,overall_status,exit_code,log_path
```

状态只能为 `PASS`、`VALUE_MISMATCH`、`GRAD_MISMATCH`、`SHAPE_MISMATCH`、`OOM`、`NCCL_ERROR`、`TIMEOUT`、`CRASH`。

如果 output PASS 但 grad buffer FAIL，优先标记 `GRAD_MISMATCH`；不要自动修改 hook 或 all-reduce。若 config A 失败而 B/C 通过，明确区分一般 mapping 与固定 direction-table 路径。

## 6. Task C：单层 profiling 隔离冒烟

目标：确认 `bench_attn_gdn.py` 只构建并测量一个目标层，没有误跑完整 8-layer stack。

运行四项，均使用独立进程：

```text
module ∈ {attention, gdn}
cp ∈ {2, 8}
seq_length = 8192
batch_size = 1
warmup = 5
iters = 10
```

命令模板：

```bash
timeout 30m torchrun --nproc-per-node 8 heterogeneous_cp/bench_attn_gdn.py \
  --module <attention|gdn> --seq-length 8192 --cp <2|8> \
  --batch-size 1 --hidden-size 4096 --attention-backend flash \
  --warmup 5 --iters 10 \
  --log-csv <run_dir>/single_layer/<唯一文件名>.csv
```

验收：

- 每个 CSV 新文件只有 header + 1 row；
- module、CP、sequence、batch 与命令一致；
- fwd/bwd/both latency、MFU 和 memory 为有限非负数；
- 日志没有 randomized layer schedule、多个 layer 或 transition 输出；
- MFU > 1 标记 `METRIC_SUSPECT`，不删除数据。

这里只验证单层隔离和指标管线，不形成论文结论。

## 7. Task D：固定 global load 的 DP–CP 单层小矩阵

目标：验证重构后的 `bench_dp_cp.py` 参数和 CSV 口径，生成一个低成本 frontier sanity check。

矩阵：

```text
module ∈ {attention, gdn}
cp ∈ {1, 2, 4, 8}
seq_length = 16384
global_batch_size = 8
world_size = 8
warmup = 5
iters = 10
```

共 8 个独立 launch。命令模板：

```bash
timeout 30m torchrun --nproc-per-node 8 heterogeneous_cp/bench_dp_cp.py \
  --module <attention|gdn> --seq-length 16384 --cp <1|2|4|8> \
  --global-batch-size 8 --hidden-size 4096 --attention-backend flash \
  --warmup 5 --iters 10 \
  --csv-path <run_dir>/dp_cp_frontier/<唯一文件名>.csv
```

每行检查：

```text
dp_size = 8 / cp
microbatch = 8 / dp_size
local_seq = 16384 / cp
dp_grad_sync = false
```

将 8 行合并为 `dp_cp_frontier/merged.csv`，失败配置写入 `failed.csv`。该任务没有参数梯度 all-reduce，报告必须保留 `dp_grad_sync=false`，不得与 `bench_fixed_mixed.py` 的 backward 口径混淆。

## 8. Task E：端到端与 isolated transition profiling

使用前序 2 Attention + 6 GDN、seed 1234 schedule：

```text
A-G-G-G-G-G-A-G
```

该 schedule 有 3 个 CP layout boundary。测试三组配置，每组只做 1 次 profiling 冒烟：

| Config | Attention CP | GDN CP | fixed CP | debug tables |
| ------ | -----------: | -----: | -------: | -----------: |
| A      |            4 |      1 |        4 |            0 |
| B      |            8 |      2 |        8 |            1 |
| C      |            8 |      4 |        8 |            1 |

公共参数：`seq_length=16384`、global batch 8、warmup 5、iters 10。

命令模板：

```bash
MEGATRON_BENCH_DEBUG_CP_TABLES=<0|1> timeout 30m \
torchrun --nproc-per-node 8 heterogeneous_cp/bench_fixed_mixed.py \
  --seq-length 16384 --global-batch-size 8 \
  --num-attention-layers 2 --num-gdn-layers 6 --layer-seed 1234 \
  --attn-cp-size <4|8|8> --gdn-cp-size <1|2|4> \
  --fixed-cp-size <4|8|8> --mode both \
  --hidden-size 4096 --attention-backend flash \
  --warmup 5 --iters 10
```

每个成功日志必须提取：

- fixed/mixed 的 fwd、bwd、both latency；
- fixed/mixed tokens/s 和 peak allocated/reserved；
- `fixed_both_ms / mixed_both_ms`；
- mixed plan 打印的 isolated transition 条目数；
- 每条 transition 的 index、src CP、dst CP、fwd/bwd/both latency 和 allocated GiB。

验收：

- 每个 mixed 日志应出现 3 条 isolated transition profile；
- transition latency 和显存为有限非负数；
- fixed plan transition 数为 0；
- end-to-end backward 包含连续参数 grad buffer 的 WORLD all-reduce；
- 不把 end-to-end speedup 简化为 transition latency 之和，因为两种方案的模块 CP 也不同。

## 9. 执行顺序与停止规则

严格顺序：`A → B → C → D → E`。

- Task A 失败：停止 GPU 阶段；
- Task B 数值失败：继续 C/D/E 以获取独立工程证据，但所有性能结果加 `CORRECTNESS_UNVERIFIED`；
- OOM：不重试，不降低配置冒充原配置；
- NCCL transient：清理本任务 PID 后最多重试 1 次；
- timeout：不缩短 sequence/warmup/iters，记录后继续；
- parse 失败：可修复运行目录内的解析器，不重跑已有有效日志；
- 不允许无限重试。

## 10. 最终交付

`summary.csv` 每个 launch 一行：

```text
task,config,module,cp,dp,seq,gbs,batch,debug_tables,status,exit_code,
fwd_ms,bwd_ms,both_ms,tokens_s,peak_alloc_gib,peak_reserved_gib,
out_max_abs,out_max_rel,grad_max_abs,grad_max_rel,log_path
```

`REPORT.md` 必须包含：

1. Run identity；
2. A–E completion matrix；
3. correctness 三配置结果与最大误差；
4. 单层隔离检查；
5. DP–CP 小矩阵及 `dp_grad_sync=false` 边界；
6. 三组端到端结果和 3 条/配置的 transition profiles；
7. 所有失败与重试；
8. `NEEDS_RESEARCHER`；
9. 最多 3 条下一步人工工作。

Agent 最终只交付运行目录，不修改实验代码，不自动提交仓库。

## 11. 成功条件

本轮 Agent 任务完成不要求所有配置 PASS。完成条件是：

- 6 个静态文件和 4 个 CLI 均有记录；
- 3 个正确性配置均有明确状态和最大误差；
- 4 个单层隔离配置均有日志；
- 8 个固定负载 DP–CP 配置均有结果或明确失败；
- 3 个 fixed/mixed 配置均有端到端日志和 transition 条目计数；
- 每个预期项都能在 `summary.csv` 与原始日志之间追溯；
- 没有覆盖旧数据，没有修改核心代码。
