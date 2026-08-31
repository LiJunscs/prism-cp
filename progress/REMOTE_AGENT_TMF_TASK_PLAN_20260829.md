# 知识库主题：TMF Bundle 与 Spatial-Repeat 修复版远程服务器执行计划（审查批准版）

> **状态：Correctness 在当前验证边界内通过，性能阶段开放（2026-08-31）**  
> contiguous 修复后 CUDA 700 未再出现；repeat transition algebra 三类路径精确通过，fixed/mixed 的 output、input-gradient 与 reduced parameter-gradient 均满足冻结的 BF16 范数门控。same-layout replay、FP32 shadow reduction 与 grad-buffer audit 已完成根因排查并从日常回归 harness 删除。T3 性能冒烟是当前下一项。

> 日期：2026-08-23  
> 本地审查日期：2026-08-29  
> 执行对象：远程 8×H800 服务器 agent  
> 目标脚本：`bench_fixed_mixed.py`、`test_correctness.py`、`utils.py`  
> 工作性质：**先正确性验证，再性能采集**；除非任务书明确授权，不修改 benchmark / correctness / utils 代码

## 0. 本地审查结论与远程版本锁

当前工作区修改已通过逐段静态审查，具备进入远程精度验证的条件：

- `repeat` 的 CP 收缩前向执行 spatial replication，其反向对 replica gradient 求和；
- `repeat` 的 CP 扩张前向执行 mean-merge，其反向按 replication factor 分发梯度；
- 末端 replicated layout 使用 loss normalization；初始 replicated layout 将 physical leaf gradients 求和恢复为 logical input gradient；
- correctness 同时比较 canonical output、logical input gradient 和 reduced parameter gradient，并对末端 physical replicas 做一致性检查；
- GA=1 repeat-only case 能排除“借用后续 temporal microbatch”的旧语义；
- `DIRECTION_4_1` 与 `DIRECTION_8_1` 已接入 `set_transition_direction()`；
- 目标 Megatron-LM 版本的 GDN CP 为 headwise-only，wrapper 不传入该版本不存在的 `linear_cp_mode`；
- correctness 在模型层测试前先执行 repeat transition algebra gate；
- repeat P2P 在通信前将 autograd 传入的 expanded/zero-stride tensor 物化为 contiguous buffer；
- input/parameter gradient 使用 relative L2、cosine、per-tensor normalized max、shape 和 non-finite 的组合 gate；逐元素 gradient 误差仅作数值诊断；
- correctness 与性能 benchmark 均保持参数 dtype 的 grad buffer；FP32 shadow reduction 仅用于一次性根因排查，现已删除；
- 三份脚本通过无导入语法编译检查，`git diff --check` 无错误。

本地环境仍没有可用的 PyTorch/CUDA/Megatron 运行栈，但远程 8-GPU launch 已完成当前边界内的数值验证。允许写“benchmark forward/backward parity verified within frozen BF16 norm gates”；不得扩写为完整 training correctness verified。

审查时的版本指纹：

```text
HEAD                                     待研究者提交第一轮修改后填写
utils.py SHA256                          3b6c047bcdf9edaef7e6f11515c8ce06f53e2ed5e4d45b6c05ea791e20aac82c
bench_fixed_mixed.py SHA256              0d63e028caca696da5e29145dd845b177b7fb14f89824e5253020f4bf675249f
test_correctness.py SHA256               fd8d0df4ca7c49a70d8374cb60e2b12059a948e6fa920df440fecd1b80a92b64
```

远程 agent 必须先核对三份脚本 SHA256。任一不一致时，不得直接套用本计划的“已审查代码”结论：保留实际 hash 和 diff，写 `NEEDS_RESEARCHER.md` 并停止 T1 之后的任务，除非研究者明确批准新版本。

本轮验证边界保持不变：它验证当前 benchmark stack 的 forward/backward parity，不验证 optimizer update、长程 loss trajectory、dropout/RNG 等价性、checkpoint/recompute 或 pipeline parallelism。

## 0.1 2026-08-31 correctness 关闭结论与执行状态

最新 8×H800 结果支持在当前测试边界内关闭 correctness 阶段并进入性能 profiling：

- repeat transition algebra 的 roundtrip、terminal replication、initial replication 前向与 input-gradient 均精确 PASS；
- legacy redistribution、bundle 与 repeat 的 canonical output 均满足 elementwise allclose；
- input-gradient 的 global relative L2、cosine 与 normalized max 均在冻结阈值内；
- reduced parameter-gradient 的 per-parameter/global relative L2、cosine 与 normalized max 均在冻结阈值内；
- same-layout replay 本身存在与跨布局同量级的 BF16 backward/WGRAD 非确定性；FP32 shadow collective 能缩小但不能消除该差异；
- grad-buffer audit 已确认参数绑定正常、每轮只有一次 WORLD all-reduce，不存在重复规约；
- 因而逐元素 gradient fail fraction 与 max-scaled error 不再作为硬门控，也不再运行 replay、FP32 shadow snapshot 或逐参数审计日志。

日常 correctness gate 冻结为：

```text
output:
  elementwise allclose(atol=1e-2, rtol=1e-2), finite, shape

input gradient:
  global relative L2 <= 5e-3
  cosine >= 0.9999
  normalized max <= 1e-2
  finite, shape

parameter gradient:
  max per-parameter relative L2 <= 1e-2
  global relative L2 <= 1e-2
  cosine >= 0.9999
  max per-parameter normalized max <= 2e-2
  finite, shape
```

当前任务状态：

| 阶段 | 状态 | 说明 |
| --- | --- | --- |
| T0–T1 | 完成 | 版本、静态检查与目标 API 已核对 |
| T2a | 完成 | 全量 transition/legacy/bundle/repeat 已覆盖 |
| T2b | 完成 | repeat transition-local 语义与 GA=1 结构性风险已排除 |
| T2c | 关闭 | same-layout controls 已量化非确定性基线；后续独立复跑作为回归，不再阻塞性能 |
| T3 | 下一项 | 先执行 Config A 的 4096-token bundle/repeat 冒烟 |
| T4–T8 | 待执行 | 冒烟通过后按原顺序进行长序列、逐模块 profiling 与汇总 |

该结论只覆盖当前 benchmark stack 的 forward/backward parity、CP4↔CP1 目标路径和已运行的 BF16 环境；不覆盖 optimizer update、长程 loss trajectory、checkpoint/recompute、pipeline parallelism 或其他未测试 CP 组合。

---

## 1. 本轮目标

验证修复后的两种异构 CP 执行路径能够在真实 8-GPU H800 环境完成 forward、backward、参数梯度同步，并验证其训练语义，再采集 optimizer-step 级延迟、逻辑吞吐与峰值显存：

1. `bundle`：使用同一 optimizer step 内**不同的 gradient-accumulation microbatch**，在 DP 扩张时通过 temporal folding 提供更多独立 logical samples；
2. `repeat`：使用**transition-local spatial replication**。gradient-accumulation microbatch 保持彼此不同；当 CP 减小、DP 扩张时，仅在 CP transition 处复制当前 logical sample；当 CP 增大、DP 收缩时，对重复 replica 做 mean-merge；
3. 验证 repeat 的 autograd 语义：
   - replication forward 的 backward 必须对 replica gradient 求和；
   - mean-merge forward 的 backward 必须按 replication factor 分发梯度；
   - 若模型最终停在 replicated layout，loss normalization 必须消除最终 replication factor；
   - 若模型从 replicated layout 起始，多个 physical input leaf 的梯度在恢复 logical input gradient 时必须正确求和；
4. 使用 `test_correctness.py` 对 bundle 与 repeat 分别验证：
   - canonical output；
   - logical input gradient；
   - reduced parameter gradient；
   - repeat terminal physical replicas 的一致性；
5. 对比 temporal/spatial-repeat fixed、mixed 与旧 spatial redistribution oracle；
6. 利用现有完整 inline profiler，采集 Attention、GDN、legacy redistribution transition、loss backward、gradient sync 与 residual 的逐模块前向/反向分解。

### 1.1 关键语义变化

旧版 `repeat` 的语义是：

```text
gradient accumulation:
MB0 = A
MB1 = clone(A)
...
```

这会导致 CP transition **之前**的 block 也重复执行，因此不是目标方案。

修复后的 `repeat` 必须是：

```text
logical sample A
    |
    |  high-CP block: 只执行一次 logical copy
    v
CP4 / DP2
    |
    | CP4 -> CP1 repeat transition
    v
CP1 / DP8
A  A  A  A       (每个 logical DP group 内 spatial replicas)
```

即：

- temporal microbatch 仍然是 distinct samples；
- redundancy 只在低 CP / 高 DP 的物理执行区间出现；
- logical batch size 不因 repeat 增大；
- repeat 可以在 **gradient accumulation = 1** 时工作，不能依赖“借未来 microbatch”。

本轮不验证 optimizer update、长程 loss trajectory、checkpoint/recompute 或 PP；不得据此声称完整训练正确性。逐模块 profiler 仍仅用于 `--temporal-strategy none` 的 legacy spatial fixed/mixed 路径；bundle/repeat 多 payload 图尚未接入该单链 profiler。

---

## 2. 执行前硬约束

- 必须使用本轮修复后的三份代码：
  - `utils.py`：包含 `transmition_cp_size(..., strategy="repeat")` 及 repeat P2P autograd；
  - `bench_fixed_mixed.py`：包含 `repeat_logical_cp`、`make_repeat_stack`，repeat 不再 clone temporal input；
  - `test_correctness.py`：包含 `--correctness-strategy {all,transition,legacy,bundle,repeat}`。
- T2 correctness gate 已按 2026-08-31 冻结门控关闭；允许进入 T3/T4 性能任务。任何后续代码改动仍须先重跑对应 correctness 回归。
- 所有正式结果必须绑定 commit、dirty patch/hash、三份目标脚本 SHA256、完整命令、环境和原始日志。
- 不得把 repeat 的 `logical_tokens/s` 理解为“物理执行 token 数/s”。repeat 的低 CP block 会执行冗余 replicas，但 benchmark 中 logical/unique throughput 都按每个 optimizer sample 只计一次。
- 修复版 repeat 中 `unique_tokens/s == logical_tokens/s` 是预期行为；这与旧版 “executed tokens/s > unique tokens/s” 的统计口径不同。
- 必须记录 `final_replication`：
  - fixed max-CP plan 通常为 `1`；
  - mixed plan 若最终停在较小 CP，可能为 `logical_cp / final_cp`。
- `repeat` 的正确性声明只允许建立在 `test_correctness.py` PASS 基础上，不能仅凭 benchmark 无报错。
- 遇到 shape、NCCL、autograd、CUDA 或 correctness FAIL 时保留原始日志，不得静默修复后继续混用结果。
- 当前 inline profiler 的组件时间来自 rank 0 本地 CUDA event；只有 peak allocated/reserved 是 worst-rank 归约值。报告中必须保留该口径，不得称为 worst-rank 模块时间。
- 默认 `MEGATRON_BENCH_DEBUG_CP_TABLES=1` 时，必须确认 `CP1<->CP4` direction table 已实际接入 `set_transition_direction()`；若未接入，停止，不自行修补。

---

## 3. 简单任务计划表

| 阶段                    | 优先级 | Agent 操作                                                         | 验收条件                                                                               | 失败处理                                                          |
| ----------------------- | ------ | ------------------------------------------------------------------ | -------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| T0 版本冻结             | P0     | 记录 commit、`git status`、三脚本 SHA256、GPU/软件版本和拓扑       | 元数据齐全；三份代码均为修复版                                                         | 若代码不是修复版，停止并写 `NEEDS_RESEARCHER.md`                  |
| T1 静态检查             | P0     | `py_compile`、CLI help、关键符号 grep                              | 三脚本编译通过；CLI 与 repeat P2P 符号存在                                             | 失败即停止，不改代码                                              |
| T2a 全量 correctness    | P0     | 8 GPU，seq=4096，CP4↔1，mbs=1，gbs=8，`--correctness-strategy all` | legacy、bundle、repeat 全 PASS；repeat 三个 case 全 PASS                               | 任一 FAIL：停止性能任务，保留完整日志                             |
| T2b Repeat 最小语义测试 | P0     | 8 GPU，seq=4096，logical CP4，CP4→1，mbs=1，gbs=2，repeat-only     | GA=1 条件下 repeat correctness PASS                                                    | FAIL 说明 repeat 仍依赖 temporal width 或 P2P/autograd 有误；停止 |
| T2c Correctness 稳定性  | P0     | T2a/T2b 各再独立 launch 1 次                                       | 两次结果均 PASS；无随机 NCCL/数值失败                                                  | 若仅偶发失败，标记 flaky，禁止进入正式性能结论                    |
| T3 Config A 冒烟        | P0     | 8 GPU，seq=4096，CP4→1，mbs=1，gbs=8，bundle+repeat                | 四条 temporal fixed/mixed 路径完成；GA=4；repeat 日志显示 spatial replication 语义     | 同命令最多重试1次；确定性失败停止                                 |
| T4 Config A 主测        | P0     | seq=32768、65536；先各1次筛选，65536 成功后3次独立 launch          | 每次完整结束；记录 E2E、logical/unique tokens/s、final replication、allocated/reserved | OOM 记录为有效边界，不降配置伪装成功                              |
| T5 Config B 次测        | P1     | seq=65536，CP8→2，mbs=1，gbs=8；先1次                              | GA=8；bundle/repeat fixed/mixed 均完成                                                 | 若对应 mixed/fixed ≤1.03×，不追加重复                             |
| T6 逐模块 profiling     | P1     | 对 A、B 的 `temporal-strategy=none` spatial fixed/mixed 逐模块计时 | layer/transition fwd+bwd mean/std 完整；另含 loss、grad sync、residual                 | profiler 事件缺失则该 launch 作废                                 |
| T7 Spatial oracle       | P1     | 复用 T6 的 legacy spatial fixed/mixed E2E                          | 获得同环境 spatial redistribution oracle                                               | 仅作 oracle，不与 bundle/repeat 语义混写                          |
| T8 汇总审计             | P0     | 生成 correctness 表、CSV 和 `REPORT.md`                            | 每条结论可追溯到日志；正确性与性能分栏                                                 | 禁止删除失败日志或只报最好结果                                    |

---

## 4. 命令模板

执行 agent 应先进入包含 `bench_fixed_mixed.py`、`test_correctness.py`、`utils.py` 且能导入 Megatron-Core 的工作目录，并设置项目实际需要的 `PYTHONPATH`。

### 4.1 版本与环境

```bash
set -euo pipefail

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_tmf_repeat_fixed"
RUN_DIR="agent_run/${RUN_ID}"
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/metadata" "${RUN_DIR}/scripts"
printf 'stage\tlaunch_id\tstart_utc\tend_utc\texit_code\tlog_path\tcommand\n' \
  > "${RUN_DIR}/commands.tsv"

git rev-parse HEAD > "${RUN_DIR}/metadata/git_commit.txt"
git status --short > "${RUN_DIR}/metadata/git_status.txt"
git diff > "${RUN_DIR}/metadata/git_diff.patch"

sha256sum \
  bench_fixed_mixed.py \
  test_correctness.py \
  utils.py \
  > "${RUN_DIR}/metadata/code_sha256.txt"

sha256sum -c > "${RUN_DIR}/metadata/reviewed_hash_check.txt" <<'HASHES'
3b6c047bcdf9edaef7e6f11515c8ce06f53e2ed5e4d45b6c05ea791e20aac82c  utils.py
0d63e028caca696da5e29145dd845b177b7fb14f89824e5253020f4bf675249f  bench_fixed_mixed.py
fd8d0df4ca7c49a70d8374cb60e2b12059a948e6fa920df440fecd1b80a92b64  test_correctness.py
HASHES

nvidia-smi -q > "${RUN_DIR}/metadata/nvidia_smi_q.txt"
nvidia-smi topo -m > "${RUN_DIR}/metadata/topology.txt"

python - <<'PY' > "${RUN_DIR}/metadata/software.txt"
import sys
import torch
print("python", sys.version)
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("nccl", torch.cuda.nccl.version())
PY

python -m py_compile utils.py bench_fixed_mixed.py test_correctness.py

python bench_fixed_mixed.py --help > "${RUN_DIR}/metadata/bench_help.txt"
python test_correctness.py --help > "${RUN_DIR}/metadata/correctness_help.txt"

rg -n \
  "def repeat_logical_cp|def make_repeat_stack|strategy=.repeat.|final_replication|transition-local spatial replication" \
  bench_fixed_mixed.py \
  > "${RUN_DIR}/metadata/repeat_bench_code_check.txt"

rg -n \
  "_RepeatGatherSequenceForCPTransition|_RepeatScatterSequenceForCPTransition|zero-stride|strategy=.repeat.|DIRECTION_4_1" \
  utils.py \
  > "${RUN_DIR}/metadata/repeat_utils_code_check.txt"

rg -n \
  "correctness-strategy|run_repeat_transition_correctness|tensor_error_statistics|run_bundle_correctness|run_repeat_correctness|terminal_replication|initial_replication|replica consistency" \
  test_correctness.py \
  > "${RUN_DIR}/metadata/correctness_code_check.txt"
```

每个 `torchrun` 必须通过启用了 `set -o pipefail` 的独立 shell 执行，并在 `commands.tsv` 追加阶段、launch ID、UTC 起止时间、**pipeline 的真实退出码**、日志路径和完整命令。不得仅依据日志中出现 `PASS` 判断命令成功；退出码非零时，该 launch 必须记为失败并保留日志。正式 launch 不允许后台并发，以免 NCCL、显存和计时结果互相污染。

额外人工检查：

```bash
rg -n -A12 "def set_transition_direction" utils.py
```

必须能确认 `1<->4` 分支实际选择 `DIRECTION_4_1`。如果只有 `DIRECTION_4_1` 常量定义、但 `set_transition_direction()` 未接入，视为代码版本错误，停止。

如果脚本实际位于 `heterogeneous_cp/`，统一替换命令路径，不复制出第二份代码。

---

### 4.2 T2a：全量 correctness gate

使用 Config A 的核心 CP 比例，但 correctness 测试内部会构造显式 transition pattern，而不是依赖随机 pattern：

```bash
torchrun --standalone --nproc_per_node=8 test_correctness.py \
  --seq-length 4096 \
  --hidden-size 4096 \
  --num-attention-heads 32 \
  --num-query-groups 8 \
  --attention-backend flash \
  --global-batch-size 8 \
  --micro-batch-size 1 \
  --num-attention-layers 2 \
  --num-gdn-layers 6 \
  --layer-seed 1234 \
  --attn-cp-size 4 \
  --gdn-cp-size 1 \
  --fixed-cp-size 4 \
  --correctness-strategy all \
  --atol 1e-2 \
  --rtol 1e-2 \
  --input-grad-rel-l2-tol 5e-3 \
  --input-grad-cosine-min 0.9999 \
  --input-grad-normalized-max-tol 1e-2 \
  --param-grad-rel-l2-tol 1e-2 \
  --param-grad-global-rel-l2-tol 1e-2 \
  --param-grad-cosine-min 0.9999 \
  --param-grad-normalized-max-tol 2e-2 \
  2>&1 | tee "${RUN_DIR}/logs/correctness_all_cp4_1_launch1.log"
```

必须确认日志至少覆盖：

```text
Fixed vs mixed redistribution correctness
Bundle correctness (... roundtrip ...)
Repeat correctness (... roundtrip ...)
Repeat correctness (... terminal_replication ...)
Repeat correctness (... initial_replication ...)
Repeat transition algebra (... roundtrip ...)
Repeat transition algebra (... terminal_replication ...)
Repeat transition algebra (... initial_replication ...)
repeat replica consistency: PASS
out          PASS
input_grad   PASS          # temporal bundle/repeat case
grad_buffer  PASS
overall      PASS
```

说明：

- `legacy` correctness 不一定包含 `input_grad`；
- bundle/repeat correctness 必须比较 `out + input_grad + grad_buffer`；
- input-gradient 必须同时满足 global relative L2、cosine、normalized max、shape 和 non-finite gate；max-abs/max-rel 只作诊断；
- parameter-gradient 必须同时满足 max per-parameter relative L2、global relative L2、cosine、max per-parameter normalized max、shape 和 non-finite gate；逐元素误差只作诊断；
- repeat terminal case 还必须出现 physical replica consistency 检查；
- 不接受“output PASS 但 gradient FAIL”。

---

### 4.3 T2b：Repeat 最小语义测试（必须 GA=1）

这是本轮最关键的 repeat 语义验证。

8 GPU 下取：

```text
logical_cp = 4
logical_dp = 8 / 4 = 2
micro_batch_size = 1
global_batch_size = 2
gradient_accumulation_steps = 2 / (1 * 2) = 1
```

因此不存在第二个 temporal microbatch 可以被 bundle/repeat 借用。如果该 case 能通过，只能依赖 transition-local spatial replication。

```bash
torchrun --standalone --nproc_per_node=8 test_correctness.py \
  --seq-length 4096 \
  --hidden-size 4096 \
  --num-attention-heads 32 \
  --num-query-groups 8 \
  --attention-backend flash \
  --global-batch-size 2 \
  --micro-batch-size 1 \
  --num-attention-layers 2 \
  --num-gdn-layers 6 \
  --layer-seed 1234 \
  --attn-cp-size 4 \
  --gdn-cp-size 1 \
  --fixed-cp-size 4 \
  --correctness-strategy repeat \
  --atol 1e-2 \
  --rtol 1e-2 \
  --input-grad-rel-l2-tol 5e-3 \
  --input-grad-cosine-min 0.9999 \
  --input-grad-normalized-max-tol 1e-2 \
  --param-grad-rel-l2-tol 1e-2 \
  --param-grad-global-rel-l2-tol 1e-2 \
  --param-grad-cosine-min 0.9999 \
  --param-grad-normalized-max-tol 2e-2 \
  2>&1 | tee "${RUN_DIR}/logs/correctness_repeat_ga1_cp4_1_launch1.log"
```

验收：

1. transition algebra round-trip PASS；
2. transition algebra terminal replication PASS；
3. transition algebra initial replication PASS；
4. layered round-trip PASS；
5. layered terminal replication PASS；
6. layered initial replication PASS；
7. output PASS；
8. logical input-gradient norm PASS；
9. reduced parameter-gradient norm PASS；
10. physical replica consistency PASS；
11. 无 shape/NCCL/autograd error。

若此测试 FAIL，即使 `gbs=8` 的 repeat benchmark 能跑，也不得认为 repeat 语义已修复。

---

### 4.4 T2c：Correctness 独立重跑

该项已由本轮 same-layout controls 和完整目标配置结果关闭，不再阻塞 T3/T4。后续可将 T2a 和 T2b 的独立 `launch2` 作为非阻塞回归补充。

目的：

- 排除偶发 NCCL ordering；
- 排除未初始化 buffer；
- 排除 RNG / group creation 偶发问题。

若补充 launch 出现结构性失败，停止后续正式性能结论并重新打开 correctness；单纯复现已量化的 BF16 backward 非确定性不视为 transition 失败。

---

### 4.5 Config A 冒烟

Config A：

```text
world=8
Attention CP=4
GDN CP=1
fixed CP=4
Attn:GDN=2:6
seed=1234
```

当前 `seed=1234` 的随机 2:6 pattern 以 GDN 结尾，因此 mixed repeat 预计最终处于 CP1，日志中的 `final_replication` 预计为 `4`；fixed CP4 的 `final_replication` 应为 `1`。

```bash
torchrun --standalone --nproc_per_node=8 bench_fixed_mixed.py \
  --seq-length 4096 \
  --hidden-size 4096 \
  --num-attention-heads 32 \
  --num-query-groups 8 \
  --attention-backend flash \
  --global-batch-size 8 \
  --micro-batch-size 1 \
  --num-attention-layers 2 \
  --num-gdn-layers 6 \
  --layer-seed 1234 \
  --attn-cp-size 4 \
  --gdn-cp-size 1 \
  --fixed-cp-size 4 \
  --mode both \
  --temporal-strategy both \
  --warmup 2 \
  --iters 3 \
  2>&1 | tee "${RUN_DIR}/logs/configA_smoke_seq4096.log"
```

冒烟日志必须出现：

```text
temporal_bundle_fixed
temporal_bundle_mixed
temporal_repeat_fixed
temporal_repeat_mixed
distinct logical microbatches with transition-local spatial replication
grad_accumulation_steps=4
logical_global_batch=8
final_replication=
logical_tokens/s=
unique_tokens/s=
Bundle vs repeat
```

对该固定 seed，额外核对：

```text
temporal_repeat_fixed: final_replication=1
temporal_repeat_mixed: final_replication=4
```

如果实际随机 schedule 与预期不同，先以日志打印的 schedule/final CP 为准，并在报告中说明；不要为了满足预期值修改 seed 或代码。

---

### 4.6 Config A 主测

将 `SEQ` 依次设为 `32768`、`65536`：

```bash
SEQ=65536
LAUNCH=1

torchrun --standalone --nproc_per_node=8 bench_fixed_mixed.py \
  --seq-length "${SEQ}" \
  --hidden-size 4096 \
  --num-attention-heads 32 \
  --num-query-groups 8 \
  --attention-backend flash \
  --global-batch-size 8 \
  --micro-batch-size 1 \
  --num-attention-layers 2 \
  --num-gdn-layers 6 \
  --layer-seed 1234 \
  --attn-cp-size 4 \
  --gdn-cp-size 1 \
  --fixed-cp-size 4 \
  --mode both \
  --temporal-strategy both \
  --warmup 20 \
  --iters 50 \
  2>&1 | tee "${RUN_DIR}/logs/configA_seq${SEQ}_launch${LAUNCH}.log"
```

执行策略：

1. `seq=32768` 先做 1 次；
2. `seq=65536` 先做 1 次；
3. `seq=65536` 首次成功后，设置 `LAUNCH=2,3` 再做两个独立进程 launch；
4. 不要把单个进程内的 50 iterations 当作独立 launch。

必须采集：

- bundle fixed/mixed `both_ms`、std；
- repeat fixed/mixed `both_ms`、std；
- `logical_tokens/s`；
- `unique_tokens/s`；
- `final_replication`；
- peak allocated / reserved；
- matching fixed/mixed speedup。

注意：

- 修复版 repeat 中 `logical_tokens/s == unique_tokens/s` 是正常的；
- 不要自行推算“physical executed tokens/s”，因为 redundancy 随 layer/CP block 变化，不是一个单一最终 replication factor 可以完整描述的；
- `final_replication` 只描述最终 output layout 的 replication，用于 terminal loss normalization，不等于整个模型平均冗余倍数。

---

### 4.7 Config B 次测

Config B：

```text
Attention CP=8
GDN CP=2
fixed CP=8
logical_cp=8
logical_dp=1
GBS=8, MBS=1
=> repeat GA=8
```

```bash
torchrun --standalone --nproc_per_node=8 bench_fixed_mixed.py \
  --seq-length 65536 \
  --hidden-size 4096 \
  --num-attention-heads 32 \
  --num-query-groups 8 \
  --attention-backend flash \
  --global-batch-size 8 \
  --micro-batch-size 1 \
  --num-attention-layers 2 \
  --num-gdn-layers 6 \
  --layer-seed 1234 \
  --attn-cp-size 8 \
  --gdn-cp-size 2 \
  --fixed-cp-size 8 \
  --mode both \
  --temporal-strategy both \
  --warmup 20 \
  --iters 50 \
  2>&1 | tee "${RUN_DIR}/logs/configB_seq65536_launch1.log"
```

预期：

```text
grad_accumulation_steps=8
```

若对应 strategy 的 mixed 相对 fixed E2E speedup 不超过 `1.03×`，停止该 strategy / Config B 的额外 launch；保留首个结果。

---

### 4.8 逐模块 forward/backward profiling

该任务仍只使用已有 inline profiler，不修改 benchmark。

使用 legacy spatial 路径：

```text
--temporal-strategy none
```

原因：当前 bundle/repeat executor 是多 payload / 独立 logical microbatch 图，尚未接入单链 boundary-hook profiler。T6 的 transition 是 **legacy redistribution transition**，不是 repeat P2P transition，报告时必须明确。

Config A、B 在 `seq=65536` 下各做 3 次独立 launch：

```bash
for SPEC in "A 4 1 4" "B 8 2 8"; do
  read -r CFG ATTN_CP GDN_CP FIXED_CP <<< "${SPEC}"
  for LAUNCH in 1 2 3; do
    torchrun --standalone --nproc_per_node=8 bench_fixed_mixed.py \
      --seq-length 65536 \
      --hidden-size 4096 \
      --num-attention-heads 32 \
      --num-query-groups 8 \
      --attention-backend flash \
      --global-batch-size 8 \
      --num-attention-layers 2 \
      --num-gdn-layers 6 \
      --layer-seed 1234 \
      --attn-cp-size "${ATTN_CP}" \
      --gdn-cp-size "${GDN_CP}" \
      --fixed-cp-size "${FIXED_CP}" \
      --mode both \
      --temporal-strategy none \
      --warmup 20 \
      --iters 50 \
      2>&1 | tee \
        "${RUN_DIR}/logs/module_profile_config${CFG}_launch${LAUNCH}.log"
  done
done
```

对 `seed=1234`、2 Attention + 6 GDN 的当前随机 pattern，预期：

- 8 个 layer forward：2 Attention + 6 GDN；
- 3 个 transition forward；
- 与上述 component 一一对应的 backward；
- 1 个 `loss bwd`；
- 1 个 `grad sync (all_reduce)`；
- `bwd_residual`；
- E2E decomposition；
- mean/std 均为有限非负数。

fixed plan 应有 8 个 layer forward/backward、0 个 transition。

Agent 可以在 `${RUN_DIR}/scripts/` 新建日志解析脚本，但不得修改 benchmark。

输出：

1. `module_profile.csv`：每个 layer/transition 一行；
2. `module_profile_summary.csv`：按 launch 汇总。

解释边界：

- 组件时间来自 rank 0 CUDA events；
- 显存为 benchmark 的 worst-rank allocated/reserved；
- T6 不提供 repeat-specific transition breakdown。

---

### 4.9 Spatial oracle

T6 已使用：

```text
--mode both --temporal-strategy none
```

因此默认复用 T6 的 fixed/mixed E2E 作为 spatial redistribution oracle，不再重复 launch。

该 oracle 的语义是：

> 完整 spatial batch 满足所有 DP layout divisibility 时，直接使用现有 batch-redistribution transition 的乐观参考。

它不是 bundle，也不是 spatial-repeat。

---

## 5. Correctness 必须输出的结果

生成 `correctness_summary.csv`，至少包含：

```text
run_id,launch_id,test_group,case_name,strategy,
seq_length,global_batch_size,micro_batch_size,
attn_cp_size,gdn_cp_size,fixed_cp_size,
out_pass,input_grad_pass,param_grad_pass,replica_consistency_pass,
overall_pass,exit_code,status,log_path
```

字段要求：

- `test_group ∈ {all_cp4_1,repeat_ga1_cp4_1}`；
- `strategy ∈ {transition,legacy,bundle,repeat}`；
- `case_name`：
  - transition：`roundtrip`、`terminal_replication`、`initial_replication`；
  - legacy：`fixed_vs_mixed`；
  - bundle：`roundtrip`；
  - repeat：`roundtrip`、`terminal_replication`、`initial_replication`；
- legacy 没有 input-gradient 检查时填 `n/a`；
- bundle 没有 replica consistency 时填 `n/a`；
- repeat 的 terminal case 必须记录 replica consistency；
- 任何 grad FAIL 都使 `overall_pass=false`。
- 追加 `input_grad_global_rel_l2`、`input_grad_cosine`、
  `input_grad_normalized_max`、`input_grad_per_sample_max_abs`；
- 追加每个正式 norm/cosine/normalized-max/finite/shape check 的 PASS/FAIL；逐元素 gradient 误差仅作诊断字段。

另生成 `correctness_errors.csv`，尽可能从日志解析：

```text
run_id,launch_id,strategy,case_name,tensor_name,
max_abs,max_rel,max_scaled_error,violation_count,violation_fraction,nonfinite_count,
max_param_rel_l2,global_rel_l2,cosine,global_normalized_max,
max_param_normalized_max,worst_param_index,worst_param_shape,formal_checks,log_path
```

无法从某类检查得到的字段填 `n/a`，不得伪造 0。

---

## 6. 性能结果表

`summary.csv` 至少包含：

```text
run_id,launch_id,config,seq_length,strategy,plan,
global_batch_size,micro_batch_size,grad_accumulation_steps,
logical_cp,final_cp,final_replication,
both_ms,both_std_ms,logical_tokens_s,unique_tokens_s,
peak_alloc_gib,peak_reserved_gib,speedup_vs_matching_fixed,
exit_code,status,log_path
```

其中：

- `strategy ∈ {bundle, repeat, spatial_oracle}`；
- `plan ∈ {fixed, mixed}`；
- speedup 只能在相同 strategy、相同 GBS/MBS、相同序列长度的 fixed/mixed 间计算；
- repeat 中 `logical_tokens_s == unique_tokens_s` 是修复版的正常统计；
- 不再沿用旧文档中“repeat executed tokens/s 作为冗余物理吞吐”的解释；
- 若代码返回字段仍叫 `executed_tokens_s`，该字段当前实际等于 logical throughput，解析脚本应统一映射为 `logical_tokens_s`，并在报告中注明兼容字段名；
- 不得用 `final_replication` 推导整个模型 physical executed tokens，因为重复仅出现在部分 CP block。

`module_profile.csv` 至少包含：

```text
run_id,launch_id,config,plan,schedule,component_kind,
component_index,module_type,cp_size,src_cp,dst_cp,
fwd_mean_ms,fwd_std_ms,bwd_mean_ms,bwd_std_ms,
profiler_rank,transition_semantics,log_path,status
```

T6 的：

```text
transition_semantics=legacy_redistribute
```

`module_profile_summary.csv` 至少包含：

```text
run_id,launch_id,config,plan,
attention_fwd_ms,attention_bwd_ms,gdn_fwd_ms,gdn_bwd_ms,
transition_fwd_ms,transition_bwd_ms,loss_bwd_ms,
grad_sync_ms,bwd_residual_ms,e2e_ms,profiler_rank,status
```

逐 launch 检查事件数量，不允许用缺失事件均值补零。组件总和与 E2E 差异保留为 `bwd_residual_ms`。

---

## 7. 报告模板

远程 agent 最终生成 `REPORT.md`，依次包含：

1. **Run identity**
   - commit；
   - dirty files / patch；
   - 三脚本 SHA256；
   - 硬件、软件、拓扑；
   - 起止时间。
2. **Code semantic audit**
   - repeat P2P autograd symbols；
   - `make_repeat_stack`；
   - `repeat_logical_cp`；
   - 目标 Megatron-LM 的 GDN CP 为 headwise-only，且 wrapper 未传入 `linear_cp_mode`；
   - `run_repeat_transition_correctness`；
   - CP1↔CP4 direction table 是否已接入；
   - correctness CLI。
3. **Correctness gate**
   - T2a 两次独立 launch；
   - T2b GA=1 repeat 两次独立 launch；
   - output/input-grad/parameter-grad/replica-consistency；
   - 最大误差与 gradient norm/cosine 指标。
4. **Config A**
   - bundle fixed/mixed；
   - repeat fixed/mixed；
   - 三次 launch 原始值；
   - median；
   - final replication；
   - peak memory。
5. **Config B**
   - 首次筛选；
   - 是否继续；
   - 若继续，独立 launch 汇总。
6. **Module profiling**
   - 明确标注为 legacy redistribution transition；
   - 逐 layer/transition；
   - Attention/GDN/transition/loss/grad-sync/residual 汇总。
7. **Spatial oracle**
   - 仅作 legacy spatial redistribution 参考。
8. **显存**
   - worst-rank allocated/reserved。
9. **失败与重试**
   - 原始退出码；
   - 错误类别；
   - 对应日志；
   - 不删除失败 launch。
10. **Go/No-Go**
   - correctness 是否支持继续 repeat performance；
   - bundle / repeat 是否值得进入 1F1B；
   - 是否需要先做 repeat transition profiling / overlap。

---

## 8. Go/No-Go 判据

### 8.1 Correctness Go

必须同时满足：

- T2a `legacy` PASS；
- T2a repeat transition algebra 三 case 全 PASS；
- T2a `bundle roundtrip` PASS；
- T2a repeat：
  - roundtrip PASS；
  - terminal replication PASS；
  - initial replication PASS；
- T2b **GA=1 repeat** 三 case 全 PASS；
- output、input gradient、parameter gradient 均满足脚本阈值；
- terminal repeat physical replica consistency PASS；
- 两次独立 launch 均无确定性/随机 NCCL failure。

只要 T2b GA=1 FAIL，就判定：

> **Repeat semantic No-Go**

禁止用后续性能结果声称 spatial repeat 正确。

### 8.2 Bundle 性能 Go

Config A、seq=65536 三次独立 launch：

```text
median(bundle_fixed_ms / bundle_mixed_ms) >= 1.10x
```

且：

- peak allocated 不显著恶化到不可运行；
- 无 correctness 问题。

### 8.3 Repeat 性能 Go

在 correctness Go 前提下：

```text
median(repeat_fixed_ms / repeat_mixed_ms) >= 1.05x
```

即可认为 spatial-repeat 值得继续研究。

若：

```text
>= 1.10x
```

则列为强 Go，优先进入 1F1B integration。

repeat 的性能判读必须使用：

```text
logical / unique tokens/s
```

不能将 replicated low-CP compute 当成额外有效数据吞吐。

### 8.4 Pipeline Go

若：

- bundle median ≥1.10×，或
- repeat median ≥1.10×，或
- 两者在不同 GBS / GA regime 各有优势，

则建议进入：

1. non-interleaved 1F1B；
2. temporal cohort rendezvous；
3. cross-microbatch transition/computation overlap；
4. bundle/repeat/hybrid planner。

### 8.5 性能 No-Go

如果 Config A：

- bundle median <1.05×；
- repeat median <1.05×；

不要立即否定整体 idea。先做成本分解：

- layer CP gain；
- transition cost；
- repeat redundant-compute cost；
- temporal wait / cohort cost；
- peak memory。

只有在核心 layer CP gain 本身不存在时才考虑方向性 No-Go。

---

## 9. Agent 禁止事项

- 不得修改 `utils.py`、`bench_fixed_mixed.py`、`test_correctness.py`。
- 不得为了让 correctness PASS 修改 atol/rtol 后不报告。
- 当前版本不得跳过已冻结的 correctness 回归直接修改实现并沿用旧性能结论。
- 不得把旧 repeat（clone temporal microbatch）和新 repeat（spatial replication）日志混入同一 CSV。
- 不得把 `final_replication` 当成全模型平均冗余倍数。
- 不得把 `logical_tokens/s` 写成 physical hardware token throughput。
- 不得把 legacy profiler 的 transition 时间写成 repeat transition 时间。
- 不得删除 OOM、NCCL、autograd、correctness FAIL 日志。
- 不得仅汇报最快一次 launch。
- 不得因为 65536 OOM 而静默降 seq；降配置必须作为独立 experiment 明确记录。

---

## 10. 给远程 agent 的最终指令

当前从 T3 开始按：

```text
T3 -> T4 -> T5 -> T6 -> T7 -> T8
```

顺序执行。T0–T2 已按本文件 0.1 节关闭；若目标代码发生变化，则从 T0 重新开始。

核心原则：

> **先证明语义，再讨论性能。**

尤其是 `repeat`：

> `GBS=2, MBS=1, logical CP4, CP4↔CP1` 的 GA=1 correctness 是本轮最关键验收项。它用于排除“repeat 仍然只是 temporal clone/bundle 变体”的可能性。

若代码缺失、T2 确定性失败、结果字段不足或 repeat GA=1 correctness FAIL：

1. 不修改研究代码；
2. 写入 `NEEDS_RESEARCHER.md`；
3. 附原始命令、日志、退出码、最后 200 行 stderr/stdout；
4. 停止对应后续阶段。

最终交付整个 `${RUN_DIR}`，至少包括：

```text
metadata/
logs/
scripts/
commands.tsv
correctness_summary.csv
correctness_errors.csv
summary.csv
module_profile.csv
module_profile_summary.csv
REPORT.md
NEEDS_RESEARCHER.md   # 仅在需要时
```

所有成功与失败日志都必须保留。
