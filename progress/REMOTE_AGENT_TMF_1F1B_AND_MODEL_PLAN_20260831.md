# 知识库主题：TMF 当前版本桥接复测、1F1B 集成验证与性能模型 Agent 执行计划

> 日期：2026-08-31
> 执行对象：本地代码 agent + 远程 8×H800 GPU agent
> 上游依据：`TMF_PROFILING_DATA_AUDIT_AND_1F1B_ROADMAP_20260831.md`
> 优先级：本文件取代 `REMOTE_AGENT_TMF_TASK_PLAN_20260829.md` 中尚未执行的旧 T3–T8 性能步骤。

## 0. 总规则

1. 不把 2026-08-16/22 的绝对 latency 混入当前版本拟合数据；旧数据标记为 `historical_prior`。
2. 每个 run 必须冻结 commit、dirty diff、文件 hash、硬件拓扑和软件版本。
3. 原始逐迭代与逐事件数据必须保留，不能只交付 summary。
4. latency 使用 worst-rank step time；组件事件保留全部 rank 后再构造 critical path。
5. repeat 吞吐同时报告 executed 与 unique/logical tokens/s，判定只使用后者。
6. 不静默删除 OOM、NCCL、timeout、nonfinite 或 correctness FAIL。
7. correctness gate 未通过时，停止该实现对应的性能阶段。

## 1. 任务依赖

| 任务 | 执行者 | 依赖 | 产物 | 退出条件 |
| --- | --- | --- | --- | --- |
| A0 版本冻结 | GPU agent | 无 | `metadata/`、`commands.tsv` | hash 与批准版本一致 |
| A1 桥接复测 | GPU agent | A0 | `bridge_raw.csv`、`bridge_summary.csv` | 当前 bundle/repeat 均可执行 |
| A2 1F1B reference harness | code agent | 无 GPU 也可推进 | GPipe runner、state capture、CSV schema | 静态检查与 schedule 单测通过 |
| A3 fixed-CP 1F1B correctness | GPU agent | A0、A2 | `fixed_1f1b_correctness.csv` | 全矩阵 gate PASS |
| A4 fixed-CP 1F1B profiling | GPU agent | A3 | per-rank timeline、step summary | 5 launch 完整 |
| A5 stage-local boundary | code agent | A3 | scoped mesh、identity boundary | no-op parity PASS |
| A6 bundle-aware 1F1B | code agent | A5 | cohort scheduler、BoundaryRecord | 本地静态审查通过 |
| A7 TMF 1F1B correctness | GPU agent | A6 | `tmf_1f1b_correctness.csv` | 三角对照全部 PASS |
| A8 TMF 1F1B profiling | GPU agent | A7 | per-rank timeline、memory、summary | 主矩阵完整 |
| A9 性能模型 | code agent | A1、A4、A8 | fitter、event simulator、report | holdout 阈值通过 |

允许 A1 与 A2 并行；A3 之后才能修改 fixed-CP scheduler 的通信语义。A5/A6 修改后必须重新执行 A3 的固定 CP 回归。

## 2. A0：版本与环境冻结

创建：

```text
agent_run/<UTC>_tmf_1f1b_model/
  metadata/
  logs/
  raw/
  summaries/
  scripts/
  commands.tsv
```

`metadata/` 至少包含：

```text
git_commit.txt
git_status.txt
git_diff.patch
code_sha256.txt
software.txt
nvidia_smi_q.txt
topology.txt
environment_allowlist.txt
```

`code_sha256.txt` 必须包含：

```text
utils.py
bench_fixed_mixed.py
test_correctness.py
bench_1f1b_attention.py
model_cp_transition.py
```

代码或依赖版本不一致时停止 A1/A3，写 `NEEDS_RESEARCHER.md`，不得自行套用旧阈值。

## 3. A1：桥接复测

固定模型参数：hidden 4096、32 heads、8 query groups、2 Attention + 6 GDN、GBS 8、MBS 1、BF16、Flash attention。

主矩阵：

```text
Config A: attn_cp=4, gdn_cp=1, fixed_cp=4
Config B: attn_cp=8, gdn_cp=2, fixed_cp=8
seq: 65536
strategy: bundle, repeat
plan: fixed, mixed
warmup: 20
measured iterations: 50
independent launches: 5
launch order: interleaved/ABBA
```

补充矩阵：A/B、seq 32768、bundle、fixed/mixed、2 launch。

每次 iteration 输出：

```text
run_id,code_hash,launch_id,iteration,config,seq,strategy,plan,
step_ms,worst_rank,executed_tokens_s,logical_tokens_s,
max_alloc_gib,max_reserved_gib,status
```

summary 输出 median、mean、std、CV、p5、p95、跨 launch median/极差。比较 2026-08-22 历史带：median 差异 ≤5%、排序一致、allocated 差异 ≤1 GiB 为 bridge PASS。超出时不判实现失败，而是触发 32K/65K/128K 全量重采集。

## 4. A2：1F1B reference harness

Code agent 修改 `bench_1f1b_attention.py` 时遵守：

1. 保留 non-interleaved schedule 的独立 pure function；
2. 新增 GPipe runner，使用完全相同参数和 canonical microbatch inputs；
3. 捕获 final canonical output、stage-0 input gradient、每 stage parameter gradient；
4. 统一参数初始化并保存 seed；
5. 增加 `--schedule {gpipe,1f1b,both}`、`--warmup-steps`、`--iters`、`--jsonl`；
6. trace 每行包含 rank/stage/microbatch/pass/event begin/end；
7. step 计时不含模型构建和参数同步；
8. correctness 与性能模式分离，性能模式不保留全量 CPU snapshot。

无 GPU 静态验收：

- `compile(source, ..., 'exec')` 通过；
- `expected_stage_events()` 对 PP1/2/4、M1/2/4/8/16 满足每 stage 恰好 M 个 F 和 M 个 B；
- 非法 PP/CP/world、序列整除和 M 参数在 setup 前失败；
- CSV/JSONL schema 固定并有解析自检。

## 5. A3：fixed-CP 1F1B correctness

执行：

```text
PP1×CP8: M=8
PP2×CP4: M=4,8,16
PP4×CP2: M=4,8,16
seq=4096, MBS=1, layers/stage=2
schedule=both
independent launches=2
```

每个 case 验证：

- 所有 rank 无死锁且退出码为 0；
- 每 stage 的事件序列等于 canonical non-interleaved 1F1B；
- GPipe vs 1F1B output elementwise allclose；
- input gradient 和 parameter gradient 满足项目冻结的 BF16 范数 gate；
- loss、梯度、optimizer update 均 finite；
- 两次 launch 的 schedule identity 完全一致。

任何 case FAIL 时停止 A4–A8；保留 timeout 前最后 200 行与各 rank trace。

## 6. A4：fixed-CP 1F1B profiling

配置：

```text
PP2×CP4, PP4×CP2
M=8,16
seq=32768,65536
warmup steps=10
measured steps=30
independent launches=5
```

逐事件 schema：

```text
run_id,launch,step,rank,stage,cp_rank,microbatch,pass,event,
cuda_stream,start_us,end_us,duration_us,bytes,peer,group_id,status
```

事件至少包含 `layer_fwd`、`layer_bwd`、`pp_send_fwd`、`pp_recv_fwd`、`pp_send_bwd`、`pp_recv_bwd`、`grad_sync`、`idle_wait`。另保存 step worst rank、pipeline utilization、stage idle、peak allocated/reserved。

## 7. A5–A7：stage-local TMF 与 correctness

### A5 identity boundary

实现 `StageMesh` 和 `BoundaryRecord`，先运行 high-CP→high-CP no-op。GPipe/1F1B 与无 boundary 基线必须在同一 gate 内一致。

### A6 bundle boundary

只实现单个 stage 内的：

```text
Attention(high CP) -> bundle -> GDN(low CP) -> inverse bundle
```

首批配置：

```text
PP2×CP4, high CP4, low CP1, cohort=4, M=8/16
PP4×CP2, high CP2, low CP1, cohort=2, M=8/16
seq=4096 correctness
```

记录 microbatch、optimizer step、bundle、logical sample 和 layout identity。`M % cohort != 0` 必须在执行前失败。

### A7 三角对照

每个配置比较：

```text
same-plan GPipe vs 1F1B
static-high-CP vs bundle-TMF
launch1 vs launch2 schedule identity
```

除 output/input-grad/parameter-grad 外，增加一次 optimizer update parity。发生 correctness FAIL 时不得进入 A8。

## 8. A8：TMF 1F1B profiling

主矩阵：

```text
PP2×CP4: M=8,16; seq=32768,65536
PP4×CP2: M=8,16; seq=32768,65536
plans: static-high, bundle-TMF, repeat-TMF control
launches: 5; warmup: 10; measured: 30
```

除 A4 schema 外增加：

```text
bundle_ready
cohort_wait
pack
transition_fwd
low_cp_block_fwd
inverse_transition_fwd
transition_bwd
low_cp_block_bwd
unpack
activation_live_bytes
boundary_buffer_bytes
```

summary 必须给出 logical tokens/s、speedup、cohort wait 占比、critical-path transition、pipeline bubble、peak memory 和 repeat unique/executed 比。

## 9. A9：性能模型

Code agent 先实现数据校验器，再拟合模型。数据校验器拒绝：

- 缺 code hash/topology/measurement semantics；
- 同一 case launch 少于计划值；
- 非 finite 时间或负 duration；
- step time 小于 critical path；
- event 数与 schedule trace 不一致；
- 当前数据与 historical prior 无标签混合。

按完整 case 划分 train/holdout。建议把一个 PP×CP、一个序列长度和一个 M 组合完全留作 holdout。输出：

```text
model_coefficients.json
fit_cases.csv
holdout_predictions.csv
timeline_predictions.jsonl
MODEL_REPORT.md
```

验收：primitive MAPE ≤5%；fixed 1F1B step MAPE 首轮 ≤8%；TMF step MAPE ≤8%；peak allocated 误差 ≤5%；计划排序准确率 ≥90%。不满足时报告最差 case 和缺失交互项，不通过扩大容差隐藏误差。

## 10. Agent 最终交付与停止规则

最终 run 目录至少包含：

```text
metadata/
logs/
raw/
summaries/
scripts/
commands.tsv
REPORT.md
NEEDS_RESEARCHER.md       # 有阻塞时
```

Agent 可自主修复解析脚本、报告脚本和纯测试代码；不得在发现 correctness FAIL 后修改研究语义、放宽 gate 并继续性能测试。研究代码需要修改时，保留失败 run，提交独立 patch，重新从 A0/A3 开始。
