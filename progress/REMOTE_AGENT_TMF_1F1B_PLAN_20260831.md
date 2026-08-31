# 知识库主题：TMF 1F1B 集成与验证 Agent 执行计划

> 日期：2026-08-31
> 上游路线：`TMF_1F1B_ROADMAP_20260831.md`
> 执行对象：本地 code agent 与远程 8×H800 GPU agent

## 1. 执行规则

1. 每轮冻结 commit、dirty diff、五个核心脚本 hash、软件版本和 GPU 拓扑。
2. 历史数据标记为 `historical_prior`，不得与当前版本原始数据无标签合并。
3. 保留逐迭代、逐 rank 原始数据；summary 不能替代 raw data。
4. correctness FAIL、NCCL timeout、OOM 或 nonfinite 时停止对应后续性能任务。
5. repeat 使用 logical/unique tokens/s 判定，不使用 executed tokens/s 宣称加速。
6. 不通过放宽 correctness gate 或删除异常 launch 获得 PASS。

## 2. 任务表

| 任务 | 执行者 | 依赖 | 交付物 |
| --- | --- | --- | --- |
| A0 版本冻结 | GPU agent | 无 | `metadata/`、`commands.tsv` |
| A1 桥接复测 | GPU agent | A0 | `bridge_raw.csv`、`bridge_summary.csv` |
| A2 GPipe/1F1B harness | code agent | 无 | reference、state capture、trace 输出 |
| A3 fixed-CP correctness | GPU agent | A0、A2 | `fixed_1f1b_correctness.csv` |
| A4 fixed-CP profiling | GPU agent | A3 | per-rank timeline、summary |
| A5 stage-local bundle | code agent | A3 | mesh、boundary、cohort scheduler |
| A6 TMF correctness | GPU agent | A5 | `tmf_1f1b_correctness.csv` |
| A7 TMF profiling | GPU agent | A6 | timeline、memory、summary、report |

A1 与 A2 可以并行。A5 修改后必须重跑 A3 固定 CP 回归。

## 3. A0：版本冻结

创建：

```text
agent_run/<UTC>_tmf_1f1b/
  metadata/
  logs/
  raw/
  summaries/
  scripts/
  commands.tsv
```

`metadata/` 至少包含 Git commit/status/diff、软件版本、NVIDIA SMI、拓扑和以下文件 hash：

```text
utils.py
bench_fixed_mixed.py
test_correctness.py
bench_1f1b_attention.py
model_cp_transition.py
```

hash 与批准版本不一致时停止，写 `NEEDS_RESEARCHER.md`。

## 4. A1：桥接复测

固定：BF16、Flash attention、hidden 4096、2 Attention + 6 GDN、GBS 8、MBS 1、20 warmup、50 measured iterations。

```text
Config A: attn_cp=4, gdn_cp=1, fixed_cp=4
Config B: attn_cp=8, gdn_cp=2, fixed_cp=8
seq=65536
strategy=bundle,repeat
plan=fixed,mixed
launches=5
```

补充 A/B、seq=32768、bundle fixed/mixed，各 2 launch。

raw 至少记录 launch、iteration、config、seq、strategy、plan、step_ms、logical tokens/s、worst-rank peak allocated/reserved、status。summary 报告 median、p5、p95、CV 和跨 launch 极差。

若相对历史 median 差异超过 5% 或排序改变，在报告中标记 `FULL_RECOLLECT_REQUIRED`，后续补跑 32K/65K/128K 全矩阵。

## 5. A2：GPipe/1F1B harness

Code agent 在 `bench_1f1b_attention.py` 中完成：

- `--schedule {gpipe,1f1b,both}`；
- 同权重、同 canonical input 的 GPipe reference；
- output、stage-0 input gradient、各 stage parameter gradient capture；
- warmup/measured steps和 JSONL/CSV 输出；
- rank/stage/microbatch/F-B schedule trace；
- correctness 与 performance 模式分离。

提交前执行源码编译、参数检查和 pure schedule 测试。PP1/2/4、M1/2/4/8/16 下，每个 stage 必须恰好有 M 个 forward 和 M 个 backward。

## 6. A3：fixed-CP correctness

```text
PP1×CP8: M=8
PP2×CP4: M=4,8,16
PP4×CP2: M=4,8,16
seq=4096, MBS=1, layers/stage=2
schedule=both, launches=2
```

验收：

- 无死锁，事件序列等于 canonical non-interleaved 1F1B；
- GPipe vs 1F1B output 满足 elementwise allclose；
- input/parameter gradient 满足项目冻结的 BF16 范数 gate；
- loss、gradient 和一次 optimizer update finite；
- 两次 launch 的 schedule identity 一致。

任一 case FAIL 时停止 A4–A7。

## 7. A4：fixed-CP profiling

```text
PP2×CP4, PP4×CP2
M=8,16
seq=32768,65536
warmup steps=10, measured steps=30, launches=5
```

逐 rank 记录 layer F/B、PP send/recv F/B、grad sync、idle wait、step time 和 peak allocated/reserved。每条事件包含 step、rank、stage、microbatch、event、start/end、bytes/peer。

## 8. A5–A6：stage-local bundle 与 correctness

Code agent 先实现 stage-local mesh 和 identity boundary，再实现：

```text
Attention(high CP) -> bundle -> GDN(low CP) -> inverse bundle
```

首轮配置：

```text
PP2×CP4: CP4->1, cohort=4, M=8/16
PP4×CP2: CP2->1, cohort=2, M=8/16
seq=4096
```

boundary 维护 optimizer step、pipeline microbatch、bundle、logical sample 和 layout identity。cohort 不得跨 optimizer step，非法 `M % cohort != 0` 必须提前失败。

GPU agent 依次验证：

1. identity boundary vs 无 boundary；
2. 同计划 GPipe vs 1F1B；
3. static-high vs bundle-TMF；
4. output、input/parameter gradient 和一次 optimizer update；
5. 两次 launch 的 schedule/bundle identity。

全部通过后进入 A7。

## 9. A7：TMF profiling

```text
PP2×CP4, PP4×CP2
M=8,16
seq=32768,65536
plans=static-high,bundle-TMF,repeat-TMF-control
warmup=10, measured=30, launches=5
```

在 A4 事件基础上增加 cohort ready/wait、pack、transition F/B、low-CP block F/B、inverse transition 和 boundary memory。

报告 logical tokens/s、跨 launch speedup、cohort wait、pipeline bubble、transition critical path 和 worst-rank peak memory。不得用 repeat executed throughput 作为有效吞吐。

## 10. 最终交付

```text
metadata/
logs/
raw/
summaries/
scripts/
commands.tsv
REPORT.md
NEEDS_RESEARCHER.md   # 有阻塞时
```

`REPORT.md` 只回答：桥接是否通过、fixed-CP 1F1B 是否正确、bundle-TMF 是否正确、是否存在可复现加速、主要等待和显存瓶颈是什么。性能模型仅整理输入数据，不在本轮展开建模。
