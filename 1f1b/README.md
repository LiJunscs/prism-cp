# Cohort-aware TMF 1F1B

本目录承载 TMF 与流水线并行的独立实现：

- `plan.py`：逐层 CP pattern、temporal width、全局 cohort width 和 PP boundary 冻结；
- `tmf_adapter.py`：logical identity、pack/split、CP transition、共享 autograd graph 和双向 P2P；
- `scheduler.py`：以 cohort 为调度单元的 non-interleaved 1F1B；
- `runtime.py`：stage-local process group、参数/梯度同步和逐 rank JSONL；
- `profiling.py`：逐组件 CUDA event/host wait 采样；
- `analyze_profile.py`：合并逐 rank profile 并输出 stage bottleneck；
- `bench_tmf_1f1b.py`：命令行入口；
- `test_plan.py`：不依赖 GPU 的计划层测试。

这里的 `--context-parallel-size` 是每个 PP stage 的固定 GPU mesh size。某层配置 CP=`c` 时，该层的有效 DP 为 `stage_mesh/c`；因此 mixed CP 同时也是逐层 mixed DP/CP。完整映射、temporal width 公式和当前支持边界见 `progress/TMF_1F1B_ADAPTER_DESIGN_EXPLANATION_20260831.md`。

初版范围：

1. 每个 stage 内允许逐层异构 CP；
2. 一个 optimizer step 的 `M` 个 logical microbatch 被分为 `M/K` 个 pipeline cohort；
3. Adapter 对一个 cohort 的 K 个输入执行真实 temporal pack/remesh/split；
4. K 个输出梯度到齐后，以一次 `torch.autograd.backward` 遍历共享图；
5. scheduler 不直接调用 distributed P2P；
6. stage 入口/出口 temporal width 必须为 1；
7. 相邻 stage CP 必须相同，跨 stage CP transition 留待下一阶段。

固定 CP 时 `K=1`，执行自然退化为标准 1F1B。

更严格地说，固定 CP 只有在 stage entry/exit temporal width 为 1 时才退化为 K=1。full-mesh CP 总满足该条件；lower CP 还要求 logical MBS 可以被对应 DP 整除。当前尚不支持在 stage/PP boundary 上直接传 width>1 payload。

示例：

```bash
torchrun --nproc_per_node 8 1f1b/bench_tmf_1f1b.py \
  --pipeline-parallel-size 2 \
  --context-parallel-size 4 \
  --stage-layer-patterns \
  "attention:4,gdn:1,attention:4;attention:4,gdn:1,attention:4" \
  --micro-batches 8 \
  --micro-batch-size 1 \
  --seq-length 4096 \
  --trace-dir agent_run/tmf_1f1b_smoke
```

该配置得到 `K=4`、`pipeline_cohorts=2`。这只是 correctness-first 的 macro-cohort 版本；后续的流式 Adapter 需要允许 logical microbatch 在 stage 内部 TMF boundary 暂停和恢复，以降低 cohort 粒度带来的 pipeline bubble。

R4 数值正确性验证已经完成并从活跃运行路径移除。冻结源码、测试、原执行脚本和校验清单位于 `progress/archive/TMF_1F1B_R4_20260831/`。

## 分阶段 profiling

Profiling 必须单独启动，先 warmup，再采集 measured steps：

```bash
torchrun --nproc_per_node 8 1f1b/bench_tmf_1f1b.py \
  --pipeline-parallel-size 2 --context-parallel-size 4 \
  --stage-layer-patterns \
  "attention:4,gdn:1,attention:4;attention:4,gdn:2,attention:4" \
  --micro-batches 8 --micro-batch-size 1 --seq-length 4096 \
  --warmup 5 --iters 10 --profile-stages \
  --trace-dir agent_run/tmf_profile_mixed
```

每个 rank 写出 `profile_rank_XXXX.json`，rank 0 合并为 `profile_summary.json`。`cuda_ms` 表示当前 CUDA stream 的 event 时间，`wall_ms` 包含主机和 NCCL wait。汇总会避免把 `cohort_backward` 父范围与子范围相加；但 grouped backward 的多个 sibling hook range 仍可能重叠，因此 `layer_backward`/`tmf_transition_backward` 是 inclusive diagnostics，不能相加。非重叠反向阶段应读取 `cohort_backward`。Profiling 默认关闭逐 tensor finite 同步检查；如需保留，增加 `--profile-check-finite`。

远端可直接执行 `bash 1f1b/run_profile.sh <run-dir>`，依次运行 static/CP1/CP2 profiling，并为每个 case 保存 `command.sh` 与 `stdout.log`。

## Transition 本地融合与前向掩盖

TMF transition 默认使用 `--tmf-transition-kernel triton`，把 scatter/gather
两侧原有的多次 `index_select`、`cat`、`contiguous` 和 `index_copy_` 合并为
发送打包与接收最终落位 kernel。没有 Triton 或不在 CUDA 上时会自动使用
等价 PyTorch oracle；也可显式传入 `--tmf-transition-kernel torch` 做同路径
消融。

`--tmf-transition-overlap` 默认开启。在 CP 收缩（例如 CP4→CP1）时，每个
temporal lane 的生产者完成后立即 enqueue NCCL P2P，后续 Attention lane
继续在计算流执行；只有低 CP consumer 即将启动时才等待未完成的通信尾部。
可使用 `--no-tmf-transition-overlap` 恢复整 cohort 同步 transition。当前
反向仍使用同步 inverse transition，后续需单独验证 backward/inter-cohort
overlap。

`--tmf-transition-lane-group-size N` 控制一次 NCCL enqueue 聚合的连续 lane
数量，默认 `N=1`，保持最细粒度 overlap。该值必须整除 CP 收缩倍数：CP4→CP1
可测试 `1/2/4`，CP4→CP2 可测试 `1/2`。增大该值会减少 NCCL group/kernel
数量，但也会推迟首个通信的启动时间，用于探索通信粒度与 overlap 窗口之间
的折中。该参数仅影响开启 overlap 的前向 CP contraction，反向和 CP expansion
仍保持同步。
