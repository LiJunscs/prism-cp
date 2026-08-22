# 知识库主题：TMF 与重复 Microbatch 原型远程服务器执行计划

> 日期：2026-08-22  
> 执行对象：远程 8×H800 服务器 agent  
> 目标脚本：`bench_fixed_mixed.py`  
> 工作性质：验证与采集；除非任务书明确授权，不修改 benchmark 代码

## 1. 本轮目标

验证新增的两种 temporal 执行路径能够在真实 8-GPU 环境完成 forward、backward 和一次参数梯度同步，并采集 optimizer-step 级延迟、吞吐与峰值显存：

1. `bundle`：使用同一 optimizer step 内不同的 gradient-accumulation microbatch；
2. `repeat`：在每个源 DP replica 内重复第一个 microbatch；
3. 对比 temporal fixed、temporal mixed 与旧 spatial fixed/mixed oracle；
4. 利用现有完整 inline profiler，采集 Attention、GDN、transition、loss backward、gradient sync 与 residual 的逐模块前向/反向分解。

本轮不验证 optimizer update、长程 loss parity、checkpoint/recompute 或 PP，不得据此声称完整训练正确性。逐模块 profiler 仅用于 legacy spatial fixed/mixed 路径；temporal cohort graph 尚未接入该 profiler。

## 2. 执行前硬约束

- 必须使用包含 `--micro-batch-size` 和 `--temporal-strategy` 的新版 `bench_fixed_mixed.py`。
- 所有正式结果必须绑定 commit、dirty patch/hash、完整命令、环境和原始日志。
- 不得把 `repeat` 的 executed tokens/s 当作有效训练吞吐；必须同时报告 unique tokens/s。
- `repeat` 只在确定性重复样本下保持平均梯度尺度，不等价于包含不同样本的 global batch。
- 遇到 shape、NCCL、autograd 或 CUDA 错误时保留原始日志，不得静默修复后继续混用结果。
- 当前 inline profiler 的组件时间来自 rank 0 本地 CUDA event；只有 peak allocated/reserved 是 worst-rank 归约值。报告中必须保留该口径，不得称为 worst-rank 模块时间。

## 3. 简单任务计划表

| 阶段 | 优先级 | Agent 操作 | 验收条件 | 失败处理 |
|---|---|---|---|---|
| T0 版本冻结 | P0 | 记录 commit、`git status`、脚本 SHA256、GPU/软件版本和拓扑 | 元数据齐全；脚本包含 temporal CLI 和核心函数 | 若脚本不是新版，停止并写 `NEEDS_RESEARCHER.md` |
| T1 静态检查 | P0 | 执行 `py_compile`、`--help`，检查 CLI | 编译通过；help 中出现 `micro-batch-size`、`temporal-strategy` | 失败即停止，不改代码 |
| T2 Config A 冒烟 | P0 | 8 GPU，seq=4096，CP4→1，mbs=1，gbs=8，bundle+repeat | 四条 temporal fixed/mixed 路径均完成；GA steps=4；无 NaN/异常 | 同命令最多重试1次；确定性失败停止 |
| T3 Config A 主测 | P0 | seq=32768、65536；先各1次筛选，65536 成功后做3次独立 launch | 每次完整结束；记录 E2E、executed/unique tokens/s、allocated/reserved | OOM 记录为有效边界，不降配置伪装成功 |
| T4 Config B 次测 | P1 | seq=65536，CP8→2，mbs=1，gbs=8；先1次 | GA steps=8；完整输出 bundle/repeat 与 fixed/mixed 对比 | 若 mixed 相对 fixed ≤1.03×，不再追加重复 |
| T5 逐模块 profiling | P1 | 用现有 inline forward/backward profiler 对 A、B 的 spatial fixed/mixed 逐模块计时 | 每个 layer/transition 同时有 fwd/bwd mean、std；另含 loss、grad sync、residual | profiler 事件缺失或数量不符则该 launch 作废，保留日志 |
| T6 Spatial oracle | P1 | 复用 T5 中 `temporal-strategy=none` 的 fixed/mixed E2E；仅在 T5 无有效结果时补跑 | 获得同环境 spatial fixed/mixed 上界，且不新增冗余 launch | 仅作上界，不与 temporal 语义混写 |
| T7 汇总审计 | P0 | 生成 CSV 和 `REPORT.md`，核对命令数、退出码、日志数 | 报告可追溯到每条原始日志；bundle/repeat/module profile 分栏 | 禁止删除失败日志或只报最好结果 |

## 4. 命令模板

执行 agent 应先进入包含 `bench_fixed_mixed.py` 且能导入 Megatron-Core 的工作目录，并设置项目实际需要的 `PYTHONPATH`。

### 4.1 版本与环境

```bash
set -euo pipefail

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_tmf_proto"
RUN_DIR="agent_run/${RUN_ID}"
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/metadata"

git rev-parse HEAD > "${RUN_DIR}/metadata/git_commit.txt"
git status --short > "${RUN_DIR}/metadata/git_status.txt"
sha256sum bench_fixed_mixed.py > "${RUN_DIR}/metadata/bench_sha256.txt"
nvidia-smi -q > "${RUN_DIR}/metadata/nvidia_smi_q.txt"
nvidia-smi topo -m > "${RUN_DIR}/metadata/topology.txt"
python - <<'PY' > "${RUN_DIR}/metadata/software.txt"
import torch
print("python/torch environment loaded")
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("nccl", torch.cuda.nccl.version())
PY

python -m py_compile bench_fixed_mixed.py
python bench_fixed_mixed.py --help > "${RUN_DIR}/metadata/help.txt"
rg -n "temporal-strategy|def temporal_remesh_payloads|def run_temporal_plan" \
  bench_fixed_mixed.py > "${RUN_DIR}/metadata/temporal_code_check.txt"
```

如果脚本实际位于 `heterogeneous_cp/bench_fixed_mixed.py`，统一替换命令路径，不复制出第二份代码。

### 4.2 Config A 冒烟

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
grad_accumulation_steps=4
Bundle vs repeat
```

### 4.3 Config A 主测

将下列命令的 `SEQ` 依次设置为 `32768`、`65536`：

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

`seq=65536` 首次成功后，设置 `LAUNCH=2,3` 再做两个独立进程 launch。不要在同一进程内把迭代当作独立 launch。

### 4.4 Config B 次测

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

预期 `grad_accumulation_steps=8`。若 temporal mixed 相对对应 fixed 的 E2E speedup 不超过 `1.03×`，停止 Config B 的额外 launch。

### 4.5 逐模块 forward/backward profiling

该任务只使用已有 inline profiler，不修改 benchmark。使用 legacy spatial 路径，是因为当前 temporal executor 是多 payload 分支图，尚未接入单链 boundary-hook profiler。

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

每个 mixed launch 的 profiler 事件应满足：

- 8 个 layer forward：2 Attention + 6 GDN；
- 3 个 transition forward；
- 与上述 11 个 component 一一对应的 backward 时间；
- 1 个 `loss bwd`、1 个 `grad sync (all_reduce)`；
- `bwd_residual` 和 E2E decomposition；
- 所有 mean/std 均为有限非负数。

fixed plan 应有 8 个 layer forward/backward、0 个 transition。Agent 可以在 `${RUN_DIR}/scripts/` 新建日志解析脚本，但不得修改 benchmark。

输出两个层级：

1. `module_profile.csv`：每个 layer/transition 一行，保留 index、类型、CP、fwd/bwd mean 与 std；
2. `module_profile_summary.csv`：按 launch 汇总 Attention、GDN、transition、loss backward、grad sync、residual 和 E2E。

解释边界：组件时间来自 rank 0 CUDA events；显存仍使用 benchmark 的 worst-rank allocated/reserved。不得将 rank 0 模块时间写成集群最慢 rank 时间。

### 4.6 Spatial oracle

T5 的逐模块 profiling 已使用 `--mode both --temporal-strategy none`，且 profiler 位于真实 measured fwd+bwd 内，没有额外 stack pass。因此默认直接复用 T5 的 fixed/mixed E2E 作为 spatial oracle，不再重复启动相同任务。

只有当 T5 某配置没有有效 launch 时，才按相同参数补跑一次 `--temporal-strategy none`。该结果表示把完整 global batch 一次性放进空间转换路径的乐观上界，不是 TMF 结果。

## 5. 必须输出的结果表

`summary.csv` 至少包含：

```text
run_id,launch_id,config,seq_length,strategy,plan,
global_batch_size,micro_batch_size,grad_accumulation_steps,
both_ms,both_std_ms,executed_tokens_s,unique_tokens_s,
peak_alloc_gib,peak_reserved_gib,speedup_vs_matching_fixed,
exit_code,status,log_path
```

其中：

- `strategy ∈ {bundle, repeat, spatial_oracle}`；
- `plan ∈ {fixed, mixed}`；
- repeat 的 `unique_tokens_s` 必须使用唯一数据量，不能复制 executed throughput；
- speedup 只能在相同 strategy、相同 global batch、相同序列长度的 fixed/mixed 间计算。

`module_profile.csv` 至少包含：

```text
run_id,launch_id,config,plan,schedule,component_kind,
component_index,module_type,cp_size,src_cp,dst_cp,
fwd_mean_ms,fwd_std_ms,bwd_mean_ms,bwd_std_ms,
profiler_rank,log_path,status
```

`module_profile_summary.csv` 至少包含：

```text
run_id,launch_id,config,plan,
attention_fwd_ms,attention_bwd_ms,gdn_fwd_ms,gdn_bwd_ms,
transition_fwd_ms,transition_bwd_ms,loss_bwd_ms,
grad_sync_ms,bwd_residual_ms,e2e_ms,profiler_rank,status
```

验收时逐 launch 检查事件数量，不允许用缺失事件的均值补零。组件总和与 E2E 的差异必须显式保留在 `bwd_residual_ms`，不得将 residual 分摊到 Attention 或 GDN。

## 6. 报告模板

远程 agent 最终生成 `REPORT.md`，依次包含：

1. Run identity：commit、dirty files、硬件、软件、起止时间；
2. Completion matrix：每阶段计划数、成功数、失败数；
3. Config A：bundle fixed/mixed、repeat fixed/mixed、三次 launch 的原始值与 median；
4. Config B：一次筛选结果及是否继续的判定；
5. Module profiling：逐 layer/transition 表，以及 Attention、GDN、transition、loss、grad sync、residual 汇总；
6. Spatial oracle：仅作为 temporal 上界；
7. 显存：worst-rank allocated/reserved；
8. 失败与重试：原始退出码、错误类别、日志；
9. Go/No-Go：是否值得进入 checkpoint/recompute 原型。

## 7. Go/No-Go 判据

- **功能 Go**：Config A smoke 四条路径全部完成，无 shape/NCCL/autograd 错误。
- **性能 Go**：Config A 的 temporal bundle mixed 相对 temporal bundle fixed 在 seq=65536 的三次 launch median ≥`1.10×`，且 peak allocated 不高于 fixed。
- **checkpoint Go**：若功能通过但因保留多个高 CP graph 导致 OOM 或峰值显存明显恶化，结论不是 TMF 失败，而是进入 boundary checkpoint/recompute 阶段。
- **性能 No-Go**：Config A median <`1.05×`，停止扩展 Config B/C、PP 和 planner，先做成本分解。
- **repeat 判读**：repeat 只有在 unique tokens/s 和训练语义代价都被明确报告时才可称为冗余解决方案；不得仅凭 executed tokens/s 宣称训练加速。

## 8. 给远程 agent 的最终指令

按 T0→T7 顺序执行。先完成 Config A smoke，再运行任何长序列任务。不得修改 benchmark；若代码缺失、确定性失败或结果字段不足，写入 `NEEDS_RESEARCHER.md` 后停止对应阶段。交付整个 `${RUN_DIR}`，包括所有成功和失败日志、元数据、`summary.csv`、`REPORT.md` 与实际执行命令。