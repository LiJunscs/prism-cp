# 知识库主题：DSv4 CSA/HCA Contiguous-THD 甜点 CP 远程 Agent 执行计划

> 日期：2026-09-03  
> 执行对象：远程单机 8×H800 GPU agent  
> HCP 仓库入口：`bench_fixed_mixed_csa_hca.py`  
> Megatron-LM 基线：远程当前 checkout 的最新 `dev`；不得由执行 agent 自行更新  
> 研究定位：为 unified heterogeneous context semantics 收集第一组真实算子证据  
> 本轮范围：CSA/HCA operator-only CP 甜点与简单 mixed-stack 机会验证

## 1. 本轮只回答的问题

本轮回答三个问题：

1. 最新 Megatron-LM DSv4 CSA（compression ratio 4）和 HCA（compression ratio 128）在固定 global batch、等长 THD 数据上的可行 CP 集合与各自甜点 CP 是什么？
2. CSA/HCA 的甜点 CP 是否显著不同，足以构成 heterogeneous CP opportunity？
3. 在 contiguous THD 的等长特例中，利用 metadata-only CP transition 的 CSA/HCA mixed stack，能否稳定优于所有 homogeneous fixed-CP baseline？

本轮不回答：

- 完整训练 step、optimizer 或收敛；
- indexer auxiliary loss 对性能和梯度的影响；
- data-parallel gradient synchronization 成本；
- ragged THD、padding-free scheduler 或在线动态长度；
- PP、TMF、checkpoint/recompute、多节点；
- GDN/GQA 混合架构；
- 统一 Context Semantics IR 的最终实现。

当前 benchmark 明确排除了 DP gradient sync、optimizer 和 CSA indexer auxiliary loss。因此任何性能结果只能表述为 **operator-only heterogeneous CP opportunity**，不得表述为完整训练加速。

## 2. Agent 角色、权限和禁止事项

你是实验执行与证据整理 agent，不是算法开发 agent。

允许：

- 读取两个仓库、运行官方测试和 benchmark；
- 在新建的 `agent_run/<RUN_ID>/` 内创建命令包装、日志解析器、CSV 和报告；
- 修复你在运行目录内新建的编排或解析脚本；
- 对互相独立的配置继续执行并保留失败边界。

禁止：

- 不得执行 SHA256、MD5、BLAKE 或其他文件内容哈希/校验和检查；不得生成 checksum 文件；
- 不得修改、pull、checkout、rebase 或 reset Megatron-LM；
- 不得修改 `bench_fixed_mixed_csa_hca.py` 或其他 HCP 核心代码；
- 不得清理两个仓库的 dirty worktree，不得覆盖用户已有改动；
- 不得放宽 Megatron 官方 correctness gate；
- 不得通过减少 GPU、缩短指定 sequence、替换模型 profile 或关闭 fused kernel 把失败配置伪装为成功；
- 不得删除失败日志、只保留最快 launch 或跨口径合并旧 P2 数据；
- 不得 commit、push、创建 PR 或杀死非本任务创建的进程。

如果继续执行必须修改 benchmark 或 Megatron-LM，将完整错误、触发命令和建议写入 `NEEDS_RESEARCHER.md`，停止受影响阶段；其他独立阶段可继续。

## 3. 数据与 Transition 语义

本轮输入固定为：

- THD 格式；
- 每条样本长度完全相同；
- fixed global batch；
- global rank 连续组成 CP group；
- world size=8，CP 候选为 1、2、4、8；
- 每个合法布局满足 `global_batch_size % (world_size / CP) == 0`。

在这些条件下，每个 rank 的本地 THD 行数恒为：

```text
local_rows = global_batch_size * seq_length / world_size
```

并且 global rank `r` 在任意合法 CP 下拥有同一个 flattened-token interval。因此 `EqualLengthTHDContiguousTransition` 不执行 P2P activation redistribution，只验证 source/destination ownership，并让下一层使用新的 `PackedSeqParams.local_cp_size` 和 `cp_group`。

这只是等长 contiguous-THD 特例。报告不得把它推广为 ragged THD 或任意 rank mapping 下的零通信 transition。

## 4. 执行目录与元数据

使用实际路径覆盖以下变量：

```bash
export HCP_REPO="${HCP_REPO:-/workspace/repos/heterogeneous_cp}"
export MEGATRON_REPO="${MEGATRON_REPO:-/workspace/repos/Megatron-LM}"
export PYTHONPATH="${MEGATRON_REPO}:${HCP_REPO}:${PYTHONPATH:-}"

HCP_SHORT="$(git -C "${HCP_REPO}" rev-parse --short HEAD)"
MCORE_SHORT="$(git -C "${MEGATRON_REPO}" rev-parse --short HEAD)"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_${HCP_SHORT}_${MCORE_SHORT}_dsv4_csa_hca_thd_cp"
RUN_DIR="${HCP_REPO}/agent_run/${RUN_ID}"
mkdir -p \
  "${RUN_DIR}"/{metadata,static,correctness,smoke,summaries,scripts} \
  "${RUN_DIR}"/isolated/{screening,confirmation} \
  "${RUN_DIR}"/mixed/{screening,confirmation}
```

创建唯一运行目录：

```text
agent_run/<UTC>_<hcp-short-commit>_<mcore-short-commit>_dsv4_csa_hca_thd_cp/
├── metadata/
├── static/
├── correctness/
├── smoke/
├── isolated/
│   ├── screening/
│   └── confirmation/
├── mixed/
│   ├── screening/
│   └── confirmation/
├── summaries/
├── scripts/
├── commands.tsv
├── RUN_STATUS.tsv
├── NEEDS_RESEARCHER.md       # 仅有阻塞时创建
└── REPORT.md
```

`metadata/` 至少记录：

- 两个仓库的 absolute path、branch、commit identifier、`git status --short`、最近 5 个 commit；
- 两个仓库的 dirty diff 文本；未跟踪文件只记录路径，不读取或复制无关内容；
- hostname、容器/作业标识、UTC 起止时间；
- 8 张 GPU 的型号、显存、driver、拓扑与任务前空闲状态；
- Python、PyTorch、CUDA、NCCL、Transformer Engine、flash-MLA 和 cuDNN DSA 可用性；
- 非敏感的 `CUDA*`、`NCCL*`、`NVTE*`、`PYTHONPATH` 环境变量；
- `bench_fixed_mixed_csa_hca.py --help` 输出。

只记录 Git commit identifier 和文本 diff；不得计算源码 checksum。

`commands.tsv` 每个独立进程一行：

```text
command_id,phase,case,command,start_utc,end_utc,wall_seconds,exit_code,status,log_path
```

## 5. 阶段总表

| 阶段 | 优先级 | 内容 | 进入条件 | 退出条件 |
| --- | ---: | --- | --- | --- |
| P0 | P0 | 环境与版本冻结 | 无 | 8×H800 独占、依赖齐全、无 checksum |
| P1 | P0 | 静态与 Transition gate | P0 | 语法、CLI、两组 ownership/autograd 验证通过 |
| P2 | P0 | Megatron 官方 DSv4 CP correctness | P1 | CSA/HCA output、hidden grad、parameter grad gate 通过 |
| P3 | P0 | 8 GPU benchmark smoke | P2 | unfused tiny 与 fused flash mixed smoke 完成 |
| P4 | P0 | CSA/HCA isolated CP screening | P3 | CP1/2/4/8×32K 有完整结果/明确失败 |
| P5 | P1 | isolated 长序列确认 | P4 | 32K/64K 候选各 3 个独立 launch |
| P6 | P1 | mixed stack screening/确认 | P5 且存在异构机会 | mixed 与所有 homogeneous CP 公平比较 |
| P7 | P0 | 审计与报告 | P4；P5/P6 可条件缺省 | 原始日志、CSV、结论边界可追溯 |

严格按 P0→P7 执行。P2 未通过时不得进入性能阶段。

## 6. P0：环境与版本冻结

进入两个仓库分别记录 branch/commit/status/diff，不切换版本。检查：

```bash
nvidia-smi -L
nvidia-smi --query-gpu=index,name,memory.total,memory.used,temperature.gpu,pstate --format=csv
nvidia-smi topo -m

python - <<'PY'
import importlib
import torch

print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
print("nccl", torch.cuda.nccl.version())
for name in ("transformer_engine", "flash_mla", "cudnn"):
    try:
        module = importlib.import_module(name)
        print(name, "AVAILABLE", getattr(module, "__version__", "unknown"))
    except Exception as exc:
        print(name, "UNAVAILABLE", type(exc).__name__, str(exc))

try:
    from cudnn import DSA
    print("cudnn.DSA", "AVAILABLE")
except Exception as exc:
    print("cudnn.DSA", "UNAVAILABLE", type(exc).__name__, str(exc))
PY
```

P0 硬门禁：

- 必须恰好看到 8 张目标 H800，且无外部计算进程占用；
- PyTorch 必须支持 NCCL；
- Transformer Engine、flash-MLA、cuDNN DSA 必须可导入；
- Megatron-LM 必须存在 DSv4 hybrid module spec、contiguous THD CP 和 `linear_cp_mode` 字段；
- HCP 仓库必须存在 `bench_fixed_mixed_csa_hca.py`。

任一依赖缺失时创建 `NEEDS_RESEARCHER.md` 并停止，不自行安装或更新。

## 7. P1：静态检查与 Transition gate

在 HCP 仓库执行：

```bash
python -m py_compile bench_fixed_mixed_csa_hca.py
python bench_fixed_mixed_csa_hca.py --help \
  > "${RUN_DIR}/static/bench_help.txt" 2>&1

python bench_fixed_mixed_csa_hca.py \
  --seq-length 4096 \
  --global-batch-size 8 \
  --num-csa-layers 2 --num-hca-layers 2 \
  --csa-cp-size 2 --hca-cp-size 8 \
  --mode both --validate-only --validation-world-size 8 \
  > "${RUN_DIR}/static/transition_cp2_cp8.txt" 2>&1

python bench_fixed_mixed_csa_hca.py \
  --seq-length 384 \
  --global-batch-size 12 \
  --num-csa-layers 2 --num-hca-layers 2 \
  --csa-cp-size 3 --hca-cp-size 4 \
  --mode both --validate-only --validation-world-size 12 \
  > "${RUN_DIR}/static/transition_cp3_cp4_nonnested.txt" 2>&1
```

验收：

- 语法与 help 退出码为 0；
- CP2→CP8 和非嵌套 CP3→CP4 均打印 fixed/mixed `validated`；
- local rows 保持不变；
- identity forward 与 backward grad 检查通过；
- `git diff --check` 不报告本轮代码问题。

## 8. P2：Megatron 官方 DSv4 CP correctness gate

从 Megatron-LM 根目录运行当前 checkout 自带测试，不复制测试文件、不改 gate。先用 4 GPU 覆盖 CP2/CP4：

```bash
timeout 60m torchrun --standalone --nproc_per_node=4 -m pytest -q -s \
  tests/unit_tests/transformer/experimental_attention_variant/test_dsv4_hybrid_attention_cp.py \
  -k "test_thd_cp_matches_full_reference_forward_backward and (ratio_4_indexer_sparse or ratio_128_compressor) and fused_rope" \
  > "${RUN_DIR}/correctness/dsv4_csa_hca_cp_reference.log" 2>&1

timeout 30m torchrun --standalone --nproc_per_node=4 -m pytest -q -s \
  tests/unit_tests/transformer/experimental_attention_variant/test_dsv4_hybrid_attention_cp.py \
  -k "test_thd_cp_zero_indexer_loss_keeps_indexer_grads" \
  > "${RUN_DIR}/correctness/dsv4_zero_aux_loss_grad.log" 2>&1
```

验收：

- 选中的 ratio-4 CSA 与 ratio-128 HCA case 至少各有 CP2/CP4 PASS；
- output、hidden gradient、reduced parameter gradient 对 CP1 reference 的官方 gate 全部通过；
- zero auxiliary-loss case 中 indexer parameters 仍有 gradient；
- 无 xfail、异常 skip、OOM、timeout、NCCL error 或 nonfinite。

如果 `-k` 表达式因当前 pytest 参数 ID 不匹配而没有收集到 case，可以只调整 `-k` 选择表达式并记录调整前后命令；不得改测试代码。

## 9. P3：8 GPU benchmark smoke

### P3.1 Unfused tiny mixed smoke

```bash
timeout 30m torchrun --standalone --nproc_per_node=8 \
  "${HCP_REPO}/bench_fixed_mixed_csa_hca.py" \
  --megatron-path "${MEGATRON_REPO}" \
  --seq-length 128 --global-batch-size 8 \
  --num-csa-layers 1 --num-hca-layers 1 \
  --csa-cp-size 2 --hca-cp-size 8 --fixed-cp-size 8 \
  --mode both --profile tiny \
  --dsa-kernel-backend none --no-rope-fusion \
  --warmup 0 --iters 1 \
  > "${RUN_DIR}/smoke/tiny_unfused_cp2_cp8.log" 2>&1
```

### P3.2 Fused flash-shape mixed smoke

```bash
timeout 45m torchrun --standalone --nproc_per_node=8 \
  "${HCP_REPO}/bench_fixed_mixed_csa_hca.py" \
  --megatron-path "${MEGATRON_REPO}" \
  --seq-length 4096 --global-batch-size 8 \
  --num-csa-layers 1 --num-hca-layers 1 \
  --csa-cp-size 2 --hca-cp-size 8 --fixed-cp-size 8 \
  --mode both --profile flash \
  --dsa-kernel-backend cudnn --rope-fusion \
  --warmup 1 --iters 2 \
  > "${RUN_DIR}/smoke/flash_fused_cp2_cp8.log" 2>&1
```

成功日志必须出现：

- scope 声明；
- `CSA-HCA` schedule；
- fixed `DP1/CP8-DP1/CP8`；
- mixed `DP4/CP2-DP1/CP8`；
- `metadata-only transitions=1`；
- fixed/mixed 的有限、正数 latency、tokens/s 和 memory；
- 退出码 0，无 traceback、OOM、NCCL error。

P3 只验证 benchmark 接通；数值正确性由 P2 官方测试提供，不从 smoke latency 推导。

## 10. P4：CSA/HCA isolated CP screening

公共配置：

```text
world_size=8
global_batch_size=8
profile=flash
dsa_kernel_backend=cudnn
rope_fusion=true
seq_length=4096,32768
CP=1,2,4,8
warmup=5
iters=10
independent launches=1
```

每个 operator/sequence/CP 使用独立 `torchrun` 进程，共 16 个 screening launch。模板：

```bash
# CSA：OP=csa, NUM_CSA=1, NUM_HCA=0
# HCA：OP=hca, NUM_CSA=0, NUM_HCA=1

timeout 45m torchrun --standalone --nproc_per_node=8 \
  "${HCP_REPO}/bench_fixed_mixed_csa_hca.py" \
  --megatron-path "${MEGATRON_REPO}" \
  --seq-length "${SEQ}" --global-batch-size 8 \
  --num-csa-layers "${NUM_CSA}" --num-hca-layers "${NUM_HCA}" \
  --csa-cp-size "${CP}" --hca-cp-size "${CP}" \
  --mode mixed --profile flash \
  --dsa-kernel-backend cudnn --rope-fusion \
  --warmup 5 --iters 10 \
  > "${RUN_DIR}/isolated/screening/${OP}_seq${SEQ}_cp${CP}.log" 2>&1
```

执行顺序按每个 sequence 交错 CP，避免所有低 CP 或高 CP 集中在同一温度阶段：

```text
round 1: CP1, CP8, CP2, CP4
round 2: CP4, CP2, CP8, CP1
```

4096 使用 round 1，32768 使用 round 2；CSA/HCA 交替执行。

每个 launch 提取：

```text
phase,operator,profile,seq_length,cp_size,dp_size,global_batch_size,
local_rows,warmup,iters,mean_ms,std_ms,tokens_per_second,
peak_alloc_gib,peak_reserved_gib,status,exit_code,log_path
```

状态只能为：

```text
PASS,OOM,TIMEOUT,NCCL_ERROR,DEPENDENCY_ERROR,CONFIG_ERROR,CRASH,PARSE_ERROR
```

OOM 是有效可行域边界，不降低 sequence、profile 或 batch 冒充原配置成功。NCCL 明确瞬时错误最多原命令重试 1 次，两个日志都保留。

## 11. P5：isolated 长序列确认与甜点定义

P4 后，对每个 operator 选择：

- 32K 最快 CP；
- 与最快值相差不超过 5% 的其他 CP；
- 最小显存的可行 CP；
- CP8 固定参考。

去重后形成候选集。每个候选在 `seq=32768,65536` 下各执行 3 个独立 launch：

```text
warmup=10
iters=20
launches=3
```

同一轮候选执行顺序交错，下一轮反序。不要把同一进程内 iteration 当作独立 launch。

甜点判定：

1. 配置必须 3/3 launch PASS；
2. 以每个 launch 的 `mean_ms` 为原始值，跨 launch 使用 median 排序；
3. 同时报告 min/max、IQR、每个 launch 内 `std_ms / mean_ms`；
4. 比最快值慢不足 3% 的配置视为同一甜点集合，不强行排序；
5. 性能相近时优先保留 peak allocated 更低、长序列可行域更宽的 CP；
6. 32K 与 64K 最优 CP 不同则报告 crossover，不压成单一甜点。

异构机会进入 P6 的条件：

- CSA 与 HCA 的甜点集合不完全重合；并且
- 用 isolated median 按 mixed stack 的层数加权后，operator-specific CP 相对最佳共同 CP 的预测优势至少 5%；并且
- 优势不是由单个异常 launch 或 OOM 后改变工作量产生。

若预测优势 `<5%`，跳过 P6 confirmation，只做最多一个 mixed screening 验证预测并进入 P7。

## 12. P6：CSA/HCA mixed stack

固定 4 层、seed 1234：

```text
CSA-HCA-CSA-HCA
```

比较集合：

1. homogeneous fixed CP1；
2. homogeneous fixed CP2；
3. homogeneous fixed CP4；
4. homogeneous fixed CP8；
5. mixed：CSA 使用 P5 选定 CP，HCA 使用 P5 选定 CP。

### P6.1 Screening

在 32K 和 64K 各做 1 个独立 launch，`warmup=5,iters=10`。homogeneous 模板：

```bash
timeout 60m torchrun --standalone --nproc_per_node=8 \
  "${HCP_REPO}/bench_fixed_mixed_csa_hca.py" \
  --megatron-path "${MEGATRON_REPO}" \
  --seq-length "${SEQ}" --global-batch-size 8 \
  --num-csa-layers 2 --num-hca-layers 2 --layer-seed 1234 \
  --csa-cp-size "${FIXED_CP}" --hca-cp-size "${FIXED_CP}" \
  --fixed-cp-size "${FIXED_CP}" --mode fixed \
  --profile flash --dsa-kernel-backend cudnn --rope-fusion \
  --warmup 5 --iters 10 \
  > "${RUN_DIR}/mixed/screening/fixed_cp${FIXED_CP}_seq${SEQ}.log" 2>&1
```

mixed 模板：

```bash
timeout 60m torchrun --standalone --nproc_per_node=8 \
  "${HCP_REPO}/bench_fixed_mixed_csa_hca.py" \
  --megatron-path "${MEGATRON_REPO}" \
  --seq-length "${SEQ}" --global-batch-size 8 \
  --num-csa-layers 2 --num-hca-layers 2 --layer-seed 1234 \
  --csa-cp-size "${CSA_CP}" --hca-cp-size "${HCA_CP}" \
  --mode mixed \
  --profile flash --dsa-kernel-backend cudnn --rope-fusion \
  --warmup 5 --iters 10 \
  > "${RUN_DIR}/mixed/screening/mixed_csa${CSA_CP}_hca${HCA_CP}_seq${SEQ}.log" 2>&1
```

### P6.2 Confirmation

只有 mixed screening 相对该 sequence 的最快可行 homogeneous baseline ≥3% 时继续：

- 最快 homogeneous baseline 与 mixed 各补到 5 个独立 launch；
- `warmup=10,iters=20`；
- 按 round 交错，奇数轮先 fixed，偶数轮先 mixed；
- 同时保留 CP8 作为架构常用高 CP reference，但 speedup 分母必须是实测最快 homogeneous CP，而不固定为 CP8。

性能 Go：

- mixed 5/5 PASS；
- mixed 相对最快 homogeneous baseline 的 launch-median speedup ≥1.05×；
- 5 轮 paired speedup 至少 4/5 同方向；
- mixed peak allocated 不超过最快 homogeneous baseline 的 1.05×，或增加部分有明确、可接受的机制解释；
- 32K/64K 至少一个长度满足，另一个长度不得出现无法解释的严重反转。

强性能 Go：上述条件成立且 median speedup ≥1.10×。

如果优势为 3%–5%，标记 `WEAK_OPPORTUNITY`，不进入大规模系统实现；先做口径补全。如果 `<3%` 或方向不稳定，标记 `NO_ROBUST_MIXED_GAIN`。

## 13. 结果文件

`summaries/isolated_launches.csv` 每个 launch 一行，字段至少为：

```text
run_id,command_id,operator,seq_length,cp_size,dp_size,global_batch_size,
profile,kernel_backend,rope_fusion,warmup,iters,mean_ms,std_ms,
tokens_per_second,peak_alloc_gib,peak_reserved_gib,status,exit_code,log_path
```

`summaries/isolated_aggregate.csv`：

```text
operator,seq_length,cp_size,valid_launches,median_ms,min_ms,max_ms,iqr_ms,
median_tokens_per_second,max_peak_alloc_gib,max_peak_reserved_gib,
relative_to_fastest,status
```

`summaries/mixed_launches.csv`：

```text
run_id,command_id,seq_length,plan,fixed_cp,csa_cp,hca_cp,schedule,
transition_count,mean_ms,std_ms,tokens_per_second,peak_alloc_gib,
peak_reserved_gib,status,exit_code,log_path
```

`summaries/mixed_comparison.csv`：

```text
seq_length,csa_cp,hca_cp,best_fixed_cp,fixed_valid,mixed_valid,
fixed_median_ms,mixed_median_ms,speedup,paired_wins,
fixed_peak_alloc_gib,mixed_peak_alloc_gib,status
```

所有 summary 行必须能回溯到 `commands.tsv` 和原始 stdout。解析失败时保留日志并标记 `PARSE_ERROR`，不得以人工猜值填充。

## 14. REPORT.md 固定结构

1. Executive summary：P0–P7 完成度、性能 Go/No-Go；
2. Run identity：两个仓库、dirty 状态、硬件、软件、起止时间；
3. Scope：operator-only、等长 contiguous THD、无 DP sync/optimizer/aux loss；
4. Official correctness：CSA/HCA CP2/CP4 reference 结果；
5. Transition gate：CP2→8 与非嵌套 CP3→4 ownership/autograd；
6. Isolated screening：CSA/HCA 的 CP1/2/4/8 可行域；
7. Isolated confirmation：每个长度的甜点集合与 crossover；
8. Mixed comparison：所有 homogeneous CP 与 mixed 的逐 launch/aggregate；
9. Memory：worst-rank allocated/reserved 与 OOM 边界；
10. Stability：launch 内 CV、跨 launch IQR、paired wins；
11. Failures/retries：原始退出码、错误分类和日志；
12. 可以得出的结论；
13. 不能得出的结论；
14. Go/No-Go 与最多 3 条建议。

报告禁止：

- 将旧 `agent_run/`、`bench_data/` 的 headwise P2 数值合并到新曲线；
- 把 metadata-only transition 表述为一般 THD 零通信；
- 把无 DP sync 的 operator-only speedup 写成训练 step speedup；
- 用 CP8 作为唯一分母而忽略更快的 homogeneous CP；
- 用最好单次 launch 替代 launch median。

## 15. 停止与重试规则

- P0 依赖或 GPU 独占失败：停止全部 GPU 阶段；
- P1 Transition gate 失败：停止并请求研究者检查 benchmark；
- P2 官方 correctness 失败：停止全部性能阶段；
- P3 smoke 失败：保留 unfused/fused 各自状态；确定性代码失败不修改核心代码；
- OOM：记录边界，不重试、不缩小原 case；
- NCCL transient：确认只清理本任务 PID 后，原命令最多重试 1 次；
- timeout：原命令最多重试 1 次；再次失败后继续其他独立配置；
- 单个 isolated case 失败：不阻塞其他 CP，但该 case 不进入甜点集合；
- mixed 无预测优势：允许按 P5 规则提前结束，不展开确认矩阵；
- 不允许无限重试。

## 16. 本轮完成条件

任务完成不要求获得性能 Go。完成条件是：

- P0/P1 元数据和静态 gate 完整，且没有执行任何 checksum；
- Megatron 官方 CSA/HCA CP correctness 有明确结果；
- 16 个 isolated screening case 均有 PASS 或明确失败；
- 每个 operator 至少形成一个可行 CP 集合和甜点/不可判定结论；
- mixed 阶段按 gate 执行或有合法跳过理由；
- 所有命令、日志、CSV、失败和结论边界可追溯；
- 未修改两个仓库的核心代码。

Agent 最终只交付完整 `${RUN_DIR}`，不要自动提交仓库。

## 17. 性能 Go 之后的后续接口

如果 P6 获得性能 Go，本轮 agent 不继续实现系统，只在 `REPORT.md` 中输出以下后续输入：

1. `{operator, sequence, sweet_cp_set, latency, memory}` 表；
2. 最快 homogeneous 与 mixed 的逐 launch 原始值和 paired speedup；
3. mixed plan 的 layer sequence、CP sequence 和 transition count；
4. 当前结果缺失的训练口径：DP sync、optimizer、auxiliary loss、ragged THD；
5. 对下一阶段最值得优先补齐的一个口径建议。

后续系统路线由研究者决定，不由本轮 agent 自动扩展。
