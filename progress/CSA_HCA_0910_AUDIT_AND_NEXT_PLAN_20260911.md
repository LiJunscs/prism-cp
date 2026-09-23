# 知识库主题：CSA/HCA 0910 批次证据审计与下一轮实验计划

> **2026-09-11 用户纠正后的替代入口：** [固定全局负载与极限显存下的 Mixed CP 计划](CSA_HCA_MEMORY_CONSTRAINED_PLAN_20260911.md)。本文的原始数据重算及 teacher-LSE 观测保留；以显存宽裕时的 CP1 为主裁决、将 teacher 修复设为 mixed 研究的前置依赖，以及过度核对 commit 的建议均已撤回。后续以共同显存预算内的最佳可行 fixed CP 为基线，研究 mixed 的速度收益与峰值增量。

日期：2026-09-11。范围：阅读最新 roadmap、0910 原始结果、已有 profiler trace 和同版本本地代码，重算统计，提出下一轮计划。本次没有运行新增 GPU 实验，没有修改训练实现或旧记录。

项目根目录：`E:/LLM论文/mine/heterogeneous_cp`。本文相对路径均相对于该目录；知识库副本也使用此路径约定。Megatron 代码位置：`E:/Project/Pythonproject/Megatron-LM`。

## 1. 本轮结论

**CSA 的 CP>1 异常已获得可操作的 teacher-LSE 证据，下一步优先做 λ=0 的受限等价消融。现有 mixed 结果仍未证明胜过最佳已测统一布局，不应据此恢复 cohort/TMF 优化主线。128K FSDP 统一训练的稳定基线仍未完成，需要作为另一条工作线补齐。**

这延续总 roadmap 顶部 2026-09-10 更新及 `NEXT_EXPERIMENT_EXECUTION_PLAN_20260910.md` 的顺序：先证据、统一基线和瓶颈，再选择受限机制。更新的是 CSA 归因的完成程度，而不是重新启动 planner、IR、headwise 或 cohort 大规模开发。

本次采用的证据优先级：原始日志/JSON trace > 同版本源码 > 汇总表 > 手工总结中的归因。`agent_run/CSA_HCA_MIXED_CP_RESULTS_20260910.md` 与 `analysis/cp_scalability_20260910/P1C_FUSED_PATH_ANALYSIS.md` 中部分解释需要修正，详见第 5 节。

## 2. 单算子 R1：性能变化成立，最优布局解释需要限定

来源：`agent_run/20260910T161757Z_p1a_csa_hca_r1_refresh/summary.tsv`。8×H800，S=65536，B=1，A=1，D=8/C；每格 2 次独立 launch，16 次均记录 rc=0。下表是 launch 内平均前反向时间再跨 launch 取中位数。

| 算子 | CP1 ms | CP2 ms | CP4 ms | CP8 ms |
|---|---:|---:|---:|---:|
| CSA | 191.440 | 638.325 | 534.745 | 480.260 |
| HCA | 125.480 | 70.445 | 36.345 | 19.990 |

| 算子 | CP1 峰值 GiB | CP2 峰值 GiB | CP4 峰值 GiB | CP8 峰值 GiB |
|---|---:|---:|---:|---:|
| CSA | 24.30 | 27.03 | 13.73 | 7.07 |
| HCA | 22.76 | 12.11 | 6.29 | 3.37 |

结论边界：

- 当前 λ=0 实现下，CSA CP2 比 CP1 慢约 3.33 倍，显存反而高约 11.2%；值得优先处理。
- HCA 的单样本延迟从 CP1 到 CP8 加速 6.277 倍，但每样本资源时间 C×t 从 125.48 增至 159.92 GPU-ms。固定 8 卡预算下，该测量口径的吞吐比为 0.7846，不能由此宣布 HCA 的吞吐最优点是 CP8。
- CSA 的 CP8 每样本资源时间为 3842.08 GPU-ms，约为 CP1 的 20.07 倍。这是当前实现和当前目标下的现象，不能写成 CSA 固有属性。
- R1 各 CP 的全机独立样本数为 8/4/2/1，不能直接把不同 CP 的延迟相加，预测固定 G 的 mixed 前反向时间。
- 0904→0910，CSA CP1 从 552.480 降至 191.440 ms；HCA CP1 也从 140.595 降至 125.480 ms，降幅约 10.75%。因此“除 CSA CP1 外全部在 1% 内复现”不成立。HCA CP8 两个 launch 为 19.00/20.98 ms，当前精度不足以解释百分之一量级差异。

## 3. 已有 trace 给出了 teacher 分支的定量证据

本次直接解析 `agent_run/20260910T170209Z_p1b_mixed_mesh/profiles_{1_1,2_2}/` 下全部 32 份 JSON：每种 pair 的 fixed/mixed 各 8 个 rank。它们不是新增 profile，也不是所有 CP 的单算子 profile；1_1 的 G=8，2_2 的 G=4，均为 CSA-HCA 注意力栈。

统计只使用设备 `cat=kernel` 事件，避免把 CPU 算子归因时间和实际 kernel 再相加。

| trace 范围 | `_csa_compressed_lse_thd_kernel` 每 rank 次数 | 耗时范围 | 占该 rank kernel 时长和 |
|---|---:|---:|---:|
| 1_1 fixed/mixed，共 16 份 | 0 | 0 | 0 |
| 2_2 fixed/mixed，共 16 份 | 1 | 435.03–449.58 ms | 62.52%–64.77% |

例如 `profiles_2_2/mixed_rank0.json`：teacher-LSE 为 448.384 ms，全部 kernel 时长和 702.511 ms，比例 63.83%。已有 text 表给出约 447.773 ms，与 JSON 并非完全同值；正式重算统一以 JSON 为准，不混用两份采集数值。

这些比例是 kernel 时长占比，**不是已证实的完整 step 可消除比例**。多 stream 可能重叠，profile 也改变调度；435–450 ms 不能直接作为修复后的加速承诺。NCCL 时长在 CP2 的偶/奇 rank 间还存在明显差异，后续需区分通信计算与等待，不能把 NCCL 总时长当作有效网络搬运成本。

同版本源码形成的链条：

1. 本地 Megatron HEAD 与 ENV_IDENTITY 记录均为 `ff44fc2b6f246912d85a785e14b4dad6f205432b`。远端未提交改动及 profiler harness 仍需下轮保存，commit 一致不替代完整运行身份。
2. `bench_fixed_mixed_csa_hca.py:322` 配置 `dsa_indexer_loss_coeff=0.0`，下一行 `dsa_indexer_use_sparse_loss=False`；运行日志也声明辅助损失为 0。
3. `csa.py:3067` 的 `use_indexer_loss` 取决于训练状态及 `compressed_topk`，没有要求系数大于 0；`csa.py:3165` 调用 `FusedCSAIndexerSparseAttnFromTopkFunc`。
4. `csa_utils/fused_sparse_attention.py:2945` 定义 FromTopk 路径。其 forward 在 dense 分支执行 indexer score、`_compute_full_csa_teacher_lse`、attention score 和 KL 计算；只有后续 indexer backward 等部分由 `loss_coeff>0` 保护。
5. CP1 的 `FusedCSAIndexerSparseAttnFunc` 在同文件 `:2551` 已用 `loss_coeff<=0` 跳过 teacher/loss 分支。
6. 两条路径均调用 `_csa_fwd_flash_mla`；“CP>1 没有 fused attention 前向”被代码和现有 trace 否定。相同 attention 前向并不意味着训练 wrapper、teacher 或 backward 的成本相同。

由此可以把“λ=0 teacher 浪费”从历史疑点提升为**当前 CP2 trace 中有显著执行负担、且有同版本源码支持的首要消融对象**。CP4/8 的贡献尚未由对应 trace 定量验证；CP2 的全部退化也不能都归给这一条 kernel。

λ>0 时，dense teacher 可能是所选训练目标的必要计算。这时问题变成在保持目标与梯度下优化 teacher，不能复用 λ=0 的删除策略。

## 4. Mixed：有较弱基线下的收益，尚无最优基线下的净收益

原始日志明确声明：DSv4 attention layer、等长 contiguous THD、aux loss=0、无 DP 梯度同步。栈顺序是 **CSA-HCA 两个注意力模块**，不含 FFN、输出头、真实 optimizer 或 FSDP 完整训练。

按日志补回汇总遗漏的 G 后：

| CSA/HCA CP | G | fixed 的统一 CP | fixed ms | mixed ms | 对该 fixed 加速 |
|---|---:|---:|---:|---:|---:|
| 1/1 | 8 | 1 | 318.83 | 320.14 | 0.996× |
| 1/4 | 8 | 4 | 1594.47 | 327.21 | 4.873× |
| 1/8 | 8 | 8 | 2475.16 | 328.22 | 7.541× |
| 2/2 | 4 | 2 | 706.12 | 706.01 | 1.000× |
| 2/4 | 4 | 4 | 924.18 | 707.25 | 1.307× |
| 2/8 | 4 | 8 | 1359.03 | 709.48 | 1.916× |
| 4/8 | 2 | 8 | 785.57 | 570.79 | 1.376× |

同一行 fixed/mixed 的 G 相同，可以做该对照；跨不同 G 行不能直接按 ms 排统一前沿。fixed 默认 CP 为该 pair 请求 CP 的最大值，不是表内第一个 CP。

在 G=8 的可比集合中：

- 最快已测统一点为 CP1：318.83 ms、42.76 GiB。
- mixed 1/4：327.21 ms、44.68 GiB，比 CP1 慢 2.63%。
- mixed 1/8：328.22 ms、44.68 GiB，比 CP1 慢 2.95%，显存高 1.92 GiB。
- 因此目前没有选择 mixed 1/8 作为最佳性能配置的证据。1/4 和 1/8 也没有足够独立重复来判断小差异。
- mixed 1/1 比 fixed 1/1 慢 0.41%，约 0.91 GiB 峰值差；“mixed 全域不输”是取整后的过强陈述。同布局下的内存差还提示需要检查运行顺序和存活对象，不能把所有显存差直接归给转换。

### 为什么 117 ms 不是 cohort pack/switch 的证据

`328.22 − (191.44 + 19.99) = 116.79 ms` 的减法混合了不同负载：CSA@CP1 的 R1 全机 G=8，而 HCA@CP8 的 R1 全机 G=1；mixed 1/8 的两层都处理 G=8。仅用线性资源记账，HCA@CP8 的对应量级也应参考 `8×19.99=159.92 ms`，不是 19.99 ms。`191.44+159.92=351.36 ms` 只是缺乏 batching/融合建模的粗估，不是精确预测或下界。

更直接的是：`bench_fixed_mixed_csa_hca.py:91` 的 transition 在验证 rank token interval 后直接返回 hidden，backward 直接返回 grad_output。日志也标记 metadata-only。该特例没有 measured cohort pack/switch，更没有证据将 116.79 ms 全部归为此项。

如果需要分析现有 mixed 1/8，只能在相同 G/形状下采集两层与边界的独占范围，核对实际调用次数、payload 和输入布局。当前 archive 只有 1/1、2/2 trace，不能凭这些推断 1/8 的时间分解。

### 可靠性边界

P1b 本地 summary 不含每 launch 退出状态，1/1 还来自 salvaged 表，原日志含两段启动记录。本次保留其可解析性能指标用于筛选，但不把它称作 7 组完成了独立重复、正确性和稳定性验收。新增整理表显式标记退出记录缺失。

## 5. 对新增总结的必要更正

| 原说法 | 本次审计后的表述 |
|---|---|
| P1b 是 FSDP mixed CP 训练完成 | P1b 是 attention-only fixed/mixed 前反向筛选；FSDP 是另一批 GDN/GQA 运行 |
| mixed 1/8 已是选定最佳配置 | 它胜过统一 CP8，但仍慢于同 G 的统一 CP1；保留为候选 |
| 117 ms 为 cohort pack/switch 成本 | G 不匹配的残差，且 transition 为 metadata-only；归因不成立 |
| fixed(1_x) 是 CSA@CP1 串行造成慢 | 日志显示 fixed(1_4/1_8) 中 CSA 分别为 CP4/CP8，原解释读错布局 |
| fused 缺失已排除，所以不需要 profile | fused 前向已排除；现有 trace 反而证明 teacher-LSE 是主要归因对象 |
| CP 退化由通信机械成本/短序列 kernel 吞吐造成 | 仍待分解；teacher 路径差异已可直接观察，不能被这类泛化解释覆盖 |
| #5992/#7064 已解释历史 552→191 ms | 提交时间相关性是线索；缺少同环境配对消融，不能确定因果和贡献比例 |
| 回退代码重跑 CP8 应回归 552 ms | 552 ms 是旧 CP1 数据；若做历史对照，应按匹配环境检查 CP1，使用隔离 checkout，不改工作仓库 |
| 其余 HCA 在 1% 内复现 | HCA CP1 实际下降约 10.75%，CP2 也超过 1% |
| P2a 为 64K 或可作为 CSA/HCA capacity 决策 | 原始配置为 S=131072、14 层 GDN/GQA、G32/B1，不能跨模型套用 |
| CP4/8 容量 FEASIBLE | 最终波次均 0 个完成 step，正式状态应为 RUNTIME_FAIL、短程训练可行性未验证 |

历史提交回退不是下一轮的优先依赖。当前源码和 trace 已足以设计最小修复；只有需要归属历史性能变化时再做版本消融。

## 6. P2a：发现 CE 显存问题，但没有拿到训练前沿

来源：`agent_run/20260910T191248Z_p2_{u2,u4,u8}/case_summary.tsv`、各 `logs/u*.log` 和 `analysis/cp_scalability_20260910/uniform_frontier.csv`。

原始配置为 8×H800、14 层、hidden 4096、FFN 16384、`experimental_attention_variant=gdn`、`linear_attention_freq=4`、TP=PP=1、S=131072、B1/G32、Megatron FSDP。此批不能和历史 H100 的 52.104/49.810 s 拼成同环境曲线。

| case | 正式原始状态 | 完成 optimizer step | 可以下的结论 |
|---|---|---:|---|
| U2 | OOM | 0 | 当前配置在 CE 的 FP32 logits 转换处 CUDA OOM |
| U4 | RUNTIME_FAIL | 0 | 被信号中断，短程稳定性和训练峰值未验证 |
| U8 | RUNTIME_FAIL | 0 | 被信号中断，短程稳定性和训练峰值未验证 |

U2 的 37.03 GiB 分配与 `(131072/2)×151936×4 bytes` 一致。报错栈定位到 `cross_entropy.py:25` 的 `vocab_parallel_logits.float()`。当前 TP=1，即使函数名称叫 vocab-parallel，也没有实际把词表分散到多个 TP rank。

报错时 free=25.05 GiB，PyTorch reserved-but-unallocated=8.71 GiB；即使粗略把全部空闲缓存都归还，合计 33.76 GiB 仍小于 37.03 GiB。这支持当前 live set 下存在实质工作集问题，单纯调整 allocator 不能被当作充分解法。它不证明所有 CE 策略下 CP2 都不可行。

`agent_run/uniform_frontier.csv` 的 72.2/58.9 GiB 属手工记账的运行中 nvidia-smi 观测值，本地没有对应原始采样用于确认最终全 rank 峰值。未跑完 optimizer step 前，不能称为完整训练峰值或短程可行。

SIGKILL 与 host-OOM storm 的因果仍未证明；P2A_DECISION 已承认相关 PID 缺少内核 OOM 记录。下一次先隔离资源并保存 PID、时间戳及系统日志，不继续在相同故障条件下反复烧机时。

## 7. 下一轮执行顺序和门禁

### A. CSA 优先：只修 λ=0 的无效 teacher/loss 工作

第一步不新增 GPU 扫描：冻结与这批运行对应的源码、完整命令、未提交改动和 profiler harness；把 λ、sparse_loss、实际 forward/backward 类名、G/B/A、local/global max-seqlen 写入 manifest。本地 bench 比远端 profile 版本少采集接口，不能只给出一个 commit 就宣称可重放全部 trace。

修复候选：在 FromTopk 的内部，为 **λ=0** 增加与 CP1 一致的 teacher/loss 跳过路径，保留决定 attention 输出的 indexer、Top-k、压缩 KV 和全部必要 CP backward 通信。

不要简单把 `csa.py` 的 `use_indexer_loss` 改成 `loss_coeff>0`：这个分支还控制 deferred gradient/RS 所有权与 overlap，粗改 dispatch 可能改变 attention 的 backward。也不要把原先零梯度改成 None，避免 weight decay/optimizer 更新语义变化。

门禁：

1. 小尺寸覆盖 CP1/2/8、输出、Top-k、loss、输入梯度、参数梯度、同初态一次 optimizer update；补一个目标长序列 case。容差按 reference replay 冻结。
2. λ=0 下验证 teacher/score-only kernel 消失、必要 attention/indexer/collective 保留。
3. λ>0 的回归对照必须保持原训练目标与原行为，不用关闭辅助损失构造“训练加速”。
4. 若跳过后仍有明显退化，按 compute、teacher、indexer、padding/grid、通信 exposed wait 继续区分，停止先验归因。

预算：先 CP2 原版/修复各 1 次诊断 profile；正确性通过后，CP1/2/8 × 原版/修复 × 3 独立无 profiler launch，共 18 次短算子测量。先不补 CP4；若新曲线出现可竞争区间，再补对应点。已有 CP2 profile 已满足“先确认负担”的门槛，不需要重新扫旧曲线来重复发现问题。

输出：`csa_lambda0_ablation.csv`、原始 trace、正确性报告，分别报告实际延迟、峰值和 C×t。λ=0 的修复收益可作为 benchmark/实现修正，不能直接宣称真实 λ>0 训练也获得同样收益。

### B. 生产主线：先恢复 U8/U4，再验证一种 CE 内存方案

CSA 不阻塞这条工作线；相同 GPU 上分时运行。

1. 在稳定、独占的运行环境先重放 U8，至少完成 4 个 optimizer step，覆盖 optimizer 状态初始化；成功后运行 U4。若 U8 再因外部信号失败，暂停后续矩阵，保留系统证据定位故障。
2. 可行点各做至少 3 warmup + 5 measured 完整 step 的短筛选，得到速度和全 rank 峰值。CP8 与 CP4 谁更快由实测决定。
3. 对 U2 的 CE 捕获临界前 live set：BF16 logits、FP32 转换、CE saved tensors、输出头/FSDP buffers。先验证一个最小 chunked/fused CE 候选，精确处理 loss 分母、mask、反向保存与 logits 生命周期。
4. 若只分块 CE 仍保留完整 logits 的工作集不够，再考虑输出投影与 CE 联合分块；局部 vocabulary TP 是后续候选，会改变布局和通信，不能称为一个无成本 CE 开关。
5. 同一 CE 改进也给 U4/U8 使用；重新选择修复后的最佳统一配置。随后仅对 1–2 个竞争布局做 baseline/treatment × 3 独立 launch（6–12 次）。

短程通过前不报告容量 FEASIBLE。U2 被救活也不自动等于吞吐更高；容量收益和训练速度分别验收。

### C. 修复后重新判断 CSA/HCA 是否值得异构布局

当前不追加七组 mixed 矩阵。先修正 CSA teacher 路径与统计口径，再做以下门禁：

- 固定 W/S/G/训练目标，以 C×t 和必要通信估计候选。G=8 attention-only 比较与 B1/G32 完整训练分别成表；不能互相借用最佳时间。
- 首先保留同版本统一 CP1 作为当前 G=8 筛选基准；其余 CP 按修复后的曲线选择最有希望的统一点和至多一个 mixed 点。比较完整训练时，在 B1 下通过 A 而非偷偷增大 B 维持 G。
- 对目标 λ>0 先验证 teacher 的正确训练成本。如果主要瓶颈仍在 teacher，优先做必要 teacher 的分块/形状优化或同 hidden CP 的表示处理，而不是立刻恢复 hidden remesh。
- 候选要进入包含 FFN 的真实 decoder 周期，再到完整模型；所有共用算子修复都必须应用到统一基线。当前两层 attention 栈不能替代此门禁。
- 工程继续门槛沿用 0910 计划：正确性通过，独立重复稳定改善约 5%，或打开有价值的新容量点。5% 是投入阈值，不是统计显著性定义。若收益接近噪声，增加到 5 次重复后仍不稳定，则记录收益未证实并停止复杂 runtime 投入。

## 8. Roadmap 状态建议

| 条目 | 建议状态 |
|---|---|
| CSA/HCA 单算子扩展性 | 当前 λ=0 版本初测完成；修复后需更新 CP2/8 等关键点 |
| CSA 异常归因 | 已有 CP2 teacher-LSE 定量证据；因果消融待做 |
| mixed 战胜最佳统一布局 | 未证实 |
| 128K 统一训练前沿 | 未完成；U2 当前 CE OOM，U4/U8 运行中断 |
| TMF/cohort、headwise、planner/IR 扩展 | 延续 0910 暂停/条件项，当前数据不构成重启依据 |
| 下一可交付里程碑 | 一个语义等价的 CSA 修正结果，以及一个可重放的 128K FSDP 最佳已测统一基线 |

## 9. 本次产物与复核入口

- `analysis/csa_hca_audit_20260911/audit_0910.py`：仅使用标准库，读取原始 TSV/日志/JSON，输出独立审计结果，不覆盖原始数据。
- `analysis/csa_hca_audit_20260911/single_operator_audit.csv`：16 次 rc=0 的 R1 launch 重算为 8 格。
- `analysis/csa_hca_audit_20260911/mixed_log_audit.csv`：14 条 fixed/mixed 指标，补齐 G、local rows、真实布局及退出记录缺失标记。
- `analysis/csa_hca_audit_20260911/trace_kernel_audit.csv`：32 份 JSON 的逐 rank teacher-LSE/kernel/NCCL 时间统计。

原始资料还包括：总 roadmap 顶部 0910 更新、`ATTENTION_PARALLELISM_ANALYSIS_ARCHIVE_20260910.md`、`ATTENTION_TRAINING_EXPERIMENT_PLAN_20260910.md`、`NEXT_EXPERIMENT_EXECUTION_PLAN_20260910.md`、0903 CSA/HCA 执行计划、`analysis/cp_scalability_20260910/ENV_IDENTITY.md`、P1C/P2A 分析与各原始运行目录。本文没有新增外部文献主张。
