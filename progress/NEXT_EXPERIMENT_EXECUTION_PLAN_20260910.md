# 知识库主题：异构注意力训练下一轮实验执行计划

日期：2026-09-10。状态：计划已整理，新增 GPU 实验尚未执行。
项目根目录：E:/LLM论文/mine/heterogeneous_cp；本文相对路径均基于项目根目录。

## 1. 本轮决策与依据

**先找出 128K 完整训练的最佳可行统一配置及时间/显存瓶颈，再验证一个有因果依据的优化。** 本轮结束时，应明确相同 GPU、样本和训练目标下的净收益，以及下一阶段 Go/No-Go。

采用材料的优先级：

1. progress/ATTENTION_PARALLELISM_ANALYSIS_ARCHIVE_20260910.md，尤其“当前采用的结论与研究约束”，代表最新取舍。
2. progress/ATTENTION_TRAINING_EXPERIMENT_PLAN_20260910.md，提供 P0–P4 的依据；本文将其细化为执行顺序。
3. progress/FSDP_COHORT_CONCLUSIONS_20260910.md，用于还原已报告配置、速度与 OOM；其中继续探索 cohort/二维 head 的建议已被当天较晚计划替代。
4. 总 roadmap 和 1F1B roadmap 保留长期目标和历史机制证据，其立即推进 IR/TMF/streaming 的旧排序不作为本轮依赖。

本轮暂停 TMF/cohort 扩展、GDN headwise/二维 head 方案、通用 planner 和大规模扩展性扫描。其他局部异构布局由瓶颈触发，不预设 CP 必须是优化中心。

| 已有证据 | 本轮含义 |
|---|---|
| 64K/B1 GDN chunkwise CP1/2/4/8：85.346/54.120/29.046/16.511 ms | 单样本延迟改善，但每样本 GPU 时间从 85.346 增至 132.088 GPU-ms；复用曲线，只补归因 |
| 同口径 CSA：552.480/639.990/536.090/479.855 ms | CP2 退化值得诊断，不能先认定是通信或 teacher |
| HCA CP1/8：140.595/19.985 ms | 已有良好扩展证据，不优先重扫 |
| 32K、2CSA+2HCA：最佳固定 CP1 481.35 ms，CSA1/HCA2 482.24 ms | 已测混合方案未胜过最佳统一布局，暂停增加同类组合 |
| 用户报告 128K FSDP baseline 52.104 s，cohort CP8→2 49.810 s | 约 1.046×，单次且档案不全，只作机时参考 |
| GDN CP2/K4 的 1 GiB 申请失败 | 只能定位失败点，不能区分先前保存激活与当前 workspace |

单模块数值核对自 analysis/attention_evidence_20260910/mbs1_aggregates.csv。HCA CP8 第三次仅有日志指标而退出未确认，未纳入正式聚合。H800/H100、headwise/chunkwise、单模块/完整训练分别成表。上述 FSDP 速度与 OOM 未证明来自完全相同配置，不拼接成性能/容量曲线。

## 2. N0：冻结真实训练身份，不占性能测量预算

从远端实际成功命令和配置 dump 建立 baseline_manifest.json。本地已知目标：

- 8×H100 80GB，TP=PP=1，14 层 GDN/GQA，hidden=4096，FFN=16384，linear-attention-freq=4。
- chunkwise、BF16、Megatron FSDP optim_grads_params，S=131072，B=1，G=32。
- 曾通过 --untie-embeddings-and-output-weights 排除特定断言；是否保留以成功基线为准。

运行前补齐：真实逐层结构与 FFN 类型、Q/K/V 头数/维度、词表/输出头、loss/mask、decoder/head checkpoint、optimizer/学习率/weight decay、dropout/RNG、梯度累积与归约、FSDP wrap/prefetch/reshard 和实际分片组成员。历史单层 32 value heads 不能替代生产配置。

保存两仓库 commit/未提交 diff、容器/依赖、GPU/互联、完整命令、初始权重和全局样本清单。激活 DP 与模型状态分片组分别记账。现有 bench_attn_gdn.py 和 bench_fixed_mixed_csa_hca.py 均不能代替包含 FFN、输出头和 optimizer 的生产训练入口。

所有统一 CP case 关闭实验性 cohort/remesh，使用普通逐 microbatch 训练。每个 optimizer step 的独立样本集合相同；随 CP 改变样本分配及累积次数，保持 loss 分母和唯一有效 token 数一致，不能只保证相同 seed。

退出条件：配置可完整重放，W=DP×CP、G=DP×B×A 校验通过，checkpoint/loss 无未知值。缺少实际生产命令阻塞对应 launch，不阻塞独立算子诊断准备。

## 3. N1：统一布局容量与速度前沿

固定 N0 的模型和所有策略；A 为每个 DP 副本每个 optimizer step 的 microbatch 数。

| case/探测顺序 | CP | 激活 DP | B | A | 每 rank 每次前反向 token |
|---|---:|---:|---:|---:|---:|
| U8 | 8 | 1 | 1 | 32 | 16384 |
| U4 | 4 | 2 | 1 | 16 | 32768 |
| U2 | 2 | 4 | 1 | 8 | 65536 |
| U1，条件执行 | 1 | 8 | 1 | 4 | 131072 |

N1a：最多 4 次独立容量探测。每个 case 至少运行 3 个完整 optimizer step，覆盖状态初始化和后续迭代，只判“短程可行”。记录 OOM rank、step、层、前反向阶段、失败分配及 allocated/reserved/device-used。

CP8 失败先恢复真实基线，暂停矩阵。低 CP 失败后，只有 live-set 证据足以说明进一步降低 CP 无容量希望才停止下探；kernel/workspace 或分片策略可能非单调，不能只凭一次 OOM 推断。未跑点记 NOT_RUN。禁止静默改 B/G、截短序列或增加 checkpoint 救活某一点；新策略另建 treatment。

N1b：所有可行统一 CP 各做 1 次无 profiler 短稳态筛选。初始化/JIT 后至少 3 warmup + 5 measured 完整 step，漂移则延长。**最低可行 CP 不是未经测量的最快基线。** 一次 launch 仅负责筛选；竞争候选进入 N4 独立重复，接近的结果不强行排序。

输出 uniform_frontier.csv：可行性、完整 step 中位数、unique tokens/s、全 rank 最大峰值、FSDP gather/RS 次数和字节、checkpoint。等长无 mask 时每 step 为 32×131072=4194304 个唯一输入 token；另报实际有效 loss token，不重复计 CP 副本。CP/FSDP 组共同变化时称为完整配置比较，不全部归因 CP。

## 4. N2：先定位主因，再选机制

### N2a：1–2 次完整训练内存与关键路径诊断，最高优先级

先选 CP8 与最低可行 CP，相同则只测一个；若 N1b 的最快配置仍有未解释异常，再追加该点。

捕获 optimizer 初始化后、各层前后、GDN 核心、FFN、输出头/loss、重计算、反向末尾和 optimizer 前后的生命周期。保留所有 rank 的轻量日志，峰值 rank 保存详细 snapshot/trace；追踪须有界，避免追踪自身改变容量。

峰值时刻分解：

| 分类 | 必须回答 |
|---|---|
| 参数/optimizer/梯度状态 | 分片组是否改变，当前 gather/prefetch 多少层？ |
| 为 backward 保存的激活 | 哪些调用仍存活，checkpoint 保留哪些张量？ |
| 当前算子工作集 | GDN 中间量、FFN、输出头分别占多少？ |
| 通信与预取缓冲 | 峰值时哪些通信缓冲同时驻留？ |
| allocator 与外部分配 | reserved 未用、碎片、非 PyTorch 分配、未解释差额多少？ |

按底层 storage 去重，在同一时刻核算，不能相加各组件不同时刻的峰值。NCCL kernel 总时长不是可消除通信量；按依赖和 max-rank step 的 exposed wait 判断关键路径。带同步的诊断耗时不混入正式吞吐。

退出条件：给出有张量/事件依据的首要主因及未解释部分。如果尚不能区分“保存激活”与“当前工作集”，先补辨别测量，不提前选择 checkpoint/kernel 优化。

### N2b：GDN 和 CSA 各三个 profile，按空档分时进行

- GDN：64K/B1、历史明确形状，CP1/2/8 各 1 次。分离核心 fwd/bwd/recompute、状态通信、kernel 形状与工作集，解释每样本 GPU 时间增长。128K 生产形状只优先测 N1 关键 CP，不能线性外推历史头数。
- CSA：64K/B1，CP1/2/8 各 1 次。记录 projection、compression、indexer/top-k、teacher/score、core attention、collective 及 local/global max-seqlen。λ=0 无效 teacher 是历史源码线索，先确认运行版本行为及定量负担。
- GQA：仅当 N2 需要估算 GQA 优化份额时补当前版本 64K CP1/8，各 3 次；出现中间异常才补 CP2/4。HCA 与旧 GDN 全曲线不重扫。

脚本能力边界：run_gdn_chunkwise_mbs1_cp.sh 可复用形状/CP 配置；run_csa_hca_cp_profiles.sh 默认同时测 CSA/HCA，本轮应显式筛选 CSA 或使用底层入口。脚本的 profile 形状选项不等于已有 profiler 采集。CSA harness 目前写死 loss_coeff=0、use_sparse_loss=False，尚需暴露 loss/bypass/trace 接口；这些是待实现项，不提供虚构 CLI。

N2b 不阻塞 N1/N2a；同一 GPU 不并发正式 benchmark。

## 5. N3：只推进一个由瓶颈触发的机制

| N2 首要主因 | 最小对照 | 推进条件 |
|---|---|---|
| GDN 临时工作集 | 同 CP/输入，分块或减少物化 | 实际 allocated 峰值下降，递推/梯度不变 |
| 跨层保存激活 | 同布局，只改一个 checkpoint 区域/保存策略 | 打开更快配置或提高完整吞吐，重算计入 |
| FSDP 参数/缓冲 | 同 checkpoint，只改一种生命周期/prefetch/wrap 策略 | 峰值/等待改善，归约与更新正确 |
| FFN/输出头 | 同形状组件分块，必要时局部 TP | 计入布局/通信/backward 后有净收益 |
| CSA teacher 或形状膨胀 | λ=0 受限 bypass 或单个 shape 修正 | 输出/top-k/loss/梯度/参数更新等价 |
| 不同子图确有相反最优布局 | 先估固定资源收益上界，再测边界往返 | 上界覆盖通信、同步、复制及额外驻留后才写窄原型 |

CSA 只有证实 teacher 有实质负担才做 CP1/2/8 × 原路径/修复 × 3 launch=18 次正式消融，无负担则停止。真实 λ>0 保留原训练目标，不能删辅助 loss 获得“等价”加速。保留决定选点的 indexer/top-k，核对 zero grad 与 None 在 weight decay 下的区别。

正确性先行：小尺寸各目标 CP 检查输出、loss、input/parameter gradient 和同初态一次 optimizer update，再覆盖一个目标长序列形状。容差依当前 reference replay 与既有规范预先确定，不照搬旧 TMF 的 1% hard cap。随机算子明确 RNG 对齐方法，不能把同 seed 当作逐位等价。

随后在包含 FFN 的真实 GDN/GQA 或 CSA/HCA decoder 周期验证，再到 128K/14 层；周期顺序依真实配置，不凭简称推定。子图速度只是进入完整模型的门槛。

## 6. N4：完整训练成对验收

N1 的 1–2 个竞争布局 × baseline/treatment × 3 独立 launch，共 6–12 次。初始化/JIT 后至少 3 warmup + 5 measured step，交替执行顺序，正式计时不带 profiler。保持同初始模型/optimizer 状态和全局样本序列，记录逐 step loss、梯度范数、非有限值；跨 CP 的样本映射、累积和 loss 归约先做小尺寸等价检查。

报告每次 launch 中位数、跨 launch 范围、全 rank 峰值及相对同版本/同资源/同负载“最佳已测统一基线”的净收益。修复若也适用于统一配置，应让统一基线使用修复；后续异构方案与修复后的最佳统一配置比较。

工程 Go：正确性通过，step time 在独立重复中稳定改善约 5% 以上；或打开有价值的新容量点。5% 是投入门槛，不是统计显著性定义。胜负不一致或收益接近噪声时扩至 5 次，仍不稳则记收益未证实。简单小修复可以保留，但不足以支持复杂 runtime。

容量改善若仍慢于最佳可行基线，应分别报告容量与速度。所有 OOM/失败/未执行点保留。最后才允许各方案在 W/G/显存预算不变下分别调 B/A，满足 DBA=32，保留 B1 对照，并给予相同 checkpoint/batch 调优机会。

本轮短稳态验收不等于长期训练、收敛或 save/resume 已验证；机制进入长期主线后再安排长跑和恢复测试。

## 7. 预算、交付与立即执行顺序

| 阶段 | 首批新增预算 | 条件扩展 |
|---|---:|---|
| N0 身份冻结 | 无性能 launch | 远端配置补齐 |
| N1a 容量 | 最多 4 launch | CP1 依容量证据执行 |
| N1b 所有可行布局筛选 | 1–4 launch | 竞争点留给 N4 |
| N2a 完整训练诊断 | 1–2 profile launch | 排序异常点 |
| N2b GDN/CSA 归因 | 各 3 profile launch | GQA 6 次、CSA 消融 18 次 |
| N3 单机制门禁 | 按机制列最小 case | 通过才扩大 |
| N4 完整训练验收 | 6–12 launch | 边际结果增至 5 次重复 |

生产首轮 N1a+N1b+N2a 最多 10 次 launch，加入两条算子归因最多 16 次。不同 launch 长度不同，不能按数量直接估机时。N3/N4 等前级结论再启动。

旧 52 s/step 仅用于粗排：N4 的 6–12 次、每次 8 step，约 42–83 分钟有效迭代墙钟，约 5.5–11.1 GPU-hours（8 卡）。另加初始化/JIT/加载/诊断/失败与延长，不作为交付时限。

执行顺序：N0/N1 → N2a → 选择一个 N3 → N4；N2b 在独立资源或空档补充，CSA 不阻塞 GDN/GQA 生产问题。

每次保存 agent_run/<UTC>_<commit>_<experiment_id>/，包含 manifest、完整命令、环境、stdout/exit code、逐 rank/step 原始指标、样本身份及正确性结果；profile 与性能分目录。状态区分 PASS、CORRECTNESS_FAIL、OOM、RUNTIME_FAIL、NOT_RUN，不用零耗时填失败。

首轮交付 baseline_manifest.json、uniform_frontier.csv、peak_live_set.md、bottleneck_decision.md；N4 后交付 paired_training_results.csv 和 Go/No-Go。

**下一次获得 GPU 后：先冻结并重放真实 128K/14 层/CP8/B1/G32 统一训练基线，确认完整 optimizer step 与日志可复现；随后只改变统一 CP 和对应 A 做容量探测。**

本轮已阅读最新进展、总/阶段 roadmap、知识库归档、现有统计和启动脚本。未运行 GPU 实验，未实现 profiler/优化接口，未修改训练代码；计划沿用本地证据，未新增外部文献主张。
