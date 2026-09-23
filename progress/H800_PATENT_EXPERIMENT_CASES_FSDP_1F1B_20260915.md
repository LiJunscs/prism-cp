# H800 专利实验方案：异构 Attention Mesh、FSDP 与 1F1B

日期：2026-09-15  
用途：为“面向大语言模型异构注意力变体的静态异构网格训练方法及系统”形成可复核的性能、容量与机制证据。

## 1. 要验证的核心命题

同一 LLM 内，GQA 与 GDN 等 Attention 变体的最优并行布局不同。保持模型、数据、有效全局批量和数值精度不变时：

- GQA 使用较大的序列 CP，以满足长上下文显存约束；
- 连续 GDN 层组使用较小 CP、较大有效 DP，以减少 GDN 的 CP 通信和小粒度执行损失；
- 在两者边界通过 TMF 对多个独立 logical microbatch 进行时空重排；
- 在 1F1B 中，以 cohort 为调度单位，并用足够大的 logical microbatch 数 `M` 摊薄宏 cohort 引起的 pipeline bubble；
- 两机时把 CP 子网格限制在单机 NVLink 域内，避免把高频 Attention 通信跨到 IB；跨机通信仅保留在固定的 FSDP/HSDP 或 PP 边界上。

专利主结果应同时满足：相对“最佳可运行的均匀 mesh”更快，或在该均匀 mesh OOM 时提供更高的可训练序列长度。不能只与 CP8 一个弱基线比较。

## 2. 固定实验口径

### 2.1 通用配置

- GPU：H800 80GB；单机 8 卡、两机 16 卡；记录 NVLink/NVSwitch 与 IB 型号、速率及拓扑。
- 精度：BF16；TF32、Flash Attention、GDN backend、随机种子固定。
- 模型：优先复用现有 14 层、hidden 4096、FFN 16384、32 heads、8 query groups、`linear-attention-freq=4` 的 GDN/GQA 模型。
- 数据：相同 checkpoint、相同 token 顺序、相同 loss mask；对照间有效 GBS 和 token 数完全一致。
- 训练：FSDP 使用已成功路径 `Megatron FSDP + optim_grads_params + distributed optimizer + untied embedding/output`；所有 case 保持相同重计算和 offload 策略。
- 统计：每个正式点 3 次独立 launch，每次 5 个 warmup step、至少 10 个 measured step；报告 launch median、范围和所有 rank 的最大显存。若优势小于 8%，扩大到 5 次 launch。
- 计时：用所有 rank 的最大 step time；吞吐为 `有效 GBS × sequence length / step time`。同时保存 transition、Attention、GDN、FSDP collective、PP P2P 和 bubble 的分项时间。

### 2.2 正确性门禁

在性能测试前，所有 heterogeneous case 必须通过：

1. 固定 seed 的 forward loss 与均匀 mesh reference 对齐；
2. BF16 input/parameter gradient 误差不超过同一均匀 mesh replay 的数值本底，并保留 FP32 shadow 检查；
3. 至少连续 5 个真实 optimizer update，无 NaN/Inf；
4. 参数、梯度、optimizer state 的 replica/shard 一致性检查；
5. FSDP case 完成一次 save/resume 后 loss 轨迹连续。

当前 `1f1b/bench_tmf_1f1b.py` 不执行真实 optimizer step，因此其结果只能证明调度、前反向、梯度同步与性能机制，不能替代上述完整训练门禁。

## 3. 第一优先级：单机 FSDP 完整训练

### 3.1 主性能 case：64K

固定：8×H800，PP1、TP1，sequence 65,536，MBS1，GBS32，14 层混合 GDN/GQA，BF16，全层重计算策略一致。

| Case | GQA mesh | GDN mesh | cohort K | 有效布局 | 目的 |
| --- | --- | --- | ---: | --- | --- |
| F1-U8 | CP8 | CP8 | 1 | DP1×CP8 | 长序列安全基线 |
| F1-U4 | CP4 | CP4 | 1 | DP2×CP4 | 必测的强均匀基线 |
| F1-U2 | CP2 | CP2 | 1 | DP4×CP2 | 显存允许时必测，防止挑弱基线 |
| F1-H4 | CP8 | CP4 | 2 | GQA: DP1×CP8；GDN: DP2×CP4 | 低风险异构点 |
| F1-H2 | CP8 | CP2 | 4 | GQA: DP1×CP8；GDN: DP4×CP2 | 预期收益最大的主张点 |

推荐执行顺序：`U8 → H4 → U4 → H2 → U2 → U8复测`。交错顺序可降低温度、缓存和机器状态漂移造成的偏差。

预期最可能获胜的是 F1-H2，原因是 GDN 连续层组在 CP2 上处理四个独立样本，GQA 和 LM head 仍保持 CP8；已有 H100 单次数据的方向性结果为 CP8→2 相对 CP8 约 1.046×，但 H800 正式结果必须重新做多 launch，并且与 U4/U2 比较。

**Go 条件**：H2 相对最佳可运行 U8/U4/U2 的吞吐提升 ≥5%，且峰值显存不高于 76GB；或吞吐持平但峰值显存下降 ≥10%。  
**降级**：H2 OOM 时先保留相同模型与序列，将目标降为 H4；不要同时改变层数、重计算和 GBS。

### 3.2 容量甜点 case：96K

固定：sequence 98,304，其余同 64K。只先跑 `U8、U4、H4、H2`，U2 作为容量对照，即使 OOM 也要保存完整日志。

该点的目标不是保证所有 case 可运行，而是寻找以下专利价值更高的结果：

- U2 因 GQA/LM head 或激活峰值 OOM；
- H2 因仅在 GDN 层组使用 CP2，而在 GQA/head 保持 CP8，能够完成训练；
- H2 同时快于可运行的 U8/U4。

历史 128K 的 H2 曾在 GDN 工作张量处接近 80GB 并 OOM，因此 96K 是更合理的容量边界搜索点。若 H2 仍 OOM，不应直接降 sequence；先记录峰值位置，再测试 H4。96K 全部稳定后才把 128K 作为压力点，不把 128K 设为第一轮主实验。

## 4. 第一优先级：两机 FSDP/HSDP 完整训练

固定：2×8 H800，PP1、TP1，sequence 65,536，MBS1，GBS64。推荐每台机器形成一个 CP8 域，使高频 Attention CP 通信只走机内 NVLink；两机之间采用相同的 FSDP16，或在实现已验证时采用“机内 shard8、机间 replica2”的 HSDP。两种方式不可混在同一对照组。

| Case | GQA mesh | GDN mesh | 全局有效布局 | 拓扑约束 |
| --- | --- | --- | --- | --- |
| F2-U8 | 每节点 CP8 | 每节点 CP8 | DP2×CP8 | 两个 CP8 均为机内连续 rank |
| F2-U4 | 每节点 2×CP4 | 每节点 2×CP4 | DP4×CP4 | CP4 不跨节点 |
| F2-U2 | 每节点 4×CP2 | 每节点 4×CP2 | DP8×CP2 | CP2 不跨节点 |
| F2-H4 | GQA: 每节点 CP8 | GDN: 每节点 2×CP4 | GQA DP2×CP8；GDN DP4×CP4 | transition 只在节点内 |
| F2-H2 | GQA: 每节点 CP8 | GDN: 每节点 4×CP2 | GQA DP2×CP8；GDN DP8×CP2 | transition 只在节点内 |

主张点为 F2-H2。它同时体现“算子异构”和“拓扑感知静态异构 mesh”：GQA 保持机内 CP8，GDN 切换为多个机内 CP2，避免 Attention collective 跨 IB。

额外消融可加入 `U16 = CP16×DP1`。它不是主基线，而是证明拓扑约束价值的负对照：若 U16 因跨机 CP 通信显著变慢，需用 NCCL trace 证明差异来自跨节点 collective，而非不同 GBS。

两机 case 的强证据标准：H2 相对最佳机内均匀 U8/U4/U2 提升 ≥5%，并且相对单机保持合理的 weak-scaling 效率；或 H2 是 96K/128K 下唯一能完成 optimizer step 的布局。

## 5. 第一优先级：单机 1F1B 大 M 机制实验

当前实现要求 `world_size = PP × stage_mesh`，且每个 stage 首尾必须为 full-mesh CP。单机 8 卡采用 `PP2 × stage_mesh4`。连续放置 3 个 GDN 可把两次 transition 的成本摊到一个层组上，避免逐层来回切换。

### 5.1 正式比较

固定：PP2，stage mesh4，MBS1，M=32 和 64，sequence 16,384；稳定后增加 32,768。每个 stage 五层，模式相同。

| Case | 每 stage pattern | K | pipeline cohorts (`M/K`) | 目的 |
| --- | --- | ---: | ---: | --- |
| P1-U4 | `A4,G4,G4,G4,A4` | 1 | 32/64 | 标准 1F1B 均匀基线 |
| P1-H2 | `A4,G2,G2,G2,A4` | 2 | 16/32 | 低风险异构点 |
| P1-H1 | `A4,G1,G1,G1,A4` | 4 | 8/16 | 预期最快主张点 |

对于 PP2、宏 cohort K4，忽略 transition 后的 bubble 修正因子为 `(M+1)/(M+4)`：M32 为 0.917，M64 为 0.956。也就是说，大 M 已将 cohort 对流水线利用率的不利影响压到约 8.3% 和 4.4%；只要 GDN 低 CP 的主体收益超过该损失与 transition 开销，H1 即可获胜。

命令模板（单机）：

```bash
torchrun --standalone --nproc_per_node=8 1f1b/bench_tmf_1f1b.py \
  --pipeline-parallel-size 2 --context-parallel-size 4 \
  --stage-layer-patterns \
  "attention:4,gdn:1,gdn:1,gdn:1,attention:4;attention:4,gdn:1,gdn:1,gdn:1,attention:4" \
  --micro-batches 64 --micro-batch-size 1 --seq-length 16384 \
  --gdn-linear-cp-mode chunkwise --attention-backend flash \
  --warmup 5 --iters 10 --profile-stages \
  --trace-dir agent_run/patent_p1_h1_m64_s16k
```

U4/H2 只替换 3 个 GDN 的 CP 值与输出目录。必须保留默认 Triton transition 和 overlap；之后对最优 heterogeneous 点补 `--no-tmf-transition-overlap` 与 `--tmf-transition-kernel torch` 两个消融。

### 5.2 预期结果与停止条件

- 优先看 M64；它最能体现用户提出的“大 microbatch 下 TMF 仍可加速”。
- 若 H1 在 16K 已提升 ≥8%，再扩到 32K；若 32K OOM，保留 16K 主结果并把 H2 作为 32K 容量点。
- 若 H1 不赢而 H2 赢，专利表述应收窄为“根据算子与流水线气泡联合选择 mesh”，不能声称 CP 越小越好。
- 若 H1/H2 都不赢，先读 profile 中 `cohort_backward`、transition 和 stage tail wait；只有 transition/bubble 占主导时才进入 streaming adapter，不应通过删减均匀基线制造优势。

## 6. 第一优先级：两机 1F1B 大 M 与拓扑实验

两机 16 卡采用 `PP2 × stage_mesh8`。默认 torchrun rank 为 node-major 时，stage 0 为 rank 0–7（节点0），stage 1 为 rank 8–15（节点1）：stage 内 CP/transition 全在 NVLink 域，只有 stage 0→1 的 PP P2P 跨 IB。这是当前实现最干净的两机拓扑 case。

| Case | 每 stage pattern | K | M | 说明 |
| --- | --- | ---: | ---: | --- |
| P2-U8 | `A8,G8,G8,G8,A8` | 1 | 64 | 均匀 1F1B 基线 |
| P2-H4 | `A8,G4,G4,G4,A8` | 2 | 64 | 低风险异构 |
| P2-H2 | `A8,G2,G2,G2,A8` | 4 | 64 | 主张点 |

先用 sequence 16,384，之后增加到 32,768。暂不把 CP1/K8 设为主 case：K8 会同时增加 activation graph 驻留与宏 cohort bubble，且历史长序列已有显存风险。

两机启动模板，两台机器使用完全相同的 `MASTER_ADDR/PORT`，仅修改 `node_rank`：

```bash
# node 0
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
  --master_addr=<NODE0_IP> --master_port=<PORT> \
  1f1b/bench_tmf_1f1b.py \
  --pipeline-parallel-size 2 --context-parallel-size 8 \
  --stage-layer-patterns \
  "attention:8,gdn:2,gdn:2,gdn:2,attention:8;attention:8,gdn:2,gdn:2,gdn:2,attention:8" \
  --micro-batches 64 --micro-batch-size 1 --seq-length 16384 \
  --gdn-linear-cp-mode chunkwise --attention-backend flash \
  --warmup 5 --iters 10 --profile-stages \
  --trace-dir agent_run/patent_p2_h2_m64_s16k

# node 1: 同上，仅改为 --node_rank=1；输出目录必须是两节点可见的共享路径。
```

运行前必须核对 rank 0–7/8–15 的物理节点归属。若 launcher 的 rank 排列不同，需要显式调整 rank 或 group mapping；否则 stage 内 CP 可能跨 IB，结果不能用于上述拓扑主张。

不建议把 `PP4 × stage_mesh4` 作为首轮主实验：相同 M/K 下 PP4 的 bubble 大于 PP2。它可作为后续消融，证明 planner 会联合考虑 mesh 与 pipeline 深度。

## 7. 最小执行集与资源优先级

如果机器时间有限，按以下顺序运行：

1. 单机 FSDP 64K：U8、U4、H4、H2、U2；这是完整训练主证据。
2. 单机 1F1B 16K/M64：U4、H2、H1；这是大 M 流水线主证据。
3. 两机 FSDP 64K：U8、U4、H4、H2、U2；这是拓扑与扩展主证据。
4. 两机 1F1B 16K/M64：U8、H4、H2；这是 stage-local heterogeneous mesh + 跨机 PP 的机制证据。
5. 单机 FSDP 96K：U8、U4、H4、H2、U2/OOM；这是容量边界证据。
6. 仅对已获胜的点补 32K 1F1B、transition overlap/kernel 消融和 nsys trace。

## 8. 专利图表建议

最终至少形成四张图/表：

1. 单机 FSDP：U8/U4/U2/H4/H2 的 tokens/s 与 peak memory 双轴图；
2. 两机 FSDP：同组布局的吞吐、扩展效率和跨节点通信占比；
3. 1F1B：M=8/16/32/64 的 speedup 曲线，展示随着 M 增大，TMF bubble 被摊薄；
4. 分项时间瀑布图：GQA、GDN、TMF transition、PP P2P、FSDP collective、pipeline idle。

表格必须同时列出 `PP/TP/CP/DP/K/M/MBS/GBS/S`，以及“CP group 是否跨节点”。OOM 也是有效的容量证据，但必须记录失败算子、申请大小、allocated/reserved 和同配置的重计算策略。

## 9. 可支持与不可支持的专利表述

完成上述第一优先级实验且 heterogeneous 点胜过最佳均匀点后，可支持：

> 针对同一大语言模型内通信与状态结构不同的 Attention 变体，预先生成逐层或逐连续层组的静态异构并行网格，并通过 temporal microbatch folding 在网格边界保持独立样本语义；在 FSDP 和 1F1B 训练中，该方法能够在长上下文显存约束、算子效率与流水线气泡之间选择更优的联合布局。

在 optimizer 多步、save/resume 尚未通过前，1F1B 数据不能写成“已经验证完整训练收敛”；在只胜过 U8、未胜过 U4/U2 时，不能写成“优于最佳均匀并行策略”。
