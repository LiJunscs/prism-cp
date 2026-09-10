# 知识库主题：FSDP 下 GDN/GQA cohort 的新增结论

归档日期：2026-09-10。证据来自本轮用户提供的训练结果、OOM 日志及本地实现/历史实验；尚未形成完整的 FSDP 独立运行档案。

## 已报告结果

- 负载：8×H100 80GB，TP=PP=1，入口 CP8、MBS1、GBS32、SEQ=131072；14 层 GDN/GQA，hidden=4096、FFN=16384、linear-attention-freq=4；chunkwise GDN、BF16、Megatron FSDP optim_grads_params。
- 使用 --untie-embeddings-and-output-weights 后，用户确认此前 FSDP 梯度桶 reset 断言消失；不能据此推断通用异步计数器存在错误。
- 单次端到端对照：baseline 52103.70 ms/迭代，CP8→4 为 51539.93 ms（1.011×），CP8→2 为 49810.48 ms（1.046×）。目前只有小幅方向性收益，尚无多次独立运行的稳定性证据。
- 明确的 OOM 对应 GDN CP2、K=4、128K、decoder/head 均不重计算：GDN 申请 BF16 [1,65536,64,128]（1 GiB）失败，进程占用约 78.39 GiB。用户另报告开启重计算仍 OOM，但缺少该次完整参数和栈，不能合并判断。

## 显存解释与边界

设入口 CP=C、目标 GDN CP=L、入口 MBS=B、cohort 大小 K、序列长 S：

B_GDN = K*B*L/C；转换前后每 rank 的 token 总数均为 K*B*S/C。

- 固定 cohort 内的 CP→DP 转换保持 token 数，不能再额外乘一次 CP 收缩比例；有效 DP 指激活样本布局，FSDP 模型状态通信域保持固定。
- 相对逐 microbatch 前反向的 baseline，cohort 同时推进 K 条样本。128K CP8→4/K2 时，GQA 为 2×16K，GDN 为 1×32K；baseline 每次为 1×16K。
- GQA 串行 forward 不等于激活释放：普通 autograd 保留多条图，逐层 checkpoint 仍保留各调用输入；当前 remesh 位于 checkpoint 外。GDN/FFN 驻留及工作张量也需计入，不能把额外显存全部归于 GQA。
- OOM 栈仅定位失败申请点，未区分历史激活驻留与当前算子工作集。现象说明当前 cohort 有显存代价，尚不足以否定 cohort 或异构 CP 原理。

## 方案取舍与下一步

- Boundary repeat：只在 GDN 边界复制、回到 GQA 前合并，数学上可保持单样本语义并减少 GQA 多样本驻留，但 GDN 本地 token 数不比最小 cohort 更少，且存在重复计算。复制反向求和、平均合并反向按副本数缩放，必须与 FSDP 梯度归约一致；不能统一额外缩放整网 loss。
- 本地 64K chunkwise 单模块三次运行中位数：CP8/4/2/1 分别为 16.51/29.05/54.12/85.35 ms。低 CP 处理不同样本可摊薄耗时；重复同一样本无法获得这份独立样本吞吐收益。该数据不是 128K FSDP 性能预测。
- 8 月 22 日 temporal repeat 与后续 spatial boundary repeat 是不同版本；8 月 30 日后者严格梯度门禁未通过，不能直接视为可用训练实现。
- 优先实验：① 对照 GDN_CP8/K1、GDN_CP8/K2、GDN_CP4/K2，定位聚合与转换的逐层显存成本；② 验证固定 CP8 输入输出、GDN 内部“序列 CP×head 并行=8”的二维布局，避免 cohort 与完整样本重复；③ boundary repeat 仅做通过梯度及 optimizer-update 检查后的受限对照。二维方案尚待实现与验证。
- Cohort 保留为基准，继续优化 checkpoint 区域、算子分块和激活生命周期。对比固定独立样本数，报告重计算策略、稳态迭代耗时和峰值显存；旧 headwise、单模块、PP 宏调度结果不得替代新 FSDP 证据。

## 证据入口

本地路径相对于项目根目录 E:/LLM论文/mine/heterogeneous_cp：

- agent_run/20260904T152108Z_manual_gdn_chunkwise_mbs1_cp/summary.tsv
- agent_run/20260822T042129Z_tmf_proto/summary.csv
- agent_run/20260830T081917Z_tmf_repeat_fixed/NEEDS_RESEARCHER.md
- progress/TMF_RELIABLE_CONCLUSIONS_AND_AGENT_RUN_EVIDENCE_20260901.md

设计参考：[PyTorch checkpoint](https://pytorch.org/blog/activation-checkpointing-techniques/)、[FLA GDN CP](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/cp/README.md)、[LoongTrain](https://arxiv.org/html/2406.18485v1)。参考机制不构成当前实现的收益证明。
