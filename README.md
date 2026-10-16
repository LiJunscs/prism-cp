# prism-cp

研究 GDN:GQA=3:1 模型的序列与层联合动态上下文并行，比较 Megatron 原生 N 的全局固定容量、逐 global batch 可变容量，以及二维联合方案。

- [新集群 Codex 接手入口：数据 → 模型 → 调度 → 执行](progress/CODEX_HANDOFF.md)
- [原始 CP 扩展性数据](bench_data/README.md)与[早期 GDN/CSA/HCA 对照](analysis/README.md)
- [当前事实与实测结果](progress/ARCHIVE-20261016.md)
- [下一步研究问题](progress/HETEROGENEOUS_CP_MASTER_ROADMAP_20260918.md)
- [论文讨论稿](progress/PAPER_ABSTRACT_AND_METHOD_DRAFT_20261016.md)
- [实现与执行合同](experiments/sequence_dynamic_cp/JOINT_CP.md)
- [H100 迁移和 Git 保留范围](progress/H100_MIGRATION.md)

GPU 实验按用户要求暂停。当前代码保留原生 N 和二维冻结计划的完整训练回放、独立求解、容量分析及必要证据；实际在线选择尚未实现。文档与命令示例不构成恢复实验的指令。
