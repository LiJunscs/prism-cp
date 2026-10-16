# Codex 接手入口

先读 [progress/CODEX_HANDOFF.md](progress/CODEX_HANDOFF.md)，再按任务读取其中的源码与证据。当前事实见 [progress/ARCHIVE-20261016.md](progress/ARCHIVE-20261016.md)，研究优先级见 [roadmap](progress/HETEROGENEOUS_CP_MASTER_ROADMAP_20260918.md)。

- 保留历史 CP 扩展性原始数据、二维时间/显存模型、独立调度器与完整训练执行链。`bench_data/`、`analysis/` 中列入交接文档的 CSV 即使没有运行时引用，也属于必要研究证据，不能按“未引用文件”删除。
- 后续模型固定 GDN:GQA=3:1。N 是 Megatron 原生 dynamic CP；连续 GDN 段及其中 MLP 保持 contiguous，只在 GQA/GDN 边界和模型 I/O 按需转换。此约束针对 N，二维保留自身执行自由度。
- GPU 实验仍处于用户要求的暂停状态；用户明确恢复后才运行。迁移、读取文档和 CPU 校验不授权 GPU 实验。使用已有环境，不安装或重建环境。
- 定向读取与验证；不做全工作区巡检、全文件 hash、源码快照、环境盘点、GPU 预检或重复 profiling。
- Git 候选范围见 `progress/PRISM_GIT_PATHS.txt` 和 `progress/MEGATRON_GIT_PATHS.txt`。不把工作区所有变更一起暂存或提交。修改 Megatron 前读取该仓库自己的 `AGENTS.md`。

用户后续指令优先于本文件。不要把冻结模型预测或旧 H800 时间当成新 H100 的实测结果。
