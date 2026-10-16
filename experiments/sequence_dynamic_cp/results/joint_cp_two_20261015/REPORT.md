# 联合二维冻结计划的原始测量

本目录保存三组 16 GPU 完整训练测量。72 条结果继续用于当前容量消融；568/1136 条包含较大纯 CP1 短包，按对应显存开销解释。568 条低显存方案的实测在相邻 `joint_cp_memory_validation_20261016_r2/`，1136 条重打包只有 CPU 推导。

每组 25 个完整更新，前 10 步预热，第 11–25 步计量，取每步 16-rank 最大完整时间。包含 F/B、finalize、norm/clip、Adam 和参数同步；数值比较 skipped，不含离线搜索时间。

|输入|微批数|平均更新时间 s|计量步数|allocated 全卡峰值 GiB|
|---|---:|---:|---:|---:|
|72_samples_2p25Mi|10|14.708877194|15|44.187759|
|568_samples_16Mi|62|100.970852840|15|56.638170|
|1136_samples_32Mi|122|202.901375473|15|56.995080|

模型为 8 层 GGG*GGG*、H4096/FFN16384、BF16、TP=PP=1、无重计算。长批 GDN CP16、GQA CP8；MLP token 放置宽度为 16，前后向重分布计入时间。GDN 段含 MLP 保持 contiguous。

568 条经纯短批重打包后，allocated 为 44.591135GiB、完整更新时间为 101.306751 秒，约 12GiB 峰值减少保留了主要收益。本表的较大纯短包版本用于解释显存来源，不作为当前低显存方案。

`inputs/<case>.json` 保存输入，`plans/<case>.json` 保存计划与模型包络，executor 统一保存在 `../../configs/joint_executor.json`，`runs/<case>/records/steps_by_rank.json.gz` 无损保存原始逐 rank 更新记录；重复 jobs/replay 已删除，`summary.json` 保存计量时间。日志中的运行时路径可能不同于当前文件位置。

原生 N-K16 在相邻 `native_n_remeasure_20261016/`，只有 3 个计量步，双方跨启动窗口。72 条容量消融在 `native_capacity_ablation_20261016_r1/`；当前结果不证明全容量最优、在线计划收益或数值等价。全部测量已结束，不自动重跑。
