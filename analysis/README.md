# 早期 GDN / CSA / HCA 的 CP 扩展性

保留两份原始小表：

- [gdn_csa_hca_scaling.csv](gdn_csa_hca_scaling.csv)：12 行，GDN chunkwise、CSA、HCA 各自 CP1/2/4/8 的时间与 allocated 显存；regime 为 `W8_B1_A1_D8divC_replica_concurrent`。
- [csa_hca(csa128)_scaling.csv](csa_hca(csa128)_scaling.csv)：8 行，CSA/HCA 的另一组 fused 实现测量；保留其原有独立口径，不与第一表合并拟合。

`n` 是 launch 数，`median_ms` 是 launch 中位时间，`group_gpu_ms=cp×median_ms`，`latency_speedup=T_CP1/T_CPp`，`scaling_efficiency=latency_speedup/cp`。例如第一表 GDN 从 CP1 的 85.346ms 降为 CP8 的 16.511ms；CSA 同一表的相应值是 552.48/479.855ms。它们说明不同算子的 CP 响应差异，不能用 CSA/HCA 代替当前 GQA 的响应。

两表曾由用户从备份仓库带入，未进入 Git；本次从 2026-10-14 读取原文件的会话工具输出恢复，分别 1278/924 字节，数值未重算。文件没有记录完整序列形状，按已有列解释，不补写未知条件。新集群不需要访问旧会话即可读取这些表。

当前 GDN:GQA=3:1 调度模型使用的 GQA/GDN 长度扫描、固定 batch 数据与拟合观测见 [bench_data](../bench_data/README.md)。
