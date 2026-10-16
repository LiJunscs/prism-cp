# 保留的模块响应工具

`module_bench.py` 测量 GQA/GDN/MLP 模块，`fit_costs.py` 拟合成本，`cost_model.py` 读取拟合。Megatron 路径从 `MEGATRON_ROOT` 读取，默认使用 prism-cp 的兄弟目录。

`results/cp1_20261009_p1_v2/` 仅保留既有 H800 CP1 系数及其测量、输入和残差。这些仍被当前规划器测试使用；不是 H100 校准值。旧 profiling、sparse、P2P campaign 与重复日志已删除。

当前 GPU 任务暂停；保留工具不代表授权运行。
