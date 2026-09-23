# SFT 二维 Dynamic CP：最小实现与机会门禁

日期：2026-09-16。项目执行计划；早期调度原型在 heterogeneous_cp，后续训练实现位于 Megatron-LM。

## 最新进展：36K+28K 原生 Megatron 训练入口

按后续要求，已在 `E:/Project/Pythonproject/Megatron-LM` 接入固定父 CP4、
GDN 双 CP2 的完整原生训练路径。以下早期“尚未接入 Megatron / hooks 待实现”
描述仅记录当时状态；当前状态以本节及训练说明为准。

- `examples/gdn/train_packed_sequence_cp.py` 启动原生 `pretrain_gpt.py`，使用原生
  packing scheduler、梯度累积、distributed Adam、验证、checkpoint 与恢复。
- `MockVarlenDataset` 新增确定性 `packed_cp_case` 配方：每条数据仍是独立序列，
  交替 36864/28672，序列内移位标签、独立位置和 prompt loss mask。
- `megatron/core/packed_sequence_cp.py` 实现区间路由、子组 metadata 和可微 all-to-all；
  `TransformerBlock` 在 GQA/GDN 边界转换完整残差流，连续 GDN 层保留 CP2 布局。
- 参数梯度归约域不随激活布局改变；原生 `calculate_per_token_loss` 按全局监督 token 归一。
- 默认 GBS8 是 8 条独立序列、4 个 64K pack；每步输入 262144 token、监督 159744 token。
- 4 卡 case 使用 GQA CP4。GDN 两组每卡 18432/14336 token；最大本地 token
  比均匀 CP4 增加 12.5%，不是零代价降显存。重点候选是两节点各两卡，GDN CP2 留在节点内。
- 这一步是 layer-wise/sequence-wise 的固定父组 case；尚未接入任意父 CP 大小的 DCP 调度。

完整启动、同配置 uniform/split 对照和 save/resume 命令见
[原生训练说明](E:/Project/Pythonproject/Megatron-LM/docs/developer/packed_sequence_cp_training.md)。
旧 `benchmark_packed_sequence_cp.py` 仅作独立数值诊断，不作为完整训练入口或吞吐证据。

**验证边界：按用户要求，原生 launcher 与数据配方只做本机语法/静态检查，不在本机训练。**
此前 CPU/Gloo 的传输及参考 mixer 检查不等于真实 TE/FLA 训练验证；GPU 全流程、
distributed Adam 数值一致性、恢复和性能仍需服务器运行。首轮先小长度完整训练，
再 36K+28K 和两节点拓扑；拿到净收益后才扩展动态父组 planner。

## 决策

将近期验证方向转到 **同一个 packed microbatch 内，按序列组合及 layer group 选择 CP**。
冻结增加 TMF/cohort 宽度的优化；保留历史正确性、性能与负面结果。
先做单机、PP1、TP1、BF16、GDN×3/GQA，模型状态通信域固定。
不先造通用 planner，不同时研究 PP、TP remesh、MoE 或跨节点。

这不是“变长数据自动解决显存”。硬条件是：同一个调度 wave 的样本集合不变，
不借下一 microbatch 的样本、不复制序列、不把多轮计算图合在一起。
通过 THD packing 保持每 rank 的活跃 token 上界；真实 HBM 还取决于保存激活、
GDN 工作区、LM head、重计算和转换缓冲，必须实测。

## 当前证据与可复用代码

1. `progress/FSDP_COHORT_CONCLUSIONS_20260910.md`：128K、CP8→2、K4 的无重算
   GDN 路径 OOM；CP8→4/2 只有单次 1.011×/1.046×方向性结果。TMF 解决可整分性，
   不是降低长序列本地工作集的保证。
2. 9 月 11 日静态异构 roadmap 已要求“先有机会、再做 runtime”；沿用这一门禁。
3. 本地 Megatron 的 `DefaultDynamicCPScheduler` 已做 packing-aware 调度，
   成本近似 `sum(length²)/CP`，并非只按长度阈值选 CP。
4. 可复用 `PackedSeqParams.cp_group/local_cp_size`、预创建 dynamic CP groups、
   `resolve_cp_group()`、GDN packed/chunkwise 路径、现有 THD zigzag↔contiguous 转换。
5. `context_parallel_layout/` 是已有包；它处理 partition mode，不等于任意 CP degree
   与 sample ownership 转换。固定长度 `context_parallel_transition.py` 的旧 cohort 路径
   明确拒绝 packed/SFT，不能解除断言后直接复用。
6. `tests/unit_tests/transformer/test_thd_correctness.py::test_mixed_gdn_gqa_model_cp_correctness`
   是已有 8 卡 GDN/GQA + dynamic CP 语义测试入口；已有测试不证明本方案正确。

官方范围核对：[NVIDIA Dynamic CP 说明](https://developer.nvidia.com/blog/?p=111993)。
其按 packed microbatch 调整 CP；本文增加层组维度，仍需另行验证收益与新颖性。

## 最小策略：两条都测，优先手工 oracle

**A：保持现有 DCP packing，GDN 拆包。** 对一个 CP=C 的包，将完整独立序列分配到
两个 CP=C/2 的嵌套子组；保持序列内部顺序、padding、样本身份。只有两个子包非空、
每 rank padded tokens 均不超过预算且不超过父包上界时，才提出候选。
连续 GDN 层保留布局，下一 GQA 前恢复。单条长样本直接 fallback。

**B：联合 packing，GQA 合并、GDN 还原。** 从同一 wave 已有的相邻等宽子组出发，
GQA 使用合并后的父 CP 组，GDN 使用原子组。动机是 GQA 更接近平方长度成本，
GDN 更接近 token 线性成本；两种成本的平衡方案可能不同。
只允许预先存在的嵌套组。初版手工指定包，证明收益后再改调度策略。

B 的机制例子（不是实测加速）：4 卡同轮 4 条序列 `[32K,16K,8K,8K]`。

| 层组 | ranks 0–1 | ranks 2–3 | 每卡 padded tokens |
|---|---|---|---:|
| GQA | 全部 4 rank 用 CP4 处理同一个 64K pack | 同左 | 16K |
| GDN×3 | CP2 处理 32K | CP2 处理 16K+8K+8K | 16K |

整个 wave 只有原来的 64K token，无 K 倍扩容。GQA 若固定原来的两个 CP2 子组，
两组理想注意力计算量分别为 `32²/2=512` 和 `(16²+8²+8²)/2=192`；
合并 CP4 后每 rank 为 `352`，局部理想计算临界路径下降 31.25%。
这忽略通信、kernel 和其他 wave/rank，**不是吞吐预测**。
本方案还必须胜过“同 pack 的所有层均 CP4”，才能证明 GDN 降 CP 的价值。
若 uniform CP4 的 GDN 已更快，或最佳普通 DCP 能重新装包消除失衡，则这个例子没有净优势。

## 本轮已经实现与运行

- `tools/sft_2d_cp.py`：从本地 Megatron 源码加载两个纯调度函数，按 optimizer-step
  样本池重放；执行保守拆包筛选、相邻子组合并候选筛选；生成 sample/token 级区间路由。
- 支持 GQA zigzag 与 GDN flattened contiguous 布局，保留逐序列物理 padding。
  路由反向是精确逆置换，不引入 CP 平均。
- `tools/test_sft_2d_cp.py`：6 项 CPU 测试通过，覆盖不等长/padding、双向所有权、
  内积伴随、预算 fallback、单长序列、样本重复拒绝及联合 packing。
- `tools/sft_2d_cp_transport.py`：实际 PyTorch all-to-all/autograd 原型和 4 rank
  linear+masked loss+SGD smoke 入口已写好。**尚未执行分布式测试或接入 Megatron**。
  当前默认 Python 无 torch；本轮没有 CUDA/NCCL/TE/FLA 的训练结果。
- `analysis/sft_2d_cp/eligibility_strict.json` 和 `eligibility_growth10.json`：
  每组合成分布 256 样本、GBS64、8 rank、8K token/rank；对输入长度按 2×world 对齐。
  重放的是本地调度函数，不含真实 dataloader、运行时尾部 padding、SFT mask 或 kernel。

| 合成分布 | 原 DCP 包数 | A 可拆包 | A 可拆输入 token 比例 | B 候选相邻组对数 |
|---|---:|---:|---:|---:|
| 短序列占多数 | 101 | 0 | 0% | 7 |
| 中长度占多数 | 250 | 2 | 约 0.5% | 2 |
| 长序列占多数 | 256 | 0 | 0% | 0 |
| 等长对照 | 256 | 0 | 0% | 0 |

B 候选筛选使用局部 `sum(length²)/CP` 临界路径降低 ≥10% 的条件；尚未计入通信，
也未判断它是否是整个 wave 的关键路径。放宽 A token 增长至 10% 后，这组数据没有变化。
**当前结果不支持“给普通 DCP 加个 GDN CP 开关就有普遍收益”。**
真实长度分布与更适合联合装包的场景，才是下一步应优先验证的输入。

## 最少改动的 Megatron 接入合同

先做独立 macro oracle；通过机会门禁后才改生产文件。保留工作区已有未提交 TMF/repeat 修改。

1. **scheduler 外层 adapter**：沿用 `DefaultDynamicCPScheduler` 的 wave/sample IDs，
   附加 GQA/GDN 两个 pack layout。先 hard-code 上述 4 样本，再用离线表；不改 optimizer
   step 样本集合、不跨 step 搬运。B 需要把输入按 GQA 父组分发，不能只改 cp_size 字段。
2. **metadata factory**：为每个布局分别创建新 `PackedSeqParams`；完整重建
   `cu_seqlens_q/kv`、`*_padded`、`max_seqlen`、`seq_idx`（若 backend 需要）、
   `cp_group`、`local_cp_size`、`cp_partition_mode`、`cp_partition_route`。
   不就地覆盖 autograd/recompute 仍引用的原 metadata。prototype 暂不支持 dummy tail。
3. **边界 adapter**：在 `TransformerBlock` 的连续层组边界转换完整 residual stream，
   不仅是 attention normalized input；下一 GQA/最终 LN/head 前恢复 canonical layout。
   复用同一层对象/参数，按调用传显式 group collection，不能永久改写 layer.cp_group。
4. **transport**：在父组做 interval pack → all_to_all_single → unpack；反向执行逆路由。
   显式传入 parent group，本地 rank 与全局 rank 分开。默认同步执行先证明 collective 次序，
   不在 forward 创建 groups，不提前做通信 overlap 或 CUDA Graph。
5. **固定 state mesh**：PP1、TP1，参数梯度最终在固定 DP×CP 域归约。
   loss 以全局有效监督 token 总数归一；prompt/padding mask=0，输入 token 吞吐和监督 token
   吞吐分开。不得以 child CP size 做额外 loss 缩放。Megatron 的既有归约倍率需逐项审计。
6. **recompute**：初版关 dropout/FP8，先小模型 no-recompute 对齐；然后固定同一 checkpoint
   策略。明确边界 tensor 是否保存，不能凭 `local_tokens` 判断 full-model HBM。

最小新增内容是 adapter、metadata factory、ragged transport，以及约 2 个 layer-group hook；
不是重写 attention/FLA/FSDP。当前提交完成了 policy/route/transport 的独立原型，
adapter、metadata factory 和训练 hooks 仍是后续工作，不能宣称已有二维 SFT 训练器。

## 执行顺序和停止条件

**Gate 0，先用一个短实验时段完成。**
导出实际 SFT 在模板化/tokenize/截断之后的长度及 loss-mask 统计；扫描 A/B 覆盖率。
同时跑已有 DCP 混合模型正确性测试。不能用只含随机 token 的 mock 实验代替真实 SFT 语义。
若候选只占很小的训练耗时，即使局部优化也无法带来显著整步收益，停止扩建 runtime。

**Gate 1，手工 4/8 卡 macro。**
用同一 checkpoint、同一样本集验证 `GQA→GDN×3→GQA`，测试两个 4-rank 岛可扩到 8 卡。
先 4K/2K/1K/1K 做数值检查，再 32K/16K/8K/8K 与真实样本分布；不要首轮就冲 128K。
必须比：①相同 pack 全层高 CP；②原始 DCP；③二维 oracle。
测 forward+backward、转换耗时、all-rank max allocated/reserved、局部 token 最大值。
只测 GDN kernel 不作 Go 判据。

**Gate 2，完整 decoder。**
Gate 1 相对最佳可行对照净吞吐 ≥5% 且显存有余量，再接 layer hooks。
扩到已有 14 层模型；必要时使用两侧完全相同的已验证 FSDP 路径。
检查 output、input/parameter grads、至少 5 次 optimizer update、重计算与 save/resume。
BF16 阈值用同 kernel 的 uniform replay 误差校准，不能沿用 FP64 transport 阈值。

**Gate 3，正式收益。**
同模型/样本集合/监督 token/全局 batch/精度/重算/显存预算：
U=最佳 fixed CP + packing；D=原生 sequence-wise DCP；L=固定 layer-wise policy；
LD=联合二维 CP。另做“只改 packing、不做 layer remesh”消融，分清收益来源。
固定配置和等预算分别调优的结果分表。各 3 次独立 launch，warmup 后测完整 optimizer step。
主指标 `sum(unique nonpadding input tokens)/sum(step time)`，另列监督 tokens/s。
收益若小于噪声/转换成本，保留 DCP，结束二维 runtime 投入。

最有希望：多个独立序列同轮可装入、线性 token 成本与平方 attention 成本出现失衡、
连续 GDN 层足够长、NVLink 内转换便宜。弱场景：单条极长样本、原生 DCP 已充分均衡、
等长大样本、GQA/FFN/head 主导显存且 GDN 降 CP 无净收益。

## 可执行入口

在本项目根目录：

```powershell
python tools/test_sft_2d_cp.py -v
python tools/sft_2d_cp.py --megatron-root E:/Project/Pythonproject/Megatron-LM --output analysis/sft_2d_cp/eligibility_strict.json
# lengths.json 是实际每条样本长度的整数数组；用于机会扫描，不上传数据。
python tools/sft_2d_cp.py --megatron-root E:/Project/Pythonproject/Megatron-LM --lengths-json lengths.json --cap 16384 --step-samples 64 --output analysis/sft_2d_cp/real_lengths.json
```

在配置好 torch 的 4 卡服务器，本项目根目录（仅 transport 语义）：

```bash
torchrun --standalone --nproc_per_node=4 tools/sft_2d_cp_transport.py --backend nccl
```

在配置好 Megatron/TE≥2.9/FLA 的 8 卡服务器，Megatron 根目录（既有 DCP 基线语义）：

```bash
python -m torch.distributed.run --nproc_per_node=8 -m pytest -q tests/unit_tests/transformer/test_thd_correctness.py -k mixed_gdn_gqa_model_cp_correctness
```

以上两条服务器命令均未在本机执行。通过 transport smoke 也不表示真实 GDN/TE/FSDP 已通过。
