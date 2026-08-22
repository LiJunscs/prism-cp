import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv("bench_merged_cp2.csv")
target_cp = df["cp_size"].unique()[0]  # 选择第一个cp_size值作为目标
df_cp = df[df["cp_size"] == target_cp].copy()

# 固定有序序列长度列表
seq_list = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]
# 等间隔位置下标：0,1,2...7
x_pos = np.arange(len(seq_list))

phases = ["fwd", "bwd", "fwd+bwd"]
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

model_colors = {
    "attention": "#1f77b4",
    "gated_delta_net": "#ff7f0e"
}
model_labels = {
    "attention": "Attention",
    "gated_delta_net": "Gated Delta Net"
}

for idx, phase in enumerate(phases):
    ax = axes[idx]
    df_p = df_cp[df_cp["phase"] == phase]

    for model in ["attention", "gated_delta_net"]:
        data = df_p[df_p["model"] == model].set_index("seq_length").loc[seq_list].reset_index()
        ax.plot(x_pos, data["mfu"], marker="o", label=model_labels[model], color=model_colors[model])

    # 核心设置：等间距坐标位置 + 原始数值标签
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(s) for s in seq_list], rotation=45, ha="right")

    ax.set_title(f"{phase}, CP={target_cp}")
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("MFU")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(bottom=0)

plt.suptitle(f"MFU vs Sequence Length (CP Size = {target_cp})", fontsize=15)
plt.tight_layout()
plt.savefig(f"mfu_seq_cp{target_cp}_equal_spacing.png", dpi=300, bbox_inches="tight")
plt.show()
