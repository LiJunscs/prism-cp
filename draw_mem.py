import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv("perf_data.csv")
target_phase = "fwd+bwd"

all_seqs = sorted(df["seq_length"].unique())
all_cp = sorted(df["cp_size"].unique())
x_pos = np.arange(len(all_cp))
cp_tick_labels = [str(c) for c in all_cp]

line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h']

fig, (ax_attn, ax_gdn) = plt.subplots(ncols=2, figsize=(16, 6), sharex=True)

# Left: Attention
for i, seq_len in enumerate(all_seqs):
    ls = line_styles[i % len(line_styles)]
    mk = markers[i % len(markers)]
    sub = df[(df["seq_length"] == seq_len) & (df["phase"] == target_phase) & (df["model"] == "attention")]
    sub = sub.set_index("cp_size").reindex(all_cp).reset_index()
    ax_attn.plot(x_pos, sub["peak_mem_gib"], linestyle=ls, marker=mk, label=f"seq={seq_len}")

ax_attn.set_title(f"Attention Peak Memory | {target_phase}")
ax_attn.set_ylabel("Peak Memory (GiB)")
ax_attn.grid(True, alpha=0.3)
ax_attn.legend(fontsize=8)
ax_attn.set_ylim(bottom=0)

# Right: Gated Delta Net
for i, seq_len in enumerate(all_seqs):
    ls = line_styles[i % len(line_styles)]
    mk = markers[i % len(markers)]
    sub = df[(df["seq_length"] == seq_len) & (df["phase"] == target_phase) & (df["model"] == "gated_delta_net")]
    sub = sub.set_index("cp_size").reindex(all_cp).reset_index()
    ax_gdn.plot(x_pos, sub["peak_mem_gib"], linestyle=ls, marker=mk, label=f"seq={seq_len}")

ax_gdn.set_title(f"Gated Delta Net Peak Memory | {target_phase}")
ax_gdn.grid(True, alpha=0.3)
ax_gdn.legend(fontsize=8)
ax_gdn.set_ylim(bottom=0)

for ax in [ax_attn, ax_gdn]:
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cp_tick_labels)
    ax.set_xlabel("CP Size (Context Parallelism)")

fig.suptitle(f"Peak Memory vs CP Size (All Sequence Lengths)", fontsize=14)
plt.tight_layout()
plt.savefig(f"mem_cp_{target_phase}_two_panel.png", dpi=300, bbox_inches="tight")
plt.show()
