# Plot identity similarity vs. inference steps.
# Main lines: 3-seed mean ± std (sunglasses prompt). Dashed: single-seed neutral prompt for reference.
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEPS = [8, 6, 4, 3, 2, 1]
X = range(len(STEPS))
seeds   = pd.read_csv("/workspace/outputs/eval_ipa_seeds.csv")
neutral = pd.read_csv("/workspace/outputs/eval_ipa_neutral.csv")
agg = seeds.groupby(["subject", "step"])[["dino", "arcface"]].agg(["mean", "std"])

def series(df, subject, metric, stat=None):
    if stat is None:                      # single-seed frame
        return df[df.subject == subject].set_index("step").loc[STEPS][metric]
    return agg.loc[subject].loc[STEPS][(metric, stat)]

fig, (ax_arc, ax_dino) = plt.subplots(1, 2, figsize=(11, 4))

# ---- ArcFace (woman) ----
ax_arc.errorbar(X, series(None, "woman", "arcface", "mean"), yerr=series(None, "woman", "arcface", "std"),
                marker="o", capsize=3, label="woman / sunglasses (3 seeds, mean±std)")
ax_arc.plot(X, series(neutral, "woman", "arcface"), "--", marker="s", alpha=0.7,
            label="woman / neutral (1 seed)")

# ---- DINOv2 ----
for subject, color in [("woman", "C0"), ("dog", "C1")]:
    ax_dino.errorbar(X, series(None, subject, "dino", "mean"), yerr=series(None, subject, "dino", "std"),
                     marker="o", capsize=3, color=color, label=f"{subject} / sunglasses (3 seeds)")
    ax_dino.plot(X, series(neutral, subject, "dino"), "--", marker="s", color=color, alpha=0.7,
                 label=f"{subject} / neutral (1 seed)")

for ax, title in [(ax_arc, "ArcFace cosine (face identity)"),
                  (ax_dino, "DINOv2 CLS cosine (global appearance)")]:
    ax.set_xticks(list(X)); ax.set_xticklabels(STEPS)
    ax.set_xlabel("inference steps"); ax.set_ylabel("cosine similarity to reference")
    ax.set_title(title); ax.axhline(0, color="gray", lw=0.5); ax.grid(alpha=0.3); ax.legend(fontsize=8)

fig.suptitle("LCM-LoRA + IP-Adapter (sdxl), scale 0.6, guidance 1", fontsize=10)
fig.tight_layout()
fig.savefig("/workspace/outputs/curve_ipa.png", dpi=150)
print("saved /workspace/outputs/curve_ipa.png")
