# Plot identity similarity vs. number of inference steps.
# Left: ArcFace (woman only). Right: DINOv2 (woman + dog). Two prompt groups each.
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GROUPS = {
    "sunglasses": "/workspace/outputs/eval_ipa.csv",
    "neutral":    "/workspace/outputs/eval_ipa_neutral.csv",
}
STEPS = [8, 6, 4, 3, 2, 1]          # x-axis, descending: fewer steps to the right

fig, (ax_arc, ax_dino) = plt.subplots(1, 2, figsize=(11, 4))

for group, csv in GROUPS.items():
    df = pd.read_csv(csv).rename(columns={"arc_embed": "arcface"})  # old CSV used arc_embed
    ls = "-" if group == "neutral" else "--"

    # ArcFace: woman only
    w = df[df.subject == "woman"].set_index("step").loc[STEPS]
    ax_arc.plot(range(len(STEPS)), w["arcface"], ls, marker="o", label=f"woman / {group}")

    # DINOv2: both subjects
    for subject, color in [("woman", "C0"), ("dog", "C1")]:
        s = df[df.subject == subject].set_index("step").loc[STEPS]
        ax_dino.plot(range(len(STEPS)), s["dino"], ls, marker="o", color=color,
                     label=f"{subject} / {group}")

for ax, title in [(ax_arc, "ArcFace cosine (face identity)"),
                  (ax_dino, "DINOv2 CLS cosine (global appearance)")]:
    ax.set_xticks(range(len(STEPS)))
    ax.set_xticklabels(STEPS)
    ax.set_xlabel("inference steps")
    ax.set_ylabel("cosine similarity to reference")
    ax.set_title(title)
    ax.axhline(0, color="gray", lw=0.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

fig.suptitle("LCM-LoRA + IP-Adapter (sdxl), scale 0.6, guidance 1, seed 42", fontsize=10)
fig.tight_layout()
fig.savefig("/workspace/outputs/curve_ipa.png", dpi=150)
print("saved /workspace/outputs/curve_ipa.png")
