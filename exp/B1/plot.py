# B1 figure: per-identity paired deltas vs 1-step for the four sources + qualitative row (id00, seed 42).
import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
P = json.load(open('/workspace/exp/B1/results.json')); R = P['results']; ids = sorted(k for k in R if 'D_other' in R[k])
G = ['A_ref', 'B_meanface', 'C_silhouette', 'D_other']; L = ['A reference face', 'B mean face', 'C gray silhouette', 'D other identity']
fig = plt.figure(figsize=(16, 7)); gs = fig.add_gridspec(2, 6, height_ratios=[1.2, 1])
ax = fig.add_subplot(gs[0, :3]); x = np.arange(len(ids))
for i, g in enumerate(G):
    d = np.array([R[k][g]['arcface'] - R[k]['step1']['arcface'] for k in ids]); ax.bar(x + (i - 1.5) * 0.2, d, 0.2, label=f"{L[i]} (mean {d.mean():+.3f})")
ax.axhline(0, color='k', lw=0.8); ax.set_xticks(x); ax.set_xticklabels(ids, rotation=90, fontsize=7); ax.set_ylabel('ArcFace(source) − ArcFace(1-step), same 20 seeds'); ax.legend(fontsize=8); ax.set_title('B1 ablation, DMD2 + FaceID-PlusV2, θ=45°, ρ=0.15')
ax2 = fig.add_subplot(gs[0, 3:]); means = [np.mean([R[k]['step1']['arcface'] for k in ids])] + [np.mean([R[k][g]['arcface'] for k in ids]) for g in G]
se = [np.std([R[k]['step1']['arcface'] for k in ids], ddof=1) / np.sqrt(len(ids))] + [np.std([R[k][g]['arcface'] - R[k]['step1']['arcface'] for k in ids], ddof=1) / np.sqrt(len(ids)) for g in G]
ax2.bar(range(5), means, yerr=se, capsize=3, color=['gray', 'C0', 'C1', 'C2', 'C3']); ax2.set_xticks(range(5)); ax2.set_xticklabels(['1-step'] + L, rotation=20, fontsize=8); ax2.set_ylabel('mean ArcFace (30 ids)'); ax2.set_ylim(min(means) - 0.05, max(means) + 0.03)
for i, v in enumerate(means): ax2.text(i, v + 0.004, f"{v:.3f}", ha='center', fontsize=8)
k = 'id00'; ims = [f'/workspace/exp/AD9/ids/{k}.png'] + [f'/workspace/exp/B1/gen/{k}_src_{g}.png' for g in G] + [f'/workspace/exp/B1/gen/{k}_step1_s42.png']
tt = ['reference'] + [f'source {g[0]}' for g in G] + ['1-step s42']
for i, (p, t) in enumerate(zip(ims, tt)):
    a = fig.add_subplot(gs[1, i]); a.imshow(Image.open(p).resize((256, 256))); a.set_title(t, fontsize=9); a.axis('off')
plt.tight_layout(); plt.savefig('/workspace/outputs/B1_ablation.png', dpi=120); print('saved')
