# T1 figure: 1-step identity (baseline and method) per generator ordered by t_start.
import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
S = json.load(open('/workspace/exp/T1/summary.json'))
order = [g for g in ['lcm_lora_999', 'tcd_lora_999', 'hyper_lora_999', 'dmd2_4s_999', 'dmd2_1s_999', 'dmd2_1s_800', 'hyper_unet_800', 'dmd2_1s_600', 'dmd2_4s_399', 'dmd2_1s_399'] if g in S]
x = np.arange(len(order)); fig, ax = plt.subplots(figsize=(13, 4.5))
ax.bar(x - 0.2, [S[g]['step1'] for g in order], 0.4, label='1-step baseline', color='gray'); ax.bar(x + 0.2, [S[g]['method'] for g in order], 0.4, label='+ structured init (θ=45°)', color='C0')
for i, g in enumerate(order): ax.text(i + 0.2, S[g]['method'] + 0.005, f"Δ{S[g]['delta']:+.3f}\np={S[g]['p_t']:.2g}", ha='center', fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(order, rotation=25, fontsize=8); ax.set_ylabel('ArcFace (30 ids × 20 seeds)'); ax.set_title('T1: t_start / capacity / distillation, FaceID-PlusV2, same prompt and seeds'); ax.legend()
plt.tight_layout(); plt.savefig('/workspace/outputs/T1_tstart.png', dpi=120); print('saved')
