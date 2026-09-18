# B2 figure: theta sweep on Hyper-SD (ArcFace delta, sunglasses probe, beach probe, DINO) + rho at theta=45.
import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
S = json.load(open('/workspace/exp/B2/summary.json')); G = S['groups']; th = [15, 25, 35, 45, 60, 75]
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
d = [G[f'th{t}_rho0.15']['delta'] for t in th]; se = [G[f'th{t}_rho0.15']['delta_se'] for t in th]
ax[0].errorbar(th, d, yerr=se, fmt='o-', capsize=3, label='ρ=0.15'); ax[0].axhline(0, color='k', lw=0.8)
for r in (0.10, 0.20): ax[0].errorbar([45], [G[f'th45_rho{r:.2f}']['delta']], yerr=[G[f'th45_rho{r:.2f}']['delta_se']], fmt='s', capsize=3, label=f'ρ={r:.2f}')
for t in th: ax[0].text(t, d[th.index(t)] + 0.004, f"p={G[f'th{t}_rho0.15']['p_t']:.2g}", fontsize=7, ha='center')
ax[0].set_xlabel('θ (deg)'); ax[0].set_ylabel('ΔArcFace vs 1-step (30 ids, ±SE)'); ax[0].set_title(f"Hyper-SD 1-step UNet (t=800) + FaceID; 1-step = {S['step1']:.3f}"); ax[0].legend(fontsize=8)
ax[1].plot(th, [G[f'th{t}_rho0.15']['sg']['delta'] for t in th], 'o-', label='P(sunglasses) Δ'); ax[1].plot(th, [G[f'th{t}_rho0.15']['beach']['delta'] for t in th], 's-', label='P(beach) Δ'); ax[1].axhline(0, color='k', lw=0.8); ax[1].axhline(-0.02, color='r', ls=':', lw=0.8, label='−0.02 limit'); ax[1].set_xlabel('θ (deg)'); ax[1].legend(fontsize=8); ax[1].set_title('attribute / background probes')
ax[2].plot(th, [G[f'th{t}_rho0.15']['dino']['mean_a'] for t in th], 'o-'); ax[2].axhline(0.6, color='r', ls=':', lw=0.8, label='copy threshold'); ax[2].set_xlabel('θ (deg)'); ax[2].set_ylabel('DINOv2 cos to reference'); ax[2].legend(fontsize=8); ax[2].set_title('copy guard')
plt.tight_layout(); plt.savefig('/workspace/outputs/B2_theta_sweep_hyper.png', dpi=120); print('saved')
