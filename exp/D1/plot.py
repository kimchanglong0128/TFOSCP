# D1 figure: teacher trajectory — ArcFace(x0_hat(t)), face-box IoU, identity-token attention mass in the face box and entropy vs t; 999/800/399 marked.
import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
S = json.load(open('/workspace/exp/D1/summary.json')); A = S['per_t']; t = [a['t'] for a in A]
valid_attn = A[0]['mass64'] <= 1.0
fig, ax = plt.subplots(1, 3 if valid_attn else 2, figsize=(15 if valid_attn else 10, 4))
ax[0].plot(t, [a['arcface'] for a in A], 'o-', ms=3, label='ArcFace(x̂₀(t))'); ax[0].axhline(S['final_arcface'], color='k', ls='--', lw=0.8, label=f"final x₀ ({S['final_arcface']:.3f})"); ax[0].set_ylabel('ArcFace vs reference')
ax[1].plot(t, [a['iou'] for a in A], 'o-', ms=3, label='mean IoU'); ax[1].plot(t, [a['iou_gt05'] for a in A], 's-', ms=3, label='frac IoU>0.5'); ax[1].set_ylabel('face-box IoU vs final')
if valid_attn:
    ax[2].plot(t, [a['mass64'] for a in A], 'o-', ms=3, label='id-token mass in box (64×64 layers)'); ax[2].plot(t, [a['mass32'] for a in A], 's-', ms=3, label='(32×32 layers)'); ax[2].plot(t, [a['box_frac'] for a in A], 'k:', label='box area (chance)'); ax[2].set_ylabel('attention mass in face box')
for a_ in ax:
    for tt, lab in ((999, 'LCM/Hyper-LoRA 999'), (800, 'Hyper-UNet 800'), (399, 'DMD2 399')): a_.axvline(tt, color='r', ls=':', lw=0.8); a_.text(tt, a_.get_ylim()[1], lab, rotation=90, fontsize=6, va='top', ha='right')
    a_.invert_xaxis(); a_.set_xlabel('t (DDIM-50, SDXL teacher, CFG 5)'); a_.legend(fontsize=7)
plt.tight_layout(); plt.savefig('/workspace/outputs/D1_teacher_trajectory.png', dpi=120); print('saved')
