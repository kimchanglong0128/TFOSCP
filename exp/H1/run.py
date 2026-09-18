# Card H1 (post-processing only): where does the identity branch write? From the H1 cache (SDXL + LCM-LoRA + IP-Adapter base, steps 1..8, seeds 42-44):
# per IP cross-attn layer and step, the per-pixel norm of the IP-branch output. Metric: fraction of IP-branch norm mass inside the FINAL image's face box
# (concentration), compared with the box area fraction (chance level), as a function of the number of steps. Heatmaps for 1 step vs 8 steps.
import sys, os, json, math; sys.path.insert(0, '/workspace/pilot')
import torch, numpy as np, cv2
from PIL import Image
import idmetrics as m
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
H = '/workspace/outputs/h1'; OUT = '/workspace/exp/H1'; STEPS = [1, 2, 3, 4, 6, 8]; SEEDS = [42, 43, 44]
def box(step, seed):
    b = m.face_bbox(cv2.imread(f'{H}/woman_step{step}_seed{seed}.png')); return None if b is None else b.astype(float)
def conc(mp, b):
    n = int(round(math.sqrt(mp.numel()))); M = mp.view(n, n); x1, y1, x2, y2 = [v / 1024 * n for v in b]
    inside = float(M[int(y1):int(math.ceil(y2)), int(x1):int(math.ceil(x2))].sum() / M.sum()); area = (x2-x1)*(y2-y1)/(n*n); return inside, area
R = {}; HM = {}
for step in STEPS:
    rows = []
    for seed in SEEDS:
        b = box(step, seed)
        if b is None: print("no face", step, seed); continue
        d = torch.load(f'{H}/norms_step{step}_seed{seed}.pt'); per_call = []
        for name, calls in d.items():
            for ci, (tn, ipn) in enumerate(calls):
                ci_, ca = conc(ipn, b); ct, _ = conc(tn, b); per_call.append(dict(layer=name, call=ci, ip_in=ci_, text_in=ct, area=ca, res=int(round(math.sqrt(ipn.numel()))), ip_over_text=float(ipn.mean() / tn.mean())))
        rows += per_call
        # heatmap: mean over 32x32 layers of ip norm at the last call (final step), upsampled
        last = [calls[-1][1].view(32, 32) for calls in d.values() if calls[-1][1].numel() == 1024]; HM[(step, seed)] = torch.stack(last).mean(0).numpy()
    R[step] = dict(ip_in_box=float(np.mean([r['ip_in'] for r in rows])), text_in_box=float(np.mean([r['text_in'] for r in rows])), box_area=float(np.mean([r['area'] for r in rows])),
                   ip_in_box_last_call=float(np.mean([r['ip_in'] for r in rows if r['call'] == step - 1])), ip_in_box_first_call=float(np.mean([r['ip_in'] for r in rows if r['call'] == 0])),
                   ip_over_text=float(np.mean([r['ip_over_text'] for r in rows])), n=len(rows))
    print(f"steps={step}: IP-branch mass in face box {R[step]['ip_in_box']:.3f} (first call {R[step]['ip_in_box_first_call']:.3f}, last call {R[step]['ip_in_box_last_call']:.3f}); text-branch {R[step]['text_in_box']:.3f}; box area {R[step]['box_area']:.3f}; ip/text norm ratio {R[step]['ip_over_text']:.2f}", flush=True)
json.dump(R, open(f'{OUT}/summary.json', 'w'), indent=1)
fig, ax = plt.subplots(1, 4, figsize=(16, 4))
ax[0].plot(STEPS, [R[s]['ip_in_box'] for s in STEPS], 'o-', label='IP branch, all steps'); ax[0].plot(STEPS, [R[s]['ip_in_box_last_call'] for s in STEPS], 's--', label='IP branch, last step')
ax[0].plot(STEPS, [R[s]['text_in_box'] for s in STEPS], '^-', label='text branch'); ax[0].plot(STEPS, [R[s]['box_area'] for s in STEPS], 'k:', label='chance (box area)')
ax[0].set_xlabel('number of steps'); ax[0].set_ylabel('fraction of branch output norm inside face box'); ax[0].legend(fontsize=8); ax[0].set_title('LCM-LoRA + IP-Adapter, 3 seeds')
for i, step in enumerate([1, 4, 8]):
    hm = np.mean([HM[(step, s)] for s in SEEDS if (step, s) in HM], 0); ax[i+1].imshow(hm, cmap='magma'); ax[i+1].set_title(f'IP-branch norm, final step, {step}-step'); ax[i+1].axis('off')
plt.tight_layout(); plt.savefig('/workspace/outputs/H1_injection_heatmap.png', dpi=130); print("DONE", flush=True)
