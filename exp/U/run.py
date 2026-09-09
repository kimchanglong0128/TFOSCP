# Scheme U: face-specific adapter (ip-adapter-plus-face_sdxl_vit-h). 10-seed baselines at 1/2/8 steps (pipeline-drawn latents).
import sys, os; sys.path.insert(0, '/workspace/exp')
import torch, json, numpy as np
from PIL import Image
from common import *
pipe = load_pipe(adapter='face')
print("image encoder:", type(pipe.image_encoder).__name__, tuple(pipe.image_encoder.config.image_size for _ in [0]), "| proj dim", pipe.image_encoder.config.projection_dim)
OUT = '/workspace/exp/U'; out = {}
for steps in [1, 2, 8]:
    vals = []
    for s in range(42, 52):
        g = torch.Generator('cuda').manual_seed(s)
        img = pipe(prompt=PROMPT, num_inference_steps=steps, guidance_scale=1, ip_adapter_image=REF, generator=g).images[0]
        check_image(img); p = f'{OUT}/face_step{steps}_seed{s}.png'; img.save(p); vals.append(score(p))
    a = np.array([v[1] for v in vals]); d = np.array([v[0] for v in vals])
    out[f'step{steps}'] = dict(arcface_mean=float(np.nanmean(a)), arcface_std=float(np.nanstd(a, ddof=1)), n_nan=int(np.isnan(a).sum()),
                               dino_mean=float(d.mean()), per_seed_arcface=[round(float(x), 3) for x in a])
    print(f"face adapter step{steps}: arcface {out[f'step{steps}']['arcface_mean']:.3f}±{out[f'step{steps}']['arcface_std']:.3f} nan={out[f'step{steps}']['n_nan']} dino {out[f'step{steps}']['dino_mean']:.3f} per-seed {out[f'step{steps}']['per_seed_arcface']}")
out['vram_gb'] = vram_check()
json.dump(out, open(f'{OUT}/results.json', 'w'), indent=2)
