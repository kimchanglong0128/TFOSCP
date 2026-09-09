import sys; sys.path.insert(0, '/workspace/exp')
import json, numpy as np, torch
from common import *
pipe = load_pipe(); os.makedirs('/workspace/exp/baseline2', exist_ok=True)
out = {}
for steps in [2, 1]:
    vals = []
    for s in range(42, 52):
        g = torch.Generator('cuda').manual_seed(s)
        img = pipe(prompt=PROMPT, num_inference_steps=steps, guidance_scale=1, ip_adapter_image=REF, latents=noise_for(s), generator=g).images[0]
        p = f'/workspace/exp/baseline2/step{steps}_seed{s}.png'; img.save(p); vals.append(score(p))
    d = np.array([v[0] for v in vals]); a = np.array([v[1] for v in vals])
    out[f'step{steps}'] = dict(arcface_mean=float(np.nanmean(a)), arcface_std=float(np.nanstd(a, ddof=1)), n_nan=int(np.isnan(a).sum()), dino_mean=float(d.mean()), per_seed_arcface=[round(float(x), 3) for x in a])
json.dump(out, open('/workspace/exp/baseline2step_10.json', 'w'), indent=2); print(json.dumps(out, indent=1))
