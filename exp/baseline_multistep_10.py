import sys; sys.path.insert(0, '/workspace/exp')
import json, numpy as np, torch
from PIL import Image
from common import *
pipe = load_pipe(); os.makedirs('/workspace/exp/baseline2', exist_ok=True)
# (1) reproduction check: 2-step seed 42 WITHOUT explicit latents must equal the archived image
g = torch.Generator('cuda').manual_seed(42)
img = pipe(prompt=PROMPT, num_inference_steps=2, guidance_scale=1, ip_adapter_image=REF, generator=g).images[0]
arch = np.asarray(Image.open('/workspace/outputs/ipa_seeds/ref_woman/woman_step2_seed42.png'))
print("2-step seed42 (pipeline-drawn latents) == archive:", np.array_equal(np.asarray(img), arch))
g = torch.Generator('cuda').manual_seed(42)
img2 = pipe(prompt=PROMPT, num_inference_steps=2, guidance_scale=1, ip_adapter_image=REF, latents=noise_for(42), generator=g).images[0]
print("2-step seed42 (explicit latents)        == archive:", np.array_equal(np.asarray(img2), arch), "| mean|diff| vs archive:", np.abs(np.asarray(img2).astype(float)-arch).mean().round(2))
# (2) 10-seed multi-step baselines the archive way (no explicit latents)
out = {}
for steps in [2, 8]:
    vals = []
    for s in range(42, 52):
        g = torch.Generator('cuda').manual_seed(s)
        im = pipe(prompt=PROMPT, num_inference_steps=steps, guidance_scale=1, ip_adapter_image=REF, generator=g).images[0]
        p = f'/workspace/exp/baseline2/archway_step{steps}_seed{s}.png'; im.save(p); vals.append(score(p))
    a = np.array([v[1] for v in vals]); d = np.array([v[0] for v in vals])
    out[f'step{steps}'] = dict(arcface_mean=float(np.nanmean(a)), arcface_std=float(np.nanstd(a, ddof=1)), n_nan=int(np.isnan(a).sum()), dino_mean=float(d.mean()), per_seed_arcface=[round(float(x), 3) for x in a])
json.dump(out, open('/workspace/exp/baseline_multistep_10.json', 'w'), indent=2); print(json.dumps(out, indent=1))
