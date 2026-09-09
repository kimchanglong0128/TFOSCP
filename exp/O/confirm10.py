# 10-seed confirmation of scheme O (dog-image anchor) vs baseline. Same harness, archive check off for new seeds.
import sys; sys.path.insert(0, '/workspace/exp')
import torch, json, numpy as np
from PIL import Image
import common
from common import *
common.SEEDS = list(range(42, 52))
pipe = load_pipe()
def encode(pil):
    x = pipe.image_processor.preprocess(pil, height=1024, width=1024).to('cuda', torch.float32)
    pipe.vae.to(torch.float32)
    with torch.no_grad(): z = pipe.vae.encode(x).latent_dist.mean * pipe.vae.config.scaling_factor
    pipe.vae.to(torch.float16); return z
zdog = encode(Image.open('/workspace/pilot/refs/dog.jpg').convert('RGB'))
res = {}
for t in [999, 959]:
    name = f"O_dogimg_t{t}_10seeds"
    res[name] = run_scheme(name, pipe, lambda s, t=t: anchor(pipe, zdog, noise_for(s), t), f'/workspace/exp/O/confirm10/t{t}',
                           timesteps=[t], archive_check=False, note="10-seed confirmation")
    print(name, "per-seed arcface:", [round(r['arcface'], 3) for r in res[name]['per_seed'][name]])
    print("baseline per-seed arcface:", [round(r['arcface'], 3) for r in res[name]['per_seed']['baseline']])
json.dump({k: dict(passed=v['passed'], summary=v['summary'], delta=v['delta_vs_baseline']) for k, v in res.items()},
          open('/workspace/exp/O/confirm10/results.json', 'w'), indent=2)
