# Scheme V: dog-image anchor (t999) with the FACE adapter, ip scale 0.6 and 0.8. 1 NFE, 10 seeds.
import sys; sys.path.insert(0, '/workspace/exp')
import torch, json
from PIL import Image
import common
from common import *
common.SEEDS = list(range(42, 52)); common.REQ['arcface_min'] = 0.082
pipe = load_pipe(adapter='face')
def encode(pil):
    x = pipe.image_processor.preprocess(pil, height=1024, width=1024).to('cuda', torch.float32)
    pipe.vae.to(torch.float32)
    with torch.no_grad(): z = pipe.vae.encode(x).latent_dist.mean * pipe.vae.config.scaling_factor
    pipe.vae.to(torch.float16); return z
zdog = encode(Image.open('/workspace/pilot/refs/dog.jpg').convert('RGB'))
results = {}
for scale in [0.6, 0.8]:
    pipe.set_ip_adapter_scale(scale)
    name = f"V_face_dog_t999_s{scale}"
    results[name] = run_scheme(name, pipe, lambda s: anchor(pipe, zdog, noise_for(s), 999), f'/workspace/exp/V/{name}',
                               timesteps=[999], archive_check=False, note=f"face adapter, dog anchor t999, ip scale {scale}; baseline = pure noise 1-step same scale")
json.dump({k: dict(passed=v['passed'], summary=v['summary'], delta=v['delta_vs_baseline']) for k, v in results.items()},
          open('/workspace/exp/V/results.json', 'w'), indent=2)
print("ANY_PASSED:", any(v['passed'] for v in results.values()))
