# Scheme P: stack dog anchor (start structure) + lower start timestep + higher IP scale. 1 NFE. 10 seeds.
import sys; sys.path.insert(0, '/workspace/exp')
import torch, json
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
results = {}
for t, scale in [(999, 0.6), (899, 0.6), (999, 0.9), (899, 0.9), (899, 0.75)]:
    pipe.set_ip_adapter_scale(scale)
    name = f"P_dog_t{t}_s{scale}"
    results[name] = run_scheme(name, pipe, lambda s, t=t: anchor(pipe, zdog, noise_for(s), t), f'/workspace/exp/P/{name}',
                               timesteps=[t], archive_check=False, note=f"dog anchor t={t} (amp {abar(pipe,t)**0.5:.3f}) + ip scale {scale}; baseline = pure noise 1-step at same scale")
json.dump({k: dict(passed=v['passed'], summary=v['summary'], delta=v['delta_vs_baseline']) for k, v in results.items()},
          open('/workspace/exp/P/results.json', 'w'), indent=2)
print("ANY_PASSED:", any(v['passed'] for v in results.values()))
