# Scheme T: is the effective ingredient natural high-frequency texture (not layout)? 1 NFE, 10 seeds.
#  (a) doghp: high-pass of dog.jpg (dog - blur24) added to mid-gray -> texture only, no colour layout / subject
#  (b) furtile: a 128px fur crop of dog.jpg tiled over 1024x1024 -> pure natural texture, no subject
import sys; sys.path.insert(0, '/workspace/exp')
import torch, json, numpy as np
from PIL import Image, ImageFilter
import common
from common import *
common.SEEDS = list(range(42, 52))
pipe = load_pipe()
def encode(pil):
    x = pipe.image_processor.preprocess(pil, height=1024, width=1024).to('cuda', torch.float32)
    pipe.vae.to(torch.float32)
    with torch.no_grad(): z = pipe.vae.encode(x).latent_dist.mean * pipe.vae.config.scaling_factor
    pipe.vae.to(torch.float16); return z
dog = Image.open('/workspace/pilot/refs/dog.jpg').convert('RGB').resize((1024, 1024))
hp = np.asarray(dog).astype(np.float32) - np.asarray(dog.filter(ImageFilter.GaussianBlur(24))).astype(np.float32)
doghp = Image.fromarray(np.clip(hp + 128, 0, 255).astype(np.uint8))
crop = dog.crop((420, 420, 548, 548))                                   # fur patch (chest area)
furtile = Image.new('RGB', (1024, 1024))
for y in range(0, 1024, 128):
    for x in range(0, 1024, 128): furtile.paste(crop, (x, y))
anchors = {'doghp': doghp, 'furtile': furtile}; Z = {}
for k, im in anchors.items():
    im.save(f'/workspace/exp/T/anchor_{k}.png'); Z[k] = encode(im); assert torch.isfinite(Z[k]).all()
    print(f"anchor {k}: latent std {float(Z[k].std()):.3f} | vs ref dino/arcface {m.score(f'/workspace/exp/T/anchor_{k}.png','woman')}")
results = {}
for k in anchors:
    name = f"T_{k}_t999"
    results[name] = run_scheme(name, pipe, lambda s, z=Z[k]: anchor(pipe, z, noise_for(s), 999), f'/workspace/exp/T/{name}',
                               timesteps=[999], archive_check=False, note=f"texture-only anchor '{k}' at t999")
json.dump({k: dict(passed=v['passed'], summary=v['summary'], delta=v['delta_vs_baseline']) for k, v in results.items()},
          open('/workspace/exp/T/results.json', 'w'), indent=2)
print("ANY_PASSED:", any(v['passed'] for v in results.values()))
