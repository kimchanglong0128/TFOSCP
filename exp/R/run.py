# Scheme R: what *kind* of structure works? Procedurally synthesised anchors, VAE-encoded, t999, 10 seeds.
#  (a) silhouette: centred dark head oval + shoulders on a plain warm background
#  (b) plain: uniform background only (no subject)  -> control for "low-variance latent"
#  (c) dogblur: dog.jpg blurred (sigma 24 px) -> colour layout without detail
import sys; sys.path.insert(0, '/workspace/exp')
import torch, json, numpy as np
from PIL import Image, ImageDraw, ImageFilter
import common
from common import *
common.SEEDS = list(range(42, 52))
pipe = load_pipe()
def encode(pil):
    x = pipe.image_processor.preprocess(pil, height=1024, width=1024).to('cuda', torch.float32)
    pipe.vae.to(torch.float32)
    with torch.no_grad(): z = pipe.vae.encode(x).latent_dist.mean * pipe.vae.config.scaling_factor
    pipe.vae.to(torch.float16); return z
BG = (168, 96, 48)                                  # warm plain background similar to dog.jpg wall
sil = Image.new('RGB', (1024, 1024), BG); d = ImageDraw.Draw(sil)
d.polygon([(200, 1024), (330, 640), (700, 640), (830, 1024)], fill=(70, 55, 45))   # shoulders
d.ellipse([340, 160, 690, 620], fill=(90, 70, 60))                                   # head
plain = Image.new('RGB', (1024, 1024), BG)
dogblur = Image.open('/workspace/pilot/refs/dog.jpg').convert('RGB').resize((1024, 1024)).filter(ImageFilter.GaussianBlur(24))
anchors = {'silhouette': sil, 'plain': plain, 'dogblur': dogblur}
Z = {}
for k, im in anchors.items():
    im.save(f'/workspace/exp/R/anchor_{k}.png'); Z[k] = encode(im); assert torch.isfinite(Z[k]).all()
    print(f"anchor {k}: latent std {float(Z[k].std()):.3f} | vs ref dino/arcface {m.score(f'/workspace/exp/R/anchor_{k}.png','woman')}")
results = {}
for k in anchors:
    name = f"R_{k}_t999"
    results[name] = run_scheme(name, pipe, lambda s, z=Z[k]: anchor(pipe, z, noise_for(s), 999), f'/workspace/exp/R/{name}',
                               timesteps=[999], archive_check=False, note=f"synthetic anchor '{k}' at t999")
json.dump({k: dict(passed=v['passed'], summary=v['summary'], delta=v['delta_vs_baseline']) for k, v in results.items()},
          open('/workspace/exp/R/results.json', 'w'), indent=2)
print("ANY_PASSED:", any(v['passed'] for v in results.values()))
