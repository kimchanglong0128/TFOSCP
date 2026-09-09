# F3: structured initialization WITHOUT identity leakage (two-pass, 1+1 NFE).
#   pass 1: text-only 1-step (ip scale 0) -> layout latent z_L (no reference identity involved)
#   pass 2: x_T = sqrt(abar_t) z_L + sqrt(1-abar_t) eps, then 1-step with IP-Adapter (scale 0.6)
# Derived from H2 (start-point structure). Controls: pure-noise baseline; layout from a DIFFERENT seed.
import sys, os, json; sys.path.insert(0, '/workspace/pilot')
import torch, pandas as pd, numpy as np, cv2
from PIL import Image
from diffusers import DiffusionPipeline, LCMScheduler
from diffusers.utils.torch_utils import randn_tensor

OUT = '/workspace/outputs/f3'; os.makedirs(OUT, exist_ok=True)
REF = Image.open('/workspace/pilot/refs/person.jpeg').convert('RGB')
PROMPT = "a photo of a woman wearing sunglasses, on a beach"
SEEDS = [42, 43, 44]; T_LIST = [999, 959, 899]

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                         dtype=torch.float16, variant="fp16", use_safetensors=True).to('cuda')
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
pipe.load_ip_adapter("h94/IP-Adapter", subfolder='sdxl_models', weight_name='ip-adapter_sdxl.bin')
ac = pipe.scheduler.alphas_cumprod

def decode(z):
    pipe.vae.to(torch.float32)
    with torch.no_grad():
        img = pipe.vae.decode(z.to(torch.float32) / pipe.vae.config.scaling_factor).sample
    pipe.vae.to(torch.float16)
    return pipe.image_processor.postprocess(img, output_type='pil')[0]

# ---- pass 1: layouts (text only) ----
layouts = {}
pipe.set_ip_adapter_scale(0.0)
for seed in SEEDS:
    g = torch.Generator('cuda').manual_seed(seed)
    z = pipe(prompt=PROMPT, num_inference_steps=1, guidance_scale=1, ip_adapter_image=REF,
             generator=g, output_type='latent').images
    layouts[seed] = z
    decode(z).save(f'{OUT}/layout_seed{seed}.png')
print("layouts:", {s: tuple(z.shape) for s, z in layouts.items()}, "finite:", all(torch.isfinite(z).all() for z in layouts.values()))

# ---- pass 2 ----
pipe.set_ip_adapter_scale(0.6)
rows = []
def run(tag, latents, ts, seed, extra):
    g = torch.Generator('cuda').manual_seed(seed)
    kw = dict(timesteps=ts) if ts else dict(num_inference_steps=1)
    img = pipe(prompt=PROMPT, guidance_scale=1, ip_adapter_image=REF, latents=latents, generator=g, **kw).images[0]
    path = f'{OUT}/woman_{tag}_seed{seed}.png'; img.save(path)
    rows.append(dict(cfg=tag, seed=seed, path=path, **extra))

for seed in SEEDS:
    g = torch.Generator('cuda').manual_seed(seed)
    noise = randn_tensor(layouts[seed].shape, generator=g, device='cuda', dtype=torch.float16)
    run('noise', noise, None, seed, dict(t=-1, layout_seed=-1))                      # baseline
    for t in T_LIST:                                                                 # F3 (own-seed layout)
        x = pipe.scheduler.add_noise(layouts[seed], noise, torch.tensor([t], device='cuda'))
        run(f'f3_t{t}', x, [t], seed, dict(t=t, layout_seed=seed))
    other = SEEDS[(SEEDS.index(seed) + 1) % 3]                                       # control: other-seed layout
    x = pipe.scheduler.add_noise(layouts[other], noise, torch.tensor([999], device='cuda'))
    run('f3_t999_xseed', x, [999], seed, dict(t=999, layout_seed=other))

pd.DataFrame(rows).to_csv(f'{OUT}/runs.csv', index=False)
print("runs:", len(rows))
