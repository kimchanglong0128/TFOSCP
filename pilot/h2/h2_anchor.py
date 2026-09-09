# H2: identity-anchored initialization. Replace the pure-noise x_T of 1-step LCM sampling with the
# reference latent noised to timestep t_start. anchor=None -> standard pure-noise baseline.
import sys, os; sys.path.insert(0, '/workspace/pilot')
import torch, pandas as pd
from PIL import Image
from diffusers import DiffusionPipeline, LCMScheduler
from diffusers.utils.torch_utils import randn_tensor

OUT = '/workspace/outputs/h2'; os.makedirs(OUT, exist_ok=True)
REF = '/workspace/pilot/refs/person.jpeg'
PROMPT = "a photo of a woman wearing sunglasses, on a beach"
T_STARTS = [None, 999, 959, 899, 799, 199]        # None = pure noise baseline; 199 = positive control
SEEDS = [42, 43, 44]

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                         dtype=torch.float16, variant="fp16", use_safetensors=True).to('cuda')
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
pipe.load_ip_adapter("h94/IP-Adapter", subfolder='sdxl_models', weight_name='ip-adapter_sdxl.bin')
pipe.set_ip_adapter_scale(0.6)

# ---- encode reference to latent z0 (VAE in fp32 to avoid fp16 NaNs) ----
ref_pil = Image.open(REF).convert('RGB')
x = pipe.image_processor.preprocess(ref_pil, height=1024, width=1024).to('cuda', torch.float32)
pipe.vae.to(torch.float32)
with torch.no_grad():
    z0 = pipe.vae.encode(x).latent_dist.mean * pipe.vae.config.scaling_factor
pipe.vae.to(torch.float16)
z0 = z0.to(torch.float16)
print("z0", tuple(z0.shape), "finite:", bool(torch.isfinite(z0).all()))

# ---- sanity: alphas_cumprod at the anchor timesteps (signal amplitude = sqrt(abar)) ----
ac = pipe.scheduler.alphas_cumprod
print({t: round(float(ac[t].sqrt()), 3) for t in T_STARTS if t is not None})

rows = []
for t in T_STARTS:
    for seed in SEEDS:
        g = torch.Generator('cuda').manual_seed(seed)
        noise = randn_tensor(z0.shape, generator=g, device=z0.device, dtype=z0.dtype)
        if t is None:
            latents, ts, tag = noise, None, 'noise'
        else:
            latents = pipe.scheduler.add_noise(z0, noise, torch.tensor([t], device=z0.device))
            ts, tag = [t], f't{t}'
        kw = dict(timesteps=ts) if ts is not None else dict(num_inference_steps=1)
        img = pipe(prompt=PROMPT, guidance_scale=1, ip_adapter_image=ref_pil,
                   latents=latents, generator=g, **kw).images[0]
        path = f'{OUT}/woman_{tag}_seed{seed}.png'; img.save(path)
        rows.append(dict(anchor=tag, t_start=(-1 if t is None else t), seed=seed, path=path))
        print("done", tag, seed)

pd.DataFrame(rows).to_csv(f'{OUT}/runs.csv', index=False)
