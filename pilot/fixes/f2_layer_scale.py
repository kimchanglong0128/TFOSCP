# F2: per-resolution IP-Adapter scale at 1 step. High-res layers (64x64: down_blocks.1, up_blocks.1)
# carry fine detail (identity); low-res layers (32x32) carry layout. Derived from branch-1 (error in high freq).
import sys, os; sys.path.insert(0, '/workspace/pilot')
import torch, pandas as pd
from PIL import Image
from diffusers import DiffusionPipeline, LCMScheduler
from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0

OUT = '/workspace/outputs/f2'; os.makedirs(OUT, exist_ok=True)
REF = Image.open('/workspace/pilot/refs/person.jpeg').convert('RGB')
PROMPT = "a photo of a woman wearing sunglasses, on a beach"
CONFIGS = {            # name: (high_res_scale, low_res_scale)
    'uni0.6':   (0.6, 0.6),   # baseline
    'uni0.9':   (0.9, 0.9),   # more scale everywhere (control for "just more")
    'hi0.9lo0.4': (0.9, 0.4), # identity-heavy: high-res up, low-res down
    'hi0.3lo0.8': (0.3, 0.8), # reverse control
}
SEEDS = [42, 43, 44]
HIGH_RES = ('down_blocks.1', 'up_blocks.1')

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                         dtype=torch.float16, variant="fp16", use_safetensors=True).to('cuda')
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
pipe.load_ip_adapter("h94/IP-Adapter", subfolder='sdxl_models', weight_name='ip-adapter_sdxl.bin')

def set_scales(hi, lo):
    n_hi = n_lo = 0
    for name, p in pipe.unet.attn_processors.items():
        if isinstance(p, IPAdapterAttnProcessor2_0):
            if name.startswith(HIGH_RES):
                p.scale = [hi]; n_hi += 1
            else:
                p.scale = [lo]; n_lo += 1
    return n_hi, n_lo

rows = []
for cfg, (hi, lo) in CONFIGS.items():
    n_hi, n_lo = set_scales(hi, lo)
    for seed in SEEDS:
        g = torch.Generator('cuda').manual_seed(seed)
        img = pipe(prompt=PROMPT, num_inference_steps=1, guidance_scale=1,
                   ip_adapter_image=REF, generator=g).images[0]
        path = f'{OUT}/woman_{cfg}_seed{seed}.png'; img.save(path)
        rows.append(dict(cfg=cfg, hi=hi, lo=lo, seed=seed, path=path))
    print(f"{cfg}: high-res layers={n_hi}, low-res layers={n_lo}")
pd.DataFrame(rows).to_csv(f'{OUT}/runs.csv', index=False)
