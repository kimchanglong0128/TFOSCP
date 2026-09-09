# H1: capture text-branch vs IP-branch outputs of every IP-Adapter cross-attention layer.
# SDXL + LCM-LoRA + LCMScheduler + IP-Adapter, with the classic (non-SDPA) IP processor so we can hook it.
from diffusers import DiffusionPipeline, LCMScheduler
from diffusers.models.attention_processor import IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0
import torch
import os
from PIL import Image

os.makedirs('/workspace/outputs/h1', exist_ok=True)

# ---------- 1. load ----------
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                         dtype=torch.float16, variant="fp16", use_safetensors=True)
pipe.to('cuda')
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
pipe.load_ip_adapter("h94/IP-Adapter", subfolder='sdxl_models', weight_name='ip-adapter_sdxl.bin')
pipe.set_ip_adapter_scale(0.6)

# ---------- 2. swap IP processors: 2_0 (SDPA) -> classic, keeping trained ip_k/ip_v weights ----------
procs = {}
for name, p in pipe.unet.attn_processors.items():
    if isinstance(p, IPAdapterAttnProcessor2_0):
        q = IPAdapterAttnProcessor(hidden_size=p.hidden_size, cross_attention_dim=p.cross_attention_dim,
                                   num_tokens=p.num_tokens, scale=p.scale).to(pipe.unet.device, dtype=pipe.unet.dtype)
        q.load_state_dict(p.state_dict())
        procs[name] = q
    else:
        procs[name] = p
pipe.unet.set_attn_processor(procs)
n_ip = sum(isinstance(p, IPAdapterAttnProcessor) for p in pipe.unet.attn_processors.values())
print("IP processors swapped:", n_ip)          # expect 70

# ---------- 3. hooks: batch_to_head_dim is called twice per IP layer (text branch, then IP branch) ----------
STORE = {}   # layer name -> list of (text_out, ip_out), one entry per forward call (= per step)
def hook_attn(name, attn):
    orig = attn.batch_to_head_dim
    calls = []
    def patched(x):
        out = orig(x)
        calls.append(out.detach().float().norm(dim=-1)[0].cpu())   # (HW,) per-pixel norm
        if len(calls) == 2:
            STORE.setdefault(name, []).append((calls[0], calls[1]))
            calls.clear()
        return out
    attn.batch_to_head_dim = patched

for name, p in pipe.unet.attn_processors.items():
    if isinstance(p, IPAdapterAttnProcessor):
        hook_attn(name, pipe.unet.get_submodule(name.replace('.processor', '')))

# ---------- 4. generate: steps x seeds, dump per-pixel norms for every layer/step ----------
for step in [8, 6, 4, 3, 2, 1]:
    for seed in [42, 43, 44]:
        STORE.clear()
        generator = torch.Generator('cuda').manual_seed(seed)
        image = pipe(
            prompt="a photo of a woman wearing sunglasses, on a beach",
            num_inference_steps=step,
            guidance_scale=1,
            ip_adapter_image=Image.open('/workspace/pilot/refs/person.jpeg').convert('RGB'),
            generator=generator,
        ).images[0]
        image.save(f'/workspace/outputs/h1/woman_step{step}_seed{seed}.png')
        torch.save(dict(STORE), f'/workspace/outputs/h1/norms_step{step}_seed{seed}.pt')
        print(f"step {step} seed {seed}: {len(STORE)} layers, {len(STORE[next(iter(STORE))])} calls")
