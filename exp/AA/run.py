# Scheme AA: reference self-attention K/V injection (MasaCtrl-style) inside the single forward. 1 NFE per sample;
# reference K/V cached ONCE per reference (one UNet forward on the noised reference latent at t_ref).
# Hypothesis: giving the query real keys to attend to (reference structure) inside the same forward substitutes for pass-1.
import sys, math; sys.path.insert(0, '/workspace/exp')
import torch, torch.nn.functional as F, json, numpy as np
from PIL import Image
from diffusers.models.attention_processor import AttnProcessor2_0
import common
from common import *
common.SEEDS = list(range(42, 52)); common.REQ['arcface_min'] = 0.144
PROMPT_CU = "a close-up portrait photo of a woman wearing sunglasses, on a beach"
pipe = load_pipe(adapter='face')

class RefKVProcessor:
    """Self-attention processor: mode 'off' = vanilla; 'record' = store K,V; 'inject' = attend over [self ; lam*ref] keys/values."""
    def __init__(self, name): self.name = name; self.k = None; self.v = None
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kw):
        assert encoder_hidden_states is None, "self-attention only"
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape; hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)
        B, L, _ = hidden_states.shape
        if attn.group_norm is not None: hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        q = attn.to_q(hidden_states); k = attn.to_k(hidden_states); v = attn.to_v(hidden_states)
        if MODE['mode'] == 'record': self.k, self.v = k.detach().clone(), v.detach().clone()
        elif MODE['mode'] == 'inject' and self.k is not None and (MODE['layers'] is None or any(self.name.startswith(p) for p in MODE['layers'])):
            k = torch.cat([k, MODE['lam'] * self.k.expand(B, -1, -1)], dim=1); v = torch.cat([v, self.v.expand(B, -1, -1)], dim=1)
        heads = attn.heads; hd = k.shape[-1] // heads
        q = q.view(B, -1, heads, hd).transpose(1, 2); k = k.view(B, -1, heads, hd).transpose(1, 2); v = v.view(B, -1, heads, hd).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(B, -1, heads * hd).to(q.dtype)
        out = attn.to_out[0](out); out = attn.to_out[1](out)
        if input_ndim == 4: out = out.transpose(-1, -2).reshape(b, c, h, w)
        if attn.residual_connection: out = out + residual
        return out / attn.rescale_output_factor

MODE = dict(mode='off', lam=1.0, layers=None)
procs = dict(pipe.unet.attn_processors); n = 0
for name, p in procs.items():
    if name.endswith('attn1.processor'): procs[name] = RefKVProcessor(name); n += 1
pipe.unet.set_attn_processor(procs); print("self-attn layers replaced:", n)

# ---- check A: 'off' mode must reproduce the vanilla 1-step image bit-for-bit ----
pipe.set_ip_adapter_scale(0.6); MODE['mode'] = 'off'
img_off = generate(pipe, noise_for(42), 42, prompt=PROMPT_CU)
ref_img = np.asarray(Image.open('/workspace/exp/W/closeup_step1_seed42.png'))
print("check A: off-mode == vanilla closeup 1-step seed42:", np.array_equal(np.asarray(img_off), ref_img))
assert np.array_equal(np.asarray(img_off), ref_img), "custom self-attn processor is not equivalent"

# ---- record reference K/V once: UNet forward at t_ref on the noised reference latent ----
def encode(pil):
    x = pipe.image_processor.preprocess(pil, height=1024, width=1024).to('cuda', torch.float32)
    pipe.vae.to(torch.float32)
    with torch.no_grad(): z = pipe.vae.encode(x).latent_dist.mean * pipe.vae.config.scaling_factor
    pipe.vae.to(torch.float16); return z
z_ref = encode(REF)
def record(t_ref):
    MODE['mode'] = 'record'
    generate(pipe, anchor(pipe, z_ref, noise_for(7), t_ref), 7, prompt=PROMPT_CU, timesteps=[t_ref])   # one forward, seed 7 noise
    MODE['mode'] = 'off'
    shapes = {p.name: tuple(p.k.shape) for p in procs.values() if isinstance(p, RefKVProcessor)}
    print(f"recorded ref K/V at t={t_ref}: {len(shapes)} layers, e.g.", list(shapes.items())[:2])

results = {}
for t_ref, lam, layers, tag in [(999, 1.0, None, 'all_t999'), (799, 1.0, None, 'all_t799'), (799, 1.0, ('down_blocks.1', 'up_blocks.1'), 'hires_t799'), (799, 0.5, None, 'all_t799_lam0.5')]:
    record(t_ref)
    name = f"AA_refkv_{tag}"
    def setup_scheme(l=lam, ly=layers): MODE.update(mode='inject', lam=l, layers=ly)
    def setup_base(): MODE.update(mode='off')
    results[name] = run_scheme(name, pipe, lambda s: noise_for(s), f'/workspace/exp/AA/{name}', prompt=PROMPT_CU, archive_check=False,
                               setup_base=setup_base, setup_scheme=setup_scheme, note=f"ref self-attn K/V (cached, t_ref={t_ref}) injected, lam={lam}, layers={layers}")
    MODE['mode'] = 'off'
json.dump({k: dict(passed=v['passed'], summary=v['summary'], delta=v['delta_vs_baseline']) for k, v in results.items()},
          open('/workspace/exp/AA/results.json', 'w'), indent=2)
print("ANY_PASSED:", any(v['passed'] for v in results.values()))
