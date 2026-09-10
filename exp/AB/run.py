# Scheme AB (from FreeControl): self-attention QUERY injection. Extract Q once from a noise-free scaled latent
# x~ = (1-sigma) z_src at key timestep t_key (one cached forward per source); at generation (1 step, pure noise)
# replace Q in mid/late self-attn layers: q = (1-w) q_self + w q_src. K/V stay dynamic. 1 NFE per sample. 10 seeds.
import sys; sys.path.insert(0, '/workspace/exp')
import torch, torch.nn.functional as F, json, numpy as np
from PIL import Image
import common
from common import *
common.SEEDS = list(range(42, 52)); common.REQ['arcface_min'] = 0.144
PROMPT_CU = "a close-up portrait photo of a woman wearing sunglasses, on a beach"
pipe = load_pipe(adapter='face')
MODE = dict(mode='off', w=1.0, layers=('mid_block', 'up_blocks'))
class QProcessor:
    def __init__(self, name): self.name = name; self.q = None
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kw):
        assert encoder_hidden_states is None
        residual = hidden_states; input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape; hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)
        B = hidden_states.shape[0]
        if attn.group_norm is not None: hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        q = attn.to_q(hidden_states); k = attn.to_k(hidden_states); v = attn.to_v(hidden_states)
        if MODE['mode'] == 'record': self.q = q.detach().clone()
        elif MODE['mode'] == 'inject' and self.q is not None and self.name.startswith(MODE['layers']):
            q = (1 - MODE['w']) * q + MODE['w'] * self.q.expand(B, -1, -1)
        heads = attn.heads; hd = k.shape[-1] // heads
        q = q.view(B, -1, heads, hd).transpose(1, 2); k = k.view(B, -1, heads, hd).transpose(1, 2); v = v.view(B, -1, heads, hd).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(B, -1, heads * hd).to(q.dtype)
        out = attn.to_out[0](out); out = attn.to_out[1](out)
        if input_ndim == 4: out = out.transpose(-1, -2).reshape(b, c, h, w)
        if attn.residual_connection: out = out + residual
        return out / attn.rescale_output_factor
procs = dict(pipe.unet.attn_processors)
for name in list(procs):
    if name.endswith('attn1.processor'): procs[name] = QProcessor(name)
pipe.unet.set_attn_processor(procs)
live = [p for p in pipe.unet.attn_processors.values() if isinstance(p, QProcessor)]
print("self-attn layers:", len(live))
pipe.set_ip_adapter_scale(0.6); MODE['mode'] = 'off'
img_off = generate(pipe, noise_for(42), 42, prompt=PROMPT_CU)
ok = np.array_equal(np.asarray(img_off), np.asarray(Image.open('/workspace/exp/W/closeup_step1_seed42.png')))
print("check A: off-mode == vanilla closeup 1-step seed42:", ok); assert ok
def encode(pil):
    x = pipe.image_processor.preprocess(pil, height=1024, width=1024).to('cuda', torch.float32)
    pipe.vae.to(torch.float32)
    with torch.no_grad(): z = pipe.vae.encode(x).latent_dist.mean * pipe.vae.config.scaling_factor
    pipe.vae.to(torch.float16); return z
sources = {'ref': encode(REF), 'dog': encode(Image.open('/workspace/pilot/refs/dog.jpg').convert('RGB'))}
def record(z_src, sigma, t_key):
    MODE['mode'] = 'record'
    generate(pipe, ((1 - sigma) * z_src).half(), 7, prompt=PROMPT_CU, timesteps=[t_key])     # noise-free scaled latent, one forward
    MODE['mode'] = 'off'
    n = sum(p.q is not None for p in live); assert n == len(live), "record failed"
results = {}
for src_name, sigma, t_key, w, layers in [('ref', 0.5, 661, 1.0, ('mid_block', 'up_blocks')), ('ref', 0.5, 661, 0.5, ('mid_block', 'up_blocks')),
                                          ('ref', 0.25, 499, 1.0, ('mid_block', 'up_blocks')), ('dog', 0.5, 661, 1.0, ('mid_block', 'up_blocks')),
                                          ('ref', 0.5, 661, 1.0, ('up_blocks.1',))]:
    record(sources[src_name], sigma, t_key)
    name = f"AB_{src_name}_s{sigma}_t{t_key}_w{w}_{'late' if len(layers)==2 else 'hires'}"
    results[name] = run_scheme(name, pipe, lambda s: noise_for(s), f'/workspace/exp/AB/{name}', prompt=PROMPT_CU, archive_check=False,
                               setup_base=lambda: MODE.update(mode='off'), setup_scheme=lambda w=w, ly=layers: MODE.update(mode='inject', w=w, layers=ly),
                               note=f"Q from (1-{sigma})*z_{src_name} at t_key={t_key}; injected w={w} into {layers}; K/V dynamic")
    MODE['mode'] = 'off'
    d = np.abs(np.asarray(Image.open(f'/workspace/exp/AB/{name}/{name}_seed42.png')).astype(float) - np.asarray(Image.open(f'/workspace/exp/AB/{name}/baseline_seed42.png')).astype(float)).mean()
    print(f"{name}: injection changed seed42 image by mean|diff| {d:.1f}")
json.dump({k: dict(passed=v['passed'], summary=v['summary'], delta=v['delta_vs_baseline']) for k, v in results.items()},
          open('/workspace/exp/AB/results.json', 'w'), indent=2)
print("ANY_PASSED:", any(v['passed'] for v in results.values()))
