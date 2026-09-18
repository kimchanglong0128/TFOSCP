# S1 part 3 (post-processing): off-manifold sensitivity ||G(eps')-G(eps)|| / ||eps'-eps|| vs theta, in VAE-latent space, from saved images.
#   Hyper-SD (B2: theta 15..75), DMD2 (B1 A_ref theta 45; N1 rho variants if present), LCM-LoRA / others (T1 theta 45). 10 ids x 20 seeds.
import sys, os, json, glob, math; sys.path.insert(0, '/workspace/exp'); sys.path.insert(0, '/workspace/exp/AD')
import torch, numpy as np
from PIL import Image
import importlib.util as _ilu
_s = _ilu.spec_from_file_location('adrun', '/workspace/exp/AD/run.py'); adrun = _ilu.module_from_spec(_s); _s.loader.exec_module(adrun); rotate_lowfreq = adrun.rotate_lowfreq
from common import noise_for
from diffusers import AutoencoderKL
vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="vae", torch_dtype=torch.float32).to('cuda').eval()
def enc(p):
    x = torch.from_numpy(np.asarray(Image.open(p).convert('RGB'), dtype=np.float32) / 127.5 - 1).permute(2, 0, 1)[None].cuda()
    with torch.no_grad(): return vae.encode(x).latent_dist.mean * vae.config.scaling_factor
IDS = [f'id{i:02d}' for i in range(10)]; SEEDS = range(42, 62); S = {}
def sens(base_fn, meth_fn, theta, src_fn):
    r = []
    for k in IDS:
        for s in SEEDS:
            a, b = base_fn(k, s), meth_fn(k, s)
            if not (os.path.exists(a) and os.path.exists(b)): continue
            x, y = enc(a), enc(b); dz = float((y - x).norm()); z = src_fn(k); de = float((rotate_lowfreq(noise_for(s), z, theta).float() - noise_for(s).float()).norm())
            r.append(dict(dz=dz, de=de, ratio=dz / de, gnorm=float(x.norm())))
    return dict(n=len(r), dz=float(np.mean([v['dz'] for v in r])), de=float(np.mean([v['de'] for v in r])), ratio=float(np.mean([v['ratio'] for v in r])), gnorm=float(np.mean([v['gnorm'] for v in r]))) if r else None
srcB2 = {k: enc(f'/workspace/exp/B2/gen/{k}_src.png') for k in IDS if os.path.exists(f'/workspace/exp/B2/gen/{k}_src.png')}
srcB1 = {k: enc(f'/workspace/exp/B1/gen/{k}_src_A_ref.png') for k in IDS if os.path.exists(f'/workspace/exp/B1/gen/{k}_src_A_ref.png')}
for th in (15, 25, 35, 45, 60, 75):
    S[f'hyper_th{th}'] = sens(lambda k, s: f'/workspace/exp/B2/gen/{k}_step1_s{s}.png', lambda k, s, th=th: f'/workspace/exp/B2/gen/{k}_th{th}_rho0.15_s{s}.png', th, lambda k: srcB2[k]); print('hyper', th, S[f'hyper_th{th}'], flush=True)
S['dmd2_th45'] = sens(lambda k, s: f'/workspace/exp/B1/gen/{k}_step1_s{s}.png', lambda k, s: f'/workspace/exp/B1/gen/{k}_A_ref_s{s}.png', 45, lambda k: srcB1[k]); print('dmd2 45', S['dmd2_th45'], flush=True)
for g in ('lcm_lora_999', 'hyper_lora_999', 'tcd_lora_999', 'dmd2_4s_999', 'dmd2_1s_999'):
    S[f'{g}_th45'] = sens(lambda k, s, g=g: f'/workspace/exp/T1/gen/{k}_{g}_step1_s{s}.png', lambda k, s, g=g: f'/workspace/exp/T1/gen/{k}_{g}_method_s{s}.png', 45, lambda k: srcB1[k]); print(g, S[f'{g}_th45'], flush=True)   # T1 sources differ slightly from B1 (own boxes): de is approximate
json.dump(S, open('/workspace/exp/S1/summary_b.json', 'w'), indent=1); print("DONE", flush=True)
