# Card S1 (no generator needed): statistics and numerical checks for the theory section.
# 1. Proposition 1: KL(q_theta || N) of the rotated low-frequency direction. (a) exact formula (d_L-2) E[log sin a - log sin|a-theta|] vs closed-form -(d_L-2) log cos theta,
#    Monte Carlo over the real d_L; (b) independent check for small d by histogram KL of the rotated angle vs the uniform-direction angle density sin^(d-2).
# 2. Non-radial statistics of eps' (ReNoise-type criteria): adjacent-pixel correlation and 8x8 patch mean/variance vs theta; compared with
#    Colorful-Noise-style band replacement (eps_L <- gamma z_L) and un-normalised linear blend.
# 4. Frequency locality of the generator (Prop. 2 consequence): relative change of x0 in the low vs high band between baseline and method images (B1, DMD2).
import sys; sys.path.insert(0, '/workspace/exp'); sys.path.insert(0, '/workspace/exp/AD')
import os, json, math, glob, torch, numpy as np
from PIL import Image
import importlib.util as _ilu
_s = _ilu.spec_from_file_location('adrun', '/workspace/exp/AD/run.py'); adrun = _ilu.module_from_spec(_s); _s.loader.exec_module(adrun)
rotate_lowfreq, split, lowpass_mask = adrun.rotate_lowfreq, adrun.split, adrun.lowpass_mask
from common import noise_for
from diffusers import AutoencoderKL
OUT = '/workspace/exp/S1'; RHO = 0.15; torch.manual_seed(0); S = {}
mask = lowpass_mask(128, 128, RHO, 'cuda'); dL = int(mask.sum().item()) * 4; S['d_L'] = dL; print("d_L =", dL, "(bins per channel", int(mask.sum().item()), ")")
# ---- 1a: MC of the exact formula vs closed form at the real d_L ----
def kl_formula(d, theta, n=200000):
    g = torch.randn(n, d, device='cuda'); s = torch.zeros(d, device='cuda'); s[0] = 1
    cosa = (g @ s) / g.norm(dim=1); a = torch.acos(cosa.clamp(-1, 1)); th = math.radians(theta)
    return float((d - 2) * (torch.log(torch.sin(a)) - torch.log(torch.sin((a - th).abs()).clamp_min(1e-12))).mean())
S['prop1'] = {}
for theta in (15, 30, 45, 60, 75):
    exact = kl_formula(dL, theta); closed = -(dL - 2) * math.log(math.cos(math.radians(theta)))
    S['prop1'][theta] = dict(kl_exact_mc=exact, kl_closed=closed, ratio=exact / closed); print(f"theta={theta:2d} KL exact(MC) {exact:9.1f}  closed -(dL-2)log cos {closed:9.1f}  ratio {exact/closed:.4f}")
# ---- 1b: small-d independent check: histogram KL of rotated angle vs sin^(d-2) density ----
S['prop1_smalld'] = {}
for d in (3, 10, 50, 200):
    for theta in (30, 45):
        n = 2000000 if d <= 50 else 500000; g = torch.randn(n, d); s = torch.zeros(d); s[0] = 1; e = g / g.norm(dim=1, keepdim=True)   # CPU: avoid GPU contention
        sp = s - (e @ s)[:, None] * e; sp = sp / sp.norm(dim=1, keepdim=True).clamp_min(1e-12); th = math.radians(theta)
        u = math.cos(th) * e + math.sin(th) * sp; ang = torch.acos((u @ s).clamp(-1, 1)).cpu().numpy()          # rotated angle to s
        bins = np.linspace(0, math.pi, 400); h, _ = np.histogram(ang, bins, density=True); c = 0.5 * (bins[1:] + bins[:-1])
        p = np.sin(c) ** (d - 2); p /= (p * np.diff(bins)).sum(); nz = h > 0
        kl_hist = float((h[nz] * np.log(h[nz] / p[nz]) * np.diff(bins)[nz]).sum()); kl_f = kl_formula(d, theta, n=400000); closed = -(d - 2) * math.log(math.cos(th))
        S['prop1_smalld'][f'd{d}_th{theta}'] = dict(kl_hist=kl_hist, kl_formula=kl_f, kl_closed=closed); print(f"d={d:3d} theta={theta} KL hist {kl_hist:.4f} formula {kl_f:.4f} closed {closed:.4f}")
# ---- 2: non-radial statistics of eps' ----
vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="vae", torch_dtype=torch.float32).to('cuda').eval()
def enc(p):
    x = torch.from_numpy(np.asarray(Image.open(p).convert('RGB').resize((1024, 1024)), dtype=np.float32) / 127.5 - 1).permute(2, 0, 1)[None].cuda()
    with torch.no_grad(): return vae.encode(x).latent_dist.mean * vae.config.scaling_factor
srcs = sorted(glob.glob('/workspace/exp/B1/gen/id*_src_A_ref.png'))[:10]; Z = [enc(p) for p in srcs]; print("sources:", len(Z))
def nonradial(e):
    e = e.float()[0]; c = torch.cat([(e[:, :, 1:] * e[:, :, :-1]).flatten(), (e[:, 1:, :] * e[:, :-1, :]).flatten()]).mean() / e.var()
    pm = e.unfold(1, 8, 8).unfold(2, 8, 8).mean(dim=(-1, -2)); pv = e.unfold(1, 8, 8).unfold(2, 8, 8).var(dim=(-1, -2))
    return dict(adj_corr=float(c), patch_mean_var=float(pm.var()), patch_var_mean=float(pv.mean()), global_std=float(e.std()), norm=float(e.norm()))
rows = {}
for name, fn in [('gauss', lambda e, z: e)] + [(f'rot{t}', (lambda t: lambda e, z: rotate_lowfreq(e, z, t, RHO))(t)) for t in (15, 30, 45, 60, 75)] + \
                [(f'replace_g{g}', (lambda g: lambda e, z: (split(e, RHO)[1] + g * split(z, RHO)[0]).to(e.dtype))(g)) for g in (0.5, 1.0)] + \
                [(f'blend{l}', (lambda l: lambda e, z: ((1 - l) * e.float() + l * z.float()).to(e.dtype))(l)) for l in (0.3, 0.5)]:
    v = [nonradial(fn(noise_for(42 + i), Z[i % len(Z)])) for i in range(20)]; rows[name] = {k: float(np.mean([x[k] for x in v])) for k in v[0]}
    print(f"{name:12s} adj_corr {rows[name]['adj_corr']:+.4f} patch_mean_var {rows[name]['patch_mean_var']:.4f} (iid: {1/64:.4f}) patch_var_mean {rows[name]['patch_var_mean']:.3f} std {rows[name]['global_std']:.3f} norm {rows[name]['norm']:.1f}")
S['nonradial'] = rows
# ---- 4: frequency locality of x0 change (B1 images: step1 vs A_ref, same seed) ----
loc = []
for k in [f'id{i:02d}' for i in range(30)]:
    for s in range(42, 62):
        a, b = f'/workspace/exp/B1/gen/{k}_step1_s{s}.png', f'/workspace/exp/B1/gen/{k}_A_ref_s{s}.png'
        if not (os.path.exists(a) and os.path.exists(b)): continue
        x, y = enc(a), enc(b); xl, xh = split(x, RHO); yl, yh = split(y, RHO)
        loc.append(dict(rel_low=float((yl - xl).norm() / xl.norm()), rel_high=float((yh - xh).norm() / xh.norm()), cos_low=float((xl * yl).sum() / (xl.norm() * yl.norm())), cos_high=float((xh * yh).sum() / (xh.norm() * yh.norm()))))
    if len(loc) >= 200: break
S['freq_locality'] = {k: float(np.mean([r[k] for r in loc])) for k in loc[0]}; S['freq_locality']['n_pairs'] = len(loc)
print("x0 change (method vs baseline, DMD2, n=%d): rel change low band %.3f, high band %.3f; cos low %.3f, cos high %.3f" % (len(loc), *[S['freq_locality'][k] for k in ('rel_low', 'rel_high', 'cos_low', 'cos_high')]))
json.dump(S, open(f'{OUT}/summary.json', 'w'), indent=1); print("DONE", flush=True)
