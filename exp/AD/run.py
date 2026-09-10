# Scheme AD (from MoNO): rotate the LOW-frequency part of the initial noise toward a structure source on the
# fixed-radius sphere (||z_L|| kept, z_H kept) -> stays on the Gaussian prior manifold. 0 extra NFE. 10 seeds.
import sys, math; sys.path.insert(0, '/workspace/exp')
import torch, json, numpy as np
from PIL import Image
import common
from common import *
common.SEEDS = list(range(42, 52))
RHO = 0.15
def lowpass_mask(h, w, rho, device):
    fy = torch.fft.fftfreq(h, device=device)[:, None]; fx = torch.fft.fftfreq(w, device=device)[None, :]
    return ((fy**2 + fx**2).sqrt() <= rho * 0.5).float()          # radial cutoff, rho of Nyquist
def split(z, rho):
    m = lowpass_mask(z.shape[-2], z.shape[-1], rho, z.device)
    Z = torch.fft.fft2(z.float()); zl = torch.fft.ifft2(Z * m).real; return zl, z.float() - zl
def rotate_lowfreq(eps, src, theta_deg, rho=RHO):
    eL, eH = split(eps, rho); sL, _ = split(src, rho)
    r = eL.norm(); ehat = eL / r
    s_perp = sL - (sL * ehat).sum() * ehat; s_perp = s_perp / s_perp.norm().clamp_min(1e-8)
    th = math.radians(theta_deg); u = r * (math.cos(th) * ehat + math.sin(th) * s_perp)
    out = eH + u
    assert abs(float(out.norm()) - float(eps.float().norm())) / float(eps.float().norm()) < 1e-3, "norm not preserved"
    return out.to(eps.dtype)

def encode(pipe, pil):
    x = pipe.image_processor.preprocess(pil, height=1024, width=1024).to('cuda', torch.float32)
    pipe.vae.to(torch.float32)
    with torch.no_grad(): z = pipe.vae.encode(x).latent_dist.mean * pipe.vae.config.scaling_factor
    pipe.vae.to(torch.float16); return z

if __name__ == '__main__':
    PROMPT_CU = "a close-up portrait photo of a woman wearing sunglasses, on a beach"
    results = {}
    for adapter, prompt, req in [('face', PROMPT_CU, 0.144), ('base', None, 0.069)]:
        pipe = load_pipe(adapter=adapter); common.REQ['arcface_min'] = req
        sources = {'dog': encode(pipe, Image.open('/workspace/pilot/refs/dog.jpg').convert('RGB')),
                   'ref': encode(pipe, REF)}
        for src_name, src in sources.items():
            for theta in ([45, 90] if adapter == 'face' else [90]):
                name = f"AD_{adapter}_{src_name}_th{theta}"
                results[name] = run_scheme(name, pipe, lambda s, src=src, th=theta: rotate_lowfreq(noise_for(s), src, th),
                                           f'/workspace/exp/AD/{name}', prompt=prompt, archive_check=False,
                                           note=f"{adapter} adapter; low-freq (rho={RHO}) of eps rotated {theta} deg toward '{src_name}' latent on fixed-radius sphere; z_H kept")
        del pipe; torch.cuda.empty_cache()
    json.dump({k: dict(passed=v['passed'], summary=v['summary'], delta=v['delta_vs_baseline']) for k, v in results.items()},
              open('/workspace/exp/AD/results.json', 'w'), indent=2)
    print("ANY_PASSED:", any(v['passed'] for v in results.values()))
