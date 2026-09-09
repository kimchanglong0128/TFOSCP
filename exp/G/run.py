# Scheme G: spectrally shaped (low-pass enriched) initial noise, variance-renormalised. 0 extra NFE.
import sys; sys.path.insert(0, '/workspace/exp')
import torch, torch.nn.functional as F, math, json
from common import *

def gauss_blur(x, sigma):
    k = int(6 * sigma + 1) | 1; ax = torch.arange(k, dtype=torch.float32, device=x.device) - k // 2
    g = torch.exp(-ax**2 / (2 * sigma**2)); g = g / g.sum()
    w = (g[:, None] * g[None, :])[None, None].repeat(x.shape[1], 1, 1, 1)
    return F.conv2d(x.float(), w, padding=k // 2, groups=x.shape[1])

def shaped_noise(seed, a, sigma):
    eps = noise_for(seed).float()
    x = (1 - a) * eps + a * gauss_blur(eps, sigma)
    x = x / x.flatten(2).std(dim=2)[..., None, None]          # per-channel std -> 1
    assert abs(float(x.std()) - 1.0) < 0.05, f"std after renorm {float(x.std()):.3f}"
    return x.to(torch.float16)

pipe = load_pipe()
results = {}
for a, sigma in [(0.5, 2.0), (0.8, 4.0)]:
    name = f"G_a{a}_s{sigma}"
    results[name] = run_scheme(name, pipe, lambda s: shaped_noise(s, a, sigma), f'/workspace/exp/G/{name}',
                               note=f"x_T = renorm((1-{a})*eps + {a}*blur_sigma{sigma}(eps))")
json.dump({k: dict(passed=v['passed'], summary=v['summary'], delta=v['delta_vs_baseline']) for k, v in results.items()},
          open('/workspace/exp/G/results.json', 'w'), indent=2)
print("ANY_PASSED:", any(v['passed'] for v in results.values()))
