# Scheme N: a SINGLE sharp identity-free offline latent (text-only 8-step, seed 100/101/102) as anchor. 1 NFE.
# Tests whether structure *sharpness* (not just presence) drives the gain (dog anchor +0.13 vs mean prior +0.05).
import sys, os; sys.path.insert(0, '/workspace/exp')
import torch, json, numpy as np, cv2
from common import *
pipe = load_pipe()
def decode(z):
    pipe.vae.to(torch.float32)
    with torch.no_grad(): img = pipe.vae.decode(z.to(torch.float32) / pipe.vae.config.scaling_factor).sample
    pipe.vae.to(torch.float16); return pipe.image_processor.postprocess(img, output_type='pil')[0]
priors = {}
pipe.set_ip_adapter_scale(0.0)
for ps in [100, 101, 102]:
    path = f'/workspace/exp/N/prior_seed{ps}.pt'
    if os.path.exists(path): z = torch.load(path).cuda()
    else:
        g = torch.Generator('cuda').manual_seed(ps)
        z = pipe(prompt=PROMPT, num_inference_steps=8, guidance_scale=1, ip_adapter_image=REF, generator=g, output_type='latent').images.float()
        torch.save(z.cpu(), path)
    assert torch.isfinite(z).all(); priors[ps] = z
    decode(z).save(f'/workspace/exp/N/prior_seed{ps}.png')
    print(f"prior seed{ps}: vs ref dino/arcface = {m.score(f'/workspace/exp/N/prior_seed{ps}.png', 'woman')}")
pipe.set_ip_adapter_scale(0.6)
assert pipe.scheduler.alphas_cumprod.device.type == 'cpu'
results = {}
for ps in [100, 101, 102]:
    name = f"N_p{ps}_t999"
    results[name] = run_scheme(name, pipe, lambda s, z=priors[ps]: anchor(pipe, z, noise_for(s), 999), f'/workspace/exp/N/{name}',
                               timesteps=[999], note=f"single sharp prior (text-only 8-step seed {ps}) anchored at t999 (7%)")
# variant: sharp prior + ip scale 0.9
pipe.set_ip_adapter_scale(0.9)
name = "N_p100_t999_s0.9"
results[name] = run_scheme(name, pipe, lambda s, z=priors[100]: anchor(pipe, z, noise_for(s), 999), f'/workspace/exp/N/{name}',
                           timesteps=[999], note="prior seed100 t999 + ip scale 0.9", archive_check=False)
pipe.set_ip_adapter_scale(0.6)
json.dump({k: dict(passed=v['passed'], summary=v['summary'], delta=v['delta_vs_baseline']) for k, v in results.items()},
          open('/workspace/exp/N/results.json', 'w'), indent=2)
print("ANY_PASSED:", any(v['passed'] for v in results.values()))
