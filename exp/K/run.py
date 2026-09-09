# Scheme K: offline identity-free "mean portrait" latent prior as start-point anchor. 1 NFE per sample.
# prior = mean of N text-only 8-step LCM latents (ip scale 0, seeds 100..107), computed ONCE (amortised).
import sys, os; sys.path.insert(0, '/workspace/exp')
import torch, json, numpy as np, cv2
from common import *
pipe = load_pipe(); ac = pipe.scheduler.alphas_cumprod
PRIOR_SEEDS = list(range(100, 108)); PRIOR_PATH = '/workspace/exp/K/prior.pt'

def decode(z):
    pipe.vae.to(torch.float32)
    with torch.no_grad(): img = pipe.vae.decode(z.to(torch.float32) / pipe.vae.config.scaling_factor).sample
    pipe.vae.to(torch.float16); return pipe.image_processor.postprocess(img, output_type='pil')[0]

if os.path.exists(PRIOR_PATH):
    z_prior = torch.load(PRIOR_PATH).cuda()
else:
    pipe.set_ip_adapter_scale(0.0); zs = []
    for s in PRIOR_SEEDS:
        g = torch.Generator('cuda').manual_seed(s)
        zs.append(pipe(prompt=PROMPT, num_inference_steps=8, guidance_scale=1, ip_adapter_image=REF, generator=g, output_type='latent').images.float())
    z_prior = torch.stack(zs).mean(0); torch.save(z_prior.cpu(), PRIOR_PATH)
    pipe.set_ip_adapter_scale(0.6)
assert torch.isfinite(z_prior).all() and tuple(z_prior.shape) == LATENT_SHAPE, "bad prior"
prior_img = decode(z_prior); prior_img.save('/workspace/exp/K/prior.png')
bb = m.face_bbox(cv2.cvtColor(np.asarray(prior_img), cv2.COLOR_RGB2BGR))
print("prior: std %.3f | face bbox in decoded prior: %s | prior vs ref: dino/arcface %s" % (float(z_prior.std()), None if bb is None else bb.tolist(), m.score('/workspace/exp/K/prior.png', 'woman')))

results = {}
for t in [999, 959]:
    name = f"K_t{t}"
    def mk(s, t=t):
        return anchor(pipe, z_prior, noise_for(s), t)
    results[name] = run_scheme(name, pipe, mk, f'/workspace/exp/K/{name}', timesteps=[t],
                               note=f"x_T = sqrt(abar_{t})*z_prior + sqrt(1-abar_{t})*eps; prior = mean of 8 text-only 8-step latents")
json.dump({k: dict(passed=v['passed'], summary=v['summary'], delta=v['delta_vs_baseline']) for k, v in results.items()},
          open('/workspace/exp/K/results.json', 'w'), indent=2)
print("ANY_PASSED:", any(v['passed'] for v in results.values()))
