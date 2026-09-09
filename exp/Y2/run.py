# Scheme Y2: make pass-1 cheap or free. (a) xseed: layout from a DIFFERENT seed (-> offline layout bank, 0 extra NFE)
# (b) lowres: pass-1 text-only 1-step at 512x512 (~1/4 NFE), latent bilinearly upsampled to 128x128, re-noised to t999.
import sys; sys.path.insert(0, '/workspace/exp')
import torch, torch.nn.functional as F, json
import common
from common import *
common.SEEDS = list(range(42, 52)); common.REQ['arcface_min'] = 0.144
PROMPT_CU = "a close-up portrait photo of a woman wearing sunglasses, on a beach"
pipe = load_pipe(adapter='face'); cache = {}
def layout(s, res=1024):
    key = (s, res)
    if key not in cache:
        pipe.set_ip_adapter_scale(0.0); g = torch.Generator('cuda').manual_seed(s)
        z = pipe(prompt=PROMPT_CU, num_inference_steps=1, guidance_scale=1, ip_adapter_image=REF, generator=g,
                 height=res, width=res, output_type='latent').images.float()
        pipe.set_ip_adapter_scale(0.6); assert torch.isfinite(z).all()
        if res != 1024: z = F.interpolate(z, size=(128, 128), mode='bilinear', align_corners=False)
        cache[key] = z
    return cache[key]
SEEDS = common.SEEDS
results = {}
results['Y2_xseed_t999'] = run_scheme('Y2_xseed_t999', pipe, lambda s: anchor(pipe, layout(SEEDS[(SEEDS.index(s)+1) % len(SEEDS)]), noise_for(s), 999),
                                      '/workspace/exp/Y2/Y2_xseed_t999', timesteps=[999], prompt=PROMPT_CU, archive_check=False,
                                      setup_base=lambda: pipe.set_ip_adapter_scale(0.6), note="layout from the NEXT seed (offline-bank test), t999")
for res in [512, 768]:
    name = f"Y2_lowres{res}_t999"
    results[name] = run_scheme(name, pipe, lambda s, r=res: anchor(pipe, layout(s, r), noise_for(s), 999), f'/workspace/exp/Y2/{name}',
                               timesteps=[999], prompt=PROMPT_CU, archive_check=False, setup_base=lambda: pipe.set_ip_adapter_scale(0.6),
                               note=f"pass-1 at {res}px (~{(res/1024)**2:.2f} NFE), upsampled latent, same seed, t999")
json.dump({k: dict(passed=v['passed'], summary=v['summary'], delta=v['delta_vs_baseline']) for k, v in results.items()},
          open('/workspace/exp/Y2/results.json', 'w'), indent=2)
print("ANY_PASSED:", any(v['passed'] for v in results.values()))
