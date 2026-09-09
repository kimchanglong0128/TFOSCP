# Scheme Y (mechanism reference, 1+1 NFE): text-only 1-step layout (ip scale 0) -> re-noise to t -> 1-step with FACE adapter.
# Tests whether "second step = structure for the query" generalises to the face adapter. Close-up prompt, 10 seeds.
import sys; sys.path.insert(0, '/workspace/exp')
import torch, json
import common
from common import *
common.SEEDS = list(range(42, 52)); common.REQ['arcface_min'] = 0.144
PROMPT_CU = "a close-up portrait photo of a woman wearing sunglasses, on a beach"
pipe = load_pipe(adapter='face')
layouts = {}
def layout(s):
    if s not in layouts:
        pipe.set_ip_adapter_scale(0.0); g = torch.Generator('cuda').manual_seed(s)
        layouts[s] = pipe(prompt=PROMPT_CU, num_inference_steps=1, guidance_scale=1, ip_adapter_image=REF, generator=g, output_type='latent').images.float()
        pipe.set_ip_adapter_scale(0.6); assert torch.isfinite(layouts[s]).all()
    return layouts[s]
results = {}
for t in [999, 959, 899]:
    name = f"Y_face_cu_layout_t{t}"
    results[name] = run_scheme(name, pipe, lambda s, t=t: anchor(pipe, layout(s), noise_for(s), t), f'/workspace/exp/Y/{name}',
                               timesteps=[t], prompt=PROMPT_CU, archive_check=False, setup_base=lambda: pipe.set_ip_adapter_scale(0.6),
                               note=f"pass1 text-only 1-step layout (same seed), re-noised to t={t}, pass2 1-step with face adapter (1+1 NFE)")
json.dump({k: dict(passed=v['passed'], summary=v['summary'], delta=v['delta_vs_baseline']) for k, v in results.items()},
          open('/workspace/exp/Y/results.json', 'w'), indent=2)
print("ANY_PASSED:", any(v['passed'] for v in results.values()))
