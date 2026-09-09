# Scheme Z: IP scale sweep at 1 step, FACE adapter, close-up prompt. 1 NFE, 10 seeds. Baseline = scale 0.6 (fixed via setup_base).
import sys; sys.path.insert(0, '/workspace/exp')
import torch, json
import common
from common import *
common.SEEDS = list(range(42, 52)); common.REQ['arcface_min'] = 0.144
PROMPT_CU = "a close-up portrait photo of a woman wearing sunglasses, on a beach"
pipe = load_pipe(adapter='face'); results = {}
for scale in [0.4, 0.8, 1.0, 1.2]:
    name = f"Z_face_cu_s{scale}"
    results[name] = run_scheme(name, pipe, lambda s: noise_for(s), f'/workspace/exp/Z/{name}', prompt=PROMPT_CU, archive_check=False,
                               setup_base=lambda: pipe.set_ip_adapter_scale(0.6), setup_scheme=lambda sc=scale: pipe.set_ip_adapter_scale(sc),
                               note=f"close-up prompt, face adapter, 1 step, ip scale {scale} (baseline scale 0.6)")
json.dump({k: dict(passed=v['passed'], summary=v['summary'], delta=v['delta_vs_baseline']) for k, v in results.items()},
          open('/workspace/exp/Z/results.json', 'w'), indent=2)
print("ANY_PASSED:", any(v['passed'] for v in results.values()))
