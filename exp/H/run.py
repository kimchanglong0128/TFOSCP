# Scheme H: pure noise, scaled per scheduler, 1 step from a lower start timestep. 0 extra NFE.
import sys; sys.path.insert(0, '/workspace/exp')
import torch, json
from common import *
pipe = load_pipe(); ac = pipe.scheduler.alphas_cumprod
results = {}
for t in [959, 899, 799]:
    scale = float((1 - ac[t]).sqrt())
    name = f"H_t{t}"
    results[name] = run_scheme(name, pipe, lambda s, sc=scale: noise_for(s) * sc, f'/workspace/exp/H/{name}',
                               timesteps=[t], note=f"x_t = sqrt(1-abar_{t})*eps = {scale:.4f}*eps, single step at t={t}")
json.dump({k: dict(passed=v['passed'], summary=v['summary'], delta=v['delta_vs_baseline']) for k, v in results.items()},
          open('/workspace/exp/H/results.json', 'w'), indent=2)
print("ANY_PASSED:", any(v['passed'] for v in results.values()))
