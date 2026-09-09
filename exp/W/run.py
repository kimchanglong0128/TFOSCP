# Scheme W: composition control via prompt (close-up) with the FACE adapter. 10 seeds, steps 1/2/8 (pipeline-drawn latents).
# Also records face bbox area to test the "face too small at 1 step" explanation.
import sys, os; sys.path.insert(0, '/workspace/exp')
import torch, json, numpy as np, cv2
from common import *
PROMPT_CU = "a close-up portrait photo of a woman wearing sunglasses, on a beach"
pipe = load_pipe(adapter='face'); OUT = '/workspace/exp/W'; out = {}
for tag, prompt in [('orig', PROMPT), ('closeup', PROMPT_CU)]:
    for steps in [1, 2, 8]:
        vals, areas = [], []
        for s in range(42, 52):
            g = torch.Generator('cuda').manual_seed(s)
            img = pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=1, ip_adapter_image=REF, generator=g).images[0]
            check_image(img); p = f'{OUT}/{tag}_step{steps}_seed{s}.png'; img.save(p); vals.append(score(p))
            bb = m.face_bbox(cv2.imread(p)); areas.append(None if bb is None else int((bb[2]-bb[0])*(bb[3]-bb[1])))
        a = np.array([v[1] for v in vals]); d = np.array([v[0] for v in vals]); ar = np.array([x for x in areas if x is not None], dtype=float)
        out[f'{tag}_step{steps}'] = dict(arcface_mean=float(np.nanmean(a)), arcface_std=float(np.nanstd(a, ddof=1)), n_nan=int(np.isnan(a).sum()),
                                         dino_mean=float(d.mean()), face_area_mean=float(ar.mean()) if len(ar) else None, n_detected=int(len(ar)),
                                         per_seed_arcface=[round(float(x), 3) for x in a])
        r = out[f'{tag}_step{steps}']
        print(f"{tag:8s} step{steps}: arcface {r['arcface_mean']:+.3f}±{r['arcface_std']:.3f} nan={r['n_nan']} | dino {r['dino_mean']:.3f} | face area {r['face_area_mean'] and round(r['face_area_mean'])} (detected {r['n_detected']}/10)")
out['vram_gb'] = vram_check(); json.dump(out, open(f'{OUT}/results.json', 'w'), indent=2)
