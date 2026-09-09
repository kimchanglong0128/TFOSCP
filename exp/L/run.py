# Scheme L: K (mean-portrait prior anchor) + IP injection masked to the prior's face box. 1 NFE.
import sys, os; sys.path.insert(0, '/workspace/exp')
import torch, json, numpy as np, cv2
from PIL import Image, ImageDraw
from diffusers.image_processor import IPAdapterMaskProcessor
from common import *
pipe = load_pipe()
z_prior = torch.load('/workspace/exp/K/prior.pt').cuda()
bb = m.face_bbox(cv2.imread('/workspace/exp/K/prior.png')); assert bb is not None, "no face in prior"
def rect_mask(bbox, dilate):
    x1, y1, x2, y2 = bbox; cx, cy = (x1+x2)/2, (y1+y2)/2; w, h = (x2-x1)*dilate, (y2-y1)*dilate
    im = Image.new('L', (1024, 1024), 0); ImageDraw.Draw(im).rectangle([cx-w/2, cy-h/2, cx+w/2, cy+h/2], fill=255); return im
mp = IPAdapterMaskProcessor()
results = {}
for t, dil in [(999, 1.3), (999, 1.8), (899, 1.3)]:
    name = f"L_t{t}_d{dil}"
    mask = rect_mask(bb, dil); mask.save(f'/workspace/exp/L/mask_d{dil}.png')
    kw = dict(cross_attention_kwargs={"ip_adapter_masks": [mp.preprocess([mask], height=1024, width=1024)]})
    results[name] = run_scheme(name, pipe, lambda s, t=t: anchor(pipe, z_prior, noise_for(s), t), f'/workspace/exp/L/{name}',
                               timesteps=[t], extra_kw=kw, note=f"prior anchor t={t} + IP mask = prior face bbox x{dil}")
json.dump({k: dict(passed=v['passed'], summary=v['summary'], delta=v['delta_vs_baseline']) for k, v in results.items()},
          open('/workspace/exp/L/results.json', 'w'), indent=2)
print("ANY_PASSED:", any(v['passed'] for v in results.values()))
