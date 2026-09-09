# Scheme S: dog anchor (pins composition) + IP mask at the location where faces land under that anchor.
# Mask = mean face bbox over the 10 O_dogimg_t999 outputs (offline calibration, no per-sample cost). 1 NFE, 10 seeds.
import sys, glob; sys.path.insert(0, '/workspace/exp')
import torch, json, numpy as np, cv2
from PIL import Image, ImageDraw
from diffusers.image_processor import IPAdapterMaskProcessor
import common
from common import *
common.SEEDS = list(range(42, 52))
pipe = load_pipe()
def encode(pil):
    x = pipe.image_processor.preprocess(pil, height=1024, width=1024).to('cuda', torch.float32)
    pipe.vae.to(torch.float32)
    with torch.no_grad(): z = pipe.vae.encode(x).latent_dist.mean * pipe.vae.config.scaling_factor
    pipe.vae.to(torch.float16); return z
zdog = encode(Image.open('/workspace/pilot/refs/dog.jpg').convert('RGB'))
boxes = [m.face_bbox(cv2.imread(p)) for p in sorted(glob.glob('/workspace/exp/O/confirm10/t999/O_dogimg_t999_10seeds_seed*.png'))]
boxes = np.array([b for b in boxes if b is not None]); mean_box = boxes.mean(0); spread = boxes.std(0)
print("face boxes under dog anchor: n=%d mean=%s std=%s" % (len(boxes), mean_box.round(0).tolist(), spread.round(0).tolist()))
def rect_mask(bbox, dilate):
    x1, y1, x2, y2 = bbox; cx, cy = (x1+x2)/2, (y1+y2)/2; w, h = (x2-x1)*dilate, (y2-y1)*dilate
    im = Image.new('L', (1024, 1024), 0); ImageDraw.Draw(im).rectangle([cx-w/2, cy-h/2, cx+w/2, cy+h/2], fill=255); return im
mp = IPAdapterMaskProcessor(); results = {}
for dil in [1.3, 1.8]:
    mask = rect_mask(mean_box, dil); mask.save(f'/workspace/exp/S/mask_d{dil}.png')
    kw = dict(cross_attention_kwargs={"ip_adapter_masks": [mp.preprocess([mask], height=1024, width=1024)]})
    name = f"S_dog_mask_d{dil}"
    results[name] = run_scheme(name, pipe, lambda s: anchor(pipe, zdog, noise_for(s), 999), f'/workspace/exp/S/{name}',
                               timesteps=[999], extra_kw=kw, archive_check=False, note=f"dog anchor t999 + mask = mean face box of O outputs x{dil}")
json.dump({k: dict(passed=v['passed'], summary=v['summary'], delta=v['delta_vs_baseline']) for k, v in results.items()},
          open('/workspace/exp/S/results.json', 'w'), indent=2)
print("ANY_PASSED:", any(v['passed'] for v in results.values()))
