# AD4: (a) finer angle curve for full-reference source (15, 25 deg); (b) spatially TIGHT source: reference latent kept only
# in an inner-face box (0.5x the detected face box, ~17% of frame), zero elsewhere, at 20/30/45 deg. Metrics: ArcFace/DINO/CLIP.
import sys, math, json; sys.path.insert(0, '/workspace/exp'); sys.path.insert(0, '/workspace/exp/AD')
import torch, numpy as np, cv2
from PIL import Image
import common
from common import *
from run import rotate_lowfreq, encode as _enc
from transformers import CLIPModel, CLIPProcessor
common.SEEDS = list(range(42, 52)); common.REQ['arcface_min'] = 0.144
PROMPT_CU = "a close-up portrait photo of a woman wearing sunglasses, on a beach"
pipe = load_pipe(adapter='face'); z_ref = _enc(pipe, REF)
bb = m.face_bbox(cv2.imread('/workspace/pilot/refs/person.jpeg')); x1, y1, x2, y2 = bb / 8
cx, cy, w, h = (x1+x2)/2, (y1+y2)/2, (x2-x1)*0.5, (y2-y1)*0.5
mask = torch.zeros(1, 1, 128, 128, device='cuda'); mask[..., int(cy-h/2):int(cy+h/2), int(cx-w/2):int(cx+w/2)] = 1
z_inner = z_ref * mask
print("inner-face source box (latent px)", [int(cx-w/2), int(cy-h/2), int(cx+w/2), int(cy+h/2)], "| kept fraction", round(float(mask.mean()), 3))
results = {}
for src_name, src, theta in [('ref', z_ref, 15), ('ref', z_ref, 25), ('inner', z_inner, 20), ('inner', z_inner, 30), ('inner', z_inner, 45), ('inner', z_inner, 90)]:
    name = f"AD4_face_{src_name}_th{theta}"
    results[name] = run_scheme(name, pipe, lambda s, src=src, th=theta: rotate_lowfreq(noise_for(s), src, th), f'/workspace/exp/AD4/{name}',
                               prompt=PROMPT_CU, archive_check=False, note=f"low-freq rotation {theta} deg toward {src_name}")
cm = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").cuda().eval(); cp = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
def pclip(paths, caps):
    with torch.no_grad(): out = cm(**cp(text=caps, images=[Image.open(x).convert('RGB') for x in paths], return_tensors='pt', padding=True).to('cuda'))
    return out.logits_per_image.softmax(-1)[:, 0].cpu().numpy()
SG = ["a photo of a woman wearing sunglasses", "a photo of a woman not wearing sunglasses"]; BG = ["a photo of a woman on a beach", "a photo of a woman in front of a plain wall"]
summary = {}
for name, r in results.items():
    paths = [row['path'] for row in r['per_seed'][name]]; sg = pclip(paths, SG); bg = pclip(paths, BG); s = r['summary'][name]
    summary[name] = dict(arcface=s['arcface'], dino=s['dino'], p_sunglasses=float(sg.mean()), p_beach=float(bg.mean()), passed=r['passed'])
    print(f"{name:24s} arcface {s['arcface']['mean']:+.3f}±{s['arcface']['std']:.3f} nan={s['arcface']['n_nan']} | dino {s['dino']['mean']:.3f} | P(sunglasses) {sg.mean():.2f} | P(beach) {bg.mean():.2f}")
json.dump(summary, open('/workspace/exp/AD4/results.json', 'w'), indent=2)
