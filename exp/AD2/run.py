# AD2: angle sweep of low-frequency rotation toward the reference (face adapter, close-up), plus AD3: source = reference
# latent with the non-face region blanked (face layout only). Reports ArcFace / DINO / CLIP P(sunglasses) / P(beach). 10 seeds.
import sys, math, glob, json; sys.path.insert(0, '/workspace/exp')
sys.path.insert(0, '/workspace/exp/AD')
import torch, numpy as np, cv2
from PIL import Image
import common
from common import *
from run import rotate_lowfreq, encode as _enc   # reuse AD's rotation (module import runs nothing: guarded below)
from transformers import CLIPModel, CLIPProcessor
common.SEEDS = list(range(42, 52)); common.REQ['arcface_min'] = 0.144
PROMPT_CU = "a close-up portrait photo of a woman wearing sunglasses, on a beach"
pipe = load_pipe(adapter='face')
z_ref = _enc(pipe, REF)
# face-only source: keep reference latent inside the (dilated) face box, zero elsewhere (latent grid 128 = 1024/8)
bb = m.face_bbox(cv2.imread('/workspace/pilot/refs/person.jpeg')); x1, y1, x2, y2 = (bb / 8).astype(int)
cx, cy, w, h = (x1+x2)/2, (y1+y2)/2, (x2-x1)*1.3, (y2-y1)*1.3
mask = torch.zeros(1, 1, 128, 128, device='cuda'); mask[..., int(cy-h/2):int(cy+h/2), int(cx-w/2):int(cx+w/2)] = 1
z_ref_face = z_ref * mask
print("face-only source: face box (latent px)", [int(cx-w/2), int(cy-h/2), int(cx+w/2), int(cy+h/2)], "| kept fraction", float(mask.mean()))
results = {}
for src_name, src, theta in [('ref', z_ref, 10), ('ref', z_ref, 20), ('ref', z_ref, 30), ('reffaceonly', z_ref_face, 45), ('reffaceonly', z_ref_face, 90)]:
    name = f"AD2_face_{src_name}_th{theta}"
    results[name] = run_scheme(name, pipe, lambda s, src=src, th=theta: rotate_lowfreq(noise_for(s), src, th), f'/workspace/exp/AD2/{name}',
                               prompt=PROMPT_CU, archive_check=False, note=f"low-freq rotation {theta} deg toward {src_name}")
# CLIP attribute adherence per group
cm = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").cuda().eval(); cp = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
def pclip(paths, caps):
    with torch.no_grad(): out = cm(**cp(text=caps, images=[Image.open(x).convert('RGB') for x in paths], return_tensors='pt', padding=True).to('cuda'))
    return out.logits_per_image.softmax(-1)[:, 0].cpu().numpy()
summary = {}
for name, r in results.items():
    paths = [row['path'] for row in r['per_seed'][name]]
    sg = pclip(paths, ["a photo of a woman wearing sunglasses", "a photo of a woman not wearing sunglasses"]); bg = pclip(paths, ["a photo of a woman on a beach", "a photo of a woman in front of a plain wall"])
    s = r['summary'][name]; summary[name] = dict(arcface=s['arcface'], dino=s['dino'], p_sunglasses=float(sg.mean()), p_beach=float(bg.mean()), passed=r['passed'])
    print(f"{name:28s} arcface {s['arcface']['mean']:+.3f}±{s['arcface']['std']:.3f} nan={s['arcface']['n_nan']} | dino {s['dino']['mean']:.3f} | P(sunglasses) {sg.mean():.2f} | P(beach) {bg.mean():.2f} | passed {r['passed']}")
bp = [row['path'] for row in results[name]['per_seed']['baseline']]
print(f"{'baseline (1-step)':28s} arcface {results[name]['summary']['baseline']['arcface']['mean']:+.3f} | dino {results[name]['summary']['baseline']['dino']['mean']:.3f} | P(sunglasses) {pclip(bp, ['a photo of a woman wearing sunglasses','a photo of a woman not wearing sunglasses']).mean():.2f} | P(beach) {pclip(bp, ['a photo of a woman on a beach','a photo of a woman in front of a plain wall']).mean():.2f}")
json.dump(summary, open('/workspace/exp/AD2/results.json', 'w'), indent=2)
