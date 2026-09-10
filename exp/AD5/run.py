# AD5: low-frequency rotation toward a DE-COMPOSED identity source: the reference face aligned to where 1-step faces land,
# on a neutral gray background (optionally only the face region kept). Face adapter, close-up prompt, 10 seeds.
import sys, glob, json; sys.path.insert(0, '/workspace/exp'); sys.path.insert(0, '/workspace/exp/AD')
import torch, numpy as np, cv2
from PIL import Image, ImageDraw
from transformers import CLIPModel, CLIPProcessor
import common
from common import *
from run import rotate_lowfreq, encode as _enc
common.SEEDS = list(range(42, 52)); common.REQ['arcface_min'] = 0.144
PROMPT_CU = "a close-up portrait photo of a woman wearing sunglasses, on a beach"
pipe = load_pipe(adapter='face')
# target face box = mean over baseline close-up 1-step outputs
boxes = np.array([b for b in (m.face_bbox(cv2.imread(p)) for p in sorted(glob.glob('/workspace/exp/W/closeup_step1_seed*.png'))) if b is not None])
tb = boxes.mean(0); rb = m.face_bbox(cv2.imread('/workspace/pilot/refs/person.jpeg')).astype(float)
print("target face box (mean of 1-step outputs):", tb.round(0).tolist(), "| reference face box:", rb.round(0).tolist())
sx, sy = (tb[2]-tb[0])/(rb[2]-rb[0]), (tb[3]-tb[1])/(rb[3]-rb[1]); s = float((sx*sy) ** 0.5)
ref = REF.resize((int(1024*s), int(1024*s)), Image.LANCZOS)
rcx, rcy = (rb[0]+rb[2])/2*s, (rb[1]+rb[3])/2*s; tcx, tcy = (tb[0]+tb[2])/2, (tb[1]+tb[3])/2
canvas = Image.new('RGB', (1024, 1024), (128, 128, 128)); canvas.paste(ref, (int(tcx-rcx), int(tcy-rcy)))
aligned_full = canvas.copy(); aligned_full.save('/workspace/exp/AD5/src_aligned_full.png')
fw, fh = (tb[2]-tb[0])*1.4, (tb[3]-tb[1])*1.4
maskim = Image.new('L', (1024, 1024), 0); ImageDraw.Draw(maskim).ellipse([tcx-fw/2, tcy-fh/2, tcx+fw/2, tcy+fh/2], fill=255)
aligned_face = Image.composite(aligned_full, Image.new('RGB', (1024, 1024), (128, 128, 128)), maskim); aligned_face.save('/workspace/exp/AD5/src_aligned_face.png')
print("aligned sources: scale %.3f | src face detect:" % s, m.face_bbox(cv2.imread('/workspace/exp/AD5/src_aligned_face.png')) is not None)
Z = {'alignedfull': _enc(pipe, aligned_full), 'alignedface': _enc(pipe, aligned_face)}
results = {}
for src_name, z in Z.items():
    for theta in [15, 20, 25]:
        name = f"AD5_{src_name}_th{theta}"
        results[name] = run_scheme(name, pipe, lambda s_, z=z, th=theta: rotate_lowfreq(noise_for(s_), z, th), f'/workspace/exp/AD5/{name}',
                                   prompt=PROMPT_CU, archive_check=False, note=f"low-freq rotation {theta} deg toward {src_name} (face aligned to mean 1-step face box, gray bg)")
cm = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").cuda().eval(); cp = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
def pclip(paths, caps):
    with torch.no_grad(): out = cm(**cp(text=caps, images=[Image.open(x).convert('RGB') for x in paths], return_tensors='pt', padding=True).to('cuda'))
    return out.logits_per_image.softmax(-1)[:, 0].cpu().numpy()
SG = ["a photo of a woman wearing sunglasses", "a photo of a woman not wearing sunglasses"]; BG = ["a photo of a woman on a beach", "a photo of a woman in front of a plain wall"]
summary = {}
for name, r in results.items():
    paths = [row['path'] for row in r['per_seed'][name]]; sg = pclip(paths, SG); bg = pclip(paths, BG); s_ = r['summary'][name]
    summary[name] = dict(arcface=s_['arcface'], dino=s_['dino'], p_sunglasses=float(sg.mean()), p_beach=float(bg.mean()), passed=r['passed'])
    print(f"{name:26s} arcface {s_['arcface']['mean']:+.3f}±{s_['arcface']['std']:.3f} nan={s_['arcface']['n_nan']} | dino {s_['dino']['mean']:.3f} | P(sg) {sg.mean():.2f} P(beach) {bg.mean():.2f} | passed {r['passed']}")
json.dump(summary, open('/workspace/exp/AD5/results.json', 'w'), indent=2)
