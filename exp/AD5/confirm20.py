# 20-seed confirmation (42-61): AD5 aligned-full source at theta 20/22 vs 1-step and 2-step baselines (archive way). CLIP metrics too.
import sys, glob, json; sys.path.insert(0, '/workspace/exp'); sys.path.insert(0, '/workspace/exp/AD')
import torch, numpy as np, cv2
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import common
from common import *
from run import rotate_lowfreq, encode as _enc
common.SEEDS = list(range(42, 62)); common.REQ['arcface_min'] = 0.144
PROMPT_CU = "a close-up portrait photo of a woman wearing sunglasses, on a beach"
pipe = load_pipe(adapter='face')
z = _enc(pipe, Image.open('/workspace/exp/AD5/src_aligned_full.png').convert('RGB'))
OUT = '/workspace/exp/AD5/confirm20'; os.makedirs(OUT, exist_ok=True)
res = {}
for theta in [20, 22]:
    name = f"AD5_alignedfull_th{theta}_20s"
    res[name] = run_scheme(name, pipe, lambda s, th=theta: rotate_lowfreq(noise_for(s), z, th), f'{OUT}/{name}', prompt=PROMPT_CU, archive_check=False)
# 2-step baseline, archive way, 20 seeds
vals = []
for s in common.SEEDS:
    g = torch.Generator('cuda').manual_seed(s)
    img = pipe(prompt=PROMPT_CU, num_inference_steps=2, guidance_scale=1, ip_adapter_image=REF, generator=g).images[0]
    p = f'{OUT}/step2_seed{s}.png'; img.save(p); vals.append(score(p))
a2 = np.array([v[1] for v in vals]); d2 = np.array([v[0] for v in vals])
cm = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").cuda().eval(); cp = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
def pclip(paths, caps):
    with torch.no_grad(): out = cm(**cp(text=caps, images=[Image.open(x).convert('RGB') for x in paths], return_tensors='pt', padding=True).to('cuda'))
    return out.logits_per_image.softmax(-1)[:, 0].cpu().numpy()
SG = ["a photo of a woman wearing sunglasses", "a photo of a woman not wearing sunglasses"]; BG = ["a photo of a woman on a beach", "a photo of a woman in front of a plain wall"]
summary = {}
for name, r in res.items():
    paths = [row['path'] for row in r['per_seed'][name]]; s_ = r['summary'][name]
    summary[name] = dict(arcface=s_['arcface'], dino=s_['dino'], p_sunglasses=float(pclip(paths, SG).mean()), p_beach=float(pclip(paths, BG).mean()),
                         per_seed_arcface=[round(row['arcface'], 3) for row in r['per_seed'][name]])
bp = [row['path'] for row in res[name]['per_seed']['baseline']]; b = res[name]['summary']['baseline']
summary['baseline_1step_20s'] = dict(arcface=b['arcface'], dino=b['dino'], p_sunglasses=float(pclip(bp, SG).mean()), p_beach=float(pclip(bp, BG).mean()))
p2 = [f'{OUT}/step2_seed{s}.png' for s in common.SEEDS]
summary['baseline_2step_20s'] = dict(arcface=dict(mean=float(np.nanmean(a2)), std=float(np.nanstd(a2, ddof=1)), n_nan=int(np.isnan(a2).sum())), dino=dict(mean=float(d2.mean())),
                                     p_sunglasses=float(pclip(p2, SG).mean()), p_beach=float(pclip(p2, BG).mean()), per_seed_arcface=[round(float(x), 3) for x in a2])
json.dump(summary, open(f'{OUT}/results.json', 'w'), indent=2)
for k, v in summary.items():
    print(f"{k:28s} arcface {v['arcface']['mean']:+.3f}±{v['arcface']['std']:.3f} nan={v['arcface'].get('n_nan',0)} | dino {v['dino']['mean']:.3f} | P(sg) {v['p_sunglasses']:.2f} P(beach) {v['p_beach']:.2f}")
print("DONE")
