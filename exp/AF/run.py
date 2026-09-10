# AF: FreeU at 1 step (0 NFE). AG: IP-branch attention temperature (0 NFE). Face adapter, close-up, 10 seeds.
import sys, json; sys.path.insert(0, '/workspace/exp')
import torch, numpy as np
from PIL import Image
from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0
from transformers import CLIPModel, CLIPProcessor
import common
from common import *
common.SEEDS = list(range(42, 52)); common.REQ['arcface_min'] = 0.144
PROMPT_CU = "a close-up portrait photo of a woman wearing sunglasses, on a beach"
pipe = load_pipe(adapter='face'); results = {}
# ---- AF: FreeU ----
for tag, (s1, s2, b1, b2) in [('sdxl_default', (0.9, 0.2, 1.3, 1.4)), ('mild', (0.9, 0.5, 1.1, 1.2))]:
    name = f"AF_freeu_{tag}"
    results[name] = run_scheme(name, pipe, lambda s: noise_for(s), f'/workspace/exp/AF/{name}', prompt=PROMPT_CU, archive_check=False,
                               setup_base=lambda: pipe.disable_freeu(), setup_scheme=lambda a=(s1, s2, b1, b2): pipe.enable_freeu(*a),
                               note=f"FreeU s1={s1} s2={s2} b1={b1} b2={b2}")
pipe.disable_freeu()
# ---- AG: IP attention temperature via scaling to_k_ip output (logits *= tau) ----
TAU = {'v': 1.0}
hooks = []
for p in pipe.unet.attn_processors.values():
    if isinstance(p, IPAdapterAttnProcessor2_0):
        hooks.append(p.to_k_ip[0].register_forward_hook(lambda mod, inp, out: out * TAU['v']))
print("AG hooks:", len(hooks))
for tau in [2.0, 4.0, 0.5]:
    name = f"AG_iptemp_tau{tau}"
    results[name] = run_scheme(name, pipe, lambda s: noise_for(s), f'/workspace/exp/AG/{name}', prompt=PROMPT_CU, archive_check=False,
                               setup_base=lambda: TAU.update(v=1.0), setup_scheme=lambda t=tau: TAU.update(v=t), note=f"IP key scaled by {tau} (attention temperature)")
TAU['v'] = 1.0
# ---- CLIP adherence ----
cm = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").cuda().eval(); cp = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
def pclip(paths, caps):
    with torch.no_grad(): out = cm(**cp(text=caps, images=[Image.open(x).convert('RGB') for x in paths], return_tensors='pt', padding=True).to('cuda'))
    return out.logits_per_image.softmax(-1)[:, 0].cpu().numpy()
SG = ["a photo of a woman wearing sunglasses", "a photo of a woman not wearing sunglasses"]; BG = ["a photo of a woman on a beach", "a photo of a woman in front of a plain wall"]
summary = {}
for name, r in results.items():
    paths = [row['path'] for row in r['per_seed'][name]]; sg = pclip(paths, SG); bg = pclip(paths, BG); s = r['summary'][name]; b = r['summary']['baseline']
    summary[name] = dict(arcface=s['arcface'], dino=s['dino'], p_sunglasses=float(sg.mean()), p_beach=float(bg.mean()), passed=r['passed'], baseline_arcface=b['arcface']['mean'])
    print(f"{name:24s} arcface {s['arcface']['mean']:+.3f}±{s['arcface']['std']:.3f} nan={s['arcface']['n_nan']} | dino {s['dino']['mean']:.3f} | P(sg) {sg.mean():.2f} P(beach) {bg.mean():.2f} | baseline arc {b['arcface']['mean']:+.3f} | Δ {r['delta_vs_baseline']['arcface']:+.3f} | passed {r['passed']}")
json.dump(summary, open('/workspace/exp/AF/results.json', 'w'), indent=2)
