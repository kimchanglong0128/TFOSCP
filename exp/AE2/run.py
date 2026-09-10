# AE-2: FaceID-PlusV2 (no LoRA) on DMD2. Phase 1: id1, 20 seeds, {4-step, 1-step, method th30, th45}. Phase 2: the 30 FFHQ ids of AD9,
# {4-step, 1-step, method th45}. Fixed rho=0.15, automatic conflict region (eyes), offline beach background.
import sys, os, json, glob; sys.path.insert(0, '/workspace/exp'); sys.path.insert(0, '/workspace/exp/AD'); sys.path.insert(0, '/workspace/exp/AE2')
import torch, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFilter
from transformers import CLIPModel, CLIPProcessor
import common
from common import REF, noise_for, check_image
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location('adrun', '/workspace/exp/AD/run.py'); adrun = _ilu.module_from_spec(_spec); _spec.loader.exec_module(adrun)
rotate_lowfreq, _enc = adrun.rotate_lowfreq, adrun.encode
from faceid import build_faceid, face_embeds
import idmetrics as m
OUT = '/workspace/exp/AE2'; SEEDS = list(range(42, 62)); SUBJ = 'person'
PROMPT = f"a close-up portrait photo of a {SUBJ} wearing sunglasses, on a beach"
cm = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").cuda().eval(); cp = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
def clip_probs(paths, caps):
    out = []
    for i in range(0, len(paths), 10):
        with torch.no_grad(): o = cm(**cp(text=caps, images=[Image.open(x).convert('RGB') for x in paths[i:i+10]], return_tensors='pt', padding=True).to('cuda'))
        out.append(o.logits_per_image.softmax(-1).cpu().numpy())
    return np.concatenate(out).mean(0)
C3 = [f"a photo of a {SUBJ} wearing dark sunglasses", f"a photo of a {SUBJ} wearing clear eyeglasses", f"a photo of a {SUBJ} without glasses"]
CB = [f"a photo of a {SUBJ} on a beach", f"a photo of a {SUBJ} in front of a plain wall"]
beach_bg = Image.open('/workspace/exp/AD6/bg_beach.png').convert('RGB').filter(ImageFilter.GaussianBlur(6))
def metrics(paths, remb, rdino):
    a, d = [], []
    for p in paths:
        e = m.arc_embed(cv2.imread(p)); a.append(float('nan') if e is None else float(e @ remb)); d.append(float(m.dino_embed(Image.open(p).convert('RGB')) @ rdino))
    a = np.array(a); p3 = clip_probs(paths, C3); pb = clip_probs(paths, CB)[0]
    return dict(arcface=float(np.nanmean(a)), arcface_std=float(np.nanstd(a, ddof=1)), n_nan=int(np.isnan(a).sum()), dino=float(np.mean(d)), p_sunglasses=float(p3[0]), p_clear=float(p3[1]), p_none=float(p3[2]), p_beach=float(pb))
def make_source(ref_img, boxes):
    tb = np.array(boxes).mean(0); rb = m.face_bbox(cv2.cvtColor(np.asarray(ref_img), cv2.COLOR_RGB2BGR)).astype(float)
    s_ = float((((tb[2]-tb[0])/(rb[2]-rb[0])) * ((tb[3]-tb[1])/(rb[3]-rb[1]))) ** 0.5)
    rs = ref_img.resize((max(8, int(ref_img.size[0]*s_)), max(8, int(ref_img.size[1]*s_))), Image.LANCZOS)
    off = (int((tb[0]+tb[2])/2 - (rb[0]+rb[2])/2*s_), int((tb[1]+tb[3])/2 - (rb[1]+rb[3])/2*s_))
    im = beach_bg.copy(); im.paste(rs, off); x1, y1, x2, y2 = tb; h = y2 - y1
    ImageDraw.Draw(im).rectangle([x1, y1 + 0.28*h, x2, y1 + 0.52*h], fill=(128, 128, 128)); return im
def gen(pipe, emb, latents, seed, ts):
    g = torch.Generator('cuda').manual_seed(seed)
    return pipe(prompt=PROMPT, timesteps=ts, guidance_scale=0, ip_adapter_image_embeds=emb, latents=latents, generator=g).images[0]
ids = {'id1': REF}
for p in sorted(glob.glob('/workspace/exp/AD9/ids/id*.png')): ids[os.path.basename(p)[:-4]] = Image.open(p).convert('RGB').resize((1024, 1024), Image.LANCZOS)
results = {k: {} for k in ids}
def rep(k, key, r): print(f"{k} {key:12s} arcface {r['arcface']:+.3f}±{r['arcface_std']:.3f} nan={r['n_nan']} dino {r['dino']:.2f} sg {r['p_sunglasses']:.2f} beach {r['p_beach']:.2f}", flush=True)
pipe4 = build_faceid("dmd2_sdxl_4step_unet_fp16.safetensors", True, lora=False)
for k, rimg in ids.items():
    remb = m.arc_embed(cv2.cvtColor(np.asarray(rimg), cv2.COLOR_RGB2BGR)); rdino = m.dino_embed(rimg); emb = face_embeds(pipe4, rimg); paths = []
    for s in SEEDS:
        img = gen(pipe4, emb, None, s, [999, 749, 499, 249]); q = f'{OUT}/gen/{k}_4step_s{s}.png'; os.makedirs(os.path.dirname(q), exist_ok=True); img.save(q); paths.append(q)
    results[k]['step4'] = metrics(paths, remb, rdino); rep(k, 'step4', results[k]['step4'])
del pipe4; torch.cuda.empty_cache()
pipe1 = build_faceid("dmd2_sdxl_1step_unet_fp16.bin", False, lora=False)
for k, rimg in ids.items():
    remb = m.arc_embed(cv2.cvtColor(np.asarray(rimg), cv2.COLOR_RGB2BGR)); rdino = m.dino_embed(rimg); emb = face_embeds(pipe1, rimg); paths = []
    for s in SEEDS:
        img = gen(pipe1, emb, noise_for(s), s, [399]); q = f'{OUT}/gen/{k}_1step_s{s}.png'; img.save(q); paths.append(q)
    results[k]['step1'] = metrics(paths, remb, rdino); rep(k, 'step1', results[k]['step1'])
    boxes = [b for b in (m.face_bbox(cv2.imread(q)) for q in paths) if b is not None]
    if len(boxes) < 5: results[k]['method'] = None; print(k, "skipped (few faces)", flush=True); continue
    src = make_source(rimg, boxes); src.save(f'{OUT}/gen/{k}_src.png'); z = _enc(pipe1, src)
    for theta in ([30, 45] if k == 'id1' else [45]):
        paths = []
        for s in SEEDS:
            img = gen(pipe1, emb, rotate_lowfreq(noise_for(s), z, theta), s, [399]); q = f'{OUT}/gen/{k}_method{theta}_s{s}.png'; img.save(q); paths.append(q)
        key = 'method' if theta == 45 else f'method{theta}'; results[k][key] = metrics(paths, remb, rdino); rep(k, key, results[k][key])
    json.dump(results, open(f'{OUT}/results.json', 'w'), indent=1)
json.dump(results, open(f'{OUT}/results.json', 'w'), indent=1); print("DONE", flush=True)
