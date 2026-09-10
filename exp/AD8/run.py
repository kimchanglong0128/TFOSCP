# AD8: real-photo identity (ref3, child) on DMD2 1-step, 20 seeds; 3-way CLIP attribute metric; re-score AD6/AD7 with it.
import sys, os, json, glob; sys.path.insert(0, '/workspace/exp'); sys.path.insert(0, '/workspace/exp/AD')
import torch, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFilter
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, UNet2DConditionModel, LCMScheduler
from transformers import CLIPModel, CLIPProcessor
import common
from common import *
from run import rotate_lowfreq, encode as _enc
SEEDS = list(range(42, 62)); OUT = '/workspace/exp/AD8'; BASE = "stabilityai/stable-diffusion-xl-base-1.0"
def build(unet_file, safetensors):
    unet = UNet2DConditionModel.from_config(UNet2DConditionModel.load_config(BASE, subfolder="unet")).to('cuda', torch.float16)
    path = hf_hub_download("tianweiy/DMD2", unet_file)
    sd = __import__('safetensors.torch', fromlist=['load_file']).load_file(path) if safetensors else torch.load(path, map_location='cpu')
    missing, _ = unet.load_state_dict(sd, strict=False); assert len(missing) == 0
    pipe = DiffusionPipeline.from_pretrained(BASE, unet=unet, dtype=torch.float16, variant="fp16", use_safetensors=True).to('cuda')
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder='sdxl_models', weight_name='ip-adapter-plus-face_sdxl_vit-h.safetensors', image_encoder_folder='models/image_encoder')
    pipe.set_ip_adapter_scale(0.6); pipe.set_progress_bar_config(disable=True); return pipe
cm = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").cuda().eval(); cp = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
def pclip3(paths, subj):
    caps = [f"a photo of a {subj} wearing dark sunglasses", f"a photo of a {subj} wearing clear eyeglasses", f"a photo of a {subj} without glasses"]
    with torch.no_grad(): out = cm(**cp(text=caps, images=[Image.open(x).convert('RGB') for x in paths], return_tensors='pt', padding=True).to('cuda'))
    pr = out.logits_per_image.softmax(-1).cpu().numpy(); return pr.mean(0)          # [P(sunglasses), P(clear), P(none)]
def pbeach(paths, subj):
    caps = [f"a photo of a {subj} on a beach", f"a photo of a {subj} in front of a plain wall"]
    with torch.no_grad(): out = cm(**cp(text=caps, images=[Image.open(x).convert('RGB') for x in paths], return_tensors='pt', padding=True).to('cuda'))
    return float(out.logits_per_image.softmax(-1)[:, 0].mean())
# ---- re-score earlier groups with the 3-way metric ----
print("== 3-way attribute re-score [sunglasses, clear, none] ==", flush=True)
for tag, pat, subj in [('id1 AD6 a_gray th45', '/workspace/exp/AD6/a_gray_th45_seed*.png', 'woman'), ('id1 AD6 b_eyesgray th45', '/workspace/exp/AD6/b_gray_eyesgray_th45_seed*.png', 'woman'),
                       ('id1 AD6 bc th45', '/workspace/exp/AD6/bc_beach_eyesgray_th45_seed*.png', 'woman'), ('id1 DMD2 4-step', '/workspace/exp/AE1/dmd2_4step_seed*.png', 'woman'),
                       ('id1 DMD2 1-step', '/workspace/exp/AE1/dmd2_1step_seed*.png', 'woman'), ('id2 4-step', '/workspace/exp/AD7/id2_sunglasses_4step_seed*.png', 'woman'),
                       ('id2 1-step', '/workspace/exp/AD7/id2_sunglasses_1step_seed*.png', 'woman'), ('id2 edit_beach th45', '/workspace/exp/AD7/id2_sunglasses_edit_beach_th45_seed*.png', 'woman')]:
    pr = pclip3(sorted(glob.glob(pat)), subj); print(f"{tag:26s} {pr.round(2).tolist()}", flush=True)
# ---- identity 3 ----
REF3 = Image.open(f'{OUT}/../AD7/ref3.png').convert('RGB').resize((1024, 1024), Image.LANCZOS)
r3 = m.arc_embed(cv2.cvtColor(np.asarray(REF3), cv2.COLOR_RGB2BGR)); r3d = m.dino_embed(REF3); SUBJ = 'young boy'
PROMPT = f"a close-up portrait photo of a {SUBJ} wearing sunglasses, on a beach"
beach_bg = Image.open('/workspace/exp/AD6/bg_beach.png').convert('RGB').filter(ImageFilter.GaussianBlur(6))
def report(tag, paths):
    sc = []; ds = []
    for p in paths:
        e = m.arc_embed(cv2.imread(p)); sc.append(float('nan') if e is None else float(e @ r3)); ds.append(float(m.dino_embed(Image.open(p).convert('RGB')) @ r3d))
    a = np.array(sc); pr = pclip3(paths, SUBJ)
    r = dict(arcface=dict(mean=float(np.nanmean(a)), std=float(np.nanstd(a, ddof=1)), n_nan=int(np.isnan(a).sum())), dino=float(np.mean(ds)), p3=pr.round(3).tolist(), p_beach=pbeach(paths, SUBJ))
    print(f"{tag:28s} arcface {r['arcface']['mean']:+.3f}±{r['arcface']['std']:.3f} nan={r['arcface']['n_nan']} | dino {r['dino']:.3f} | [sg,clear,none] {pr.round(2).tolist()} | P(beach) {r['p_beach']:.2f}", flush=True)
    return r
def gen(pipe, latents, seed, ts):
    g = torch.Generator('cuda').manual_seed(seed)
    return pipe(prompt=PROMPT, timesteps=ts, guidance_scale=0, ip_adapter_image=REF3, latents=latents, generator=g).images[0]
summary = {}
pipe4 = build("dmd2_sdxl_4step_unet_fp16.safetensors", True); paths = []
for s in SEEDS:
    img = gen(pipe4, None, s, [999, 749, 499, 249]); p = f'{OUT}/id3_4step_seed{s}.png'; img.save(p); paths.append(p)
summary['id3_4step'] = report('id3 4-step', paths); del pipe4; torch.cuda.empty_cache()
pipe1 = build("dmd2_sdxl_1step_unet_fp16.bin", False); paths = []
for s in SEEDS:
    img = gen(pipe1, noise_for(s), s, [399]); p = f'{OUT}/id3_1step_seed{s}.png'; img.save(p); paths.append(p)
summary['id3_1step'] = report('id3 1-step', paths)
boxes = [b for b in (m.face_bbox(cv2.imread(p)) for p in paths) if b is not None]; tb = np.array(boxes).mean(0)
rb = m.face_bbox(cv2.cvtColor(np.asarray(REF3), cv2.COLOR_RGB2BGR)).astype(float)
s_ = float((((tb[2]-tb[0])/(rb[2]-rb[0])) * ((tb[3]-tb[1])/(rb[3]-rb[1]))) ** 0.5)
rs = REF3.resize((int(1024*s_), int(1024*s_)), Image.LANCZOS); off = (int((tb[0]+tb[2])/2 - (rb[0]+rb[2])/2*s_), int((tb[1]+tb[3])/2 - (rb[1]+rb[3])/2*s_))
print("id3 target box", tb.round(0).tolist(), "scale %.3f" % s_, "detected %d/20" % len(boxes), flush=True)
def compose(bg, eyes):
    im = bg.copy(); im.paste(rs, off)
    if eyes: x1, y1, x2, y2 = tb; h = y2 - y1; ImageDraw.Draw(im).rectangle([x1, y1 + 0.28*h, x2, y1 + 0.52*h], fill=(128, 128, 128))
    return im
SRC = {'a_gray': compose(Image.new('RGB', (1024, 1024), (128, 128, 128)), False), 'b_eyesgray': compose(Image.new('RGB', (1024, 1024), (128, 128, 128)), True), 'bc_beach_eyesgray': compose(beach_bg, True)}
for k, im in SRC.items():
    im.save(f'{OUT}/src_{k}.png'); z = _enc(pipe1, im)
    for theta in [30, 45]:
        paths = []
        for s in SEEDS:
            img = gen(pipe1, rotate_lowfreq(noise_for(s), z, theta), s, [399]); p = f'{OUT}/id3_{k}_th{theta}_seed{s}.png'; img.save(p); paths.append(p)
        summary[f'id3_{k}_th{theta}'] = report(f'id3 {k} θ{theta}', paths)
json.dump(summary, open(f'{OUT}/results.json', 'w'), indent=2); print("DONE", flush=True)
