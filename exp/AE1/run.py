# AE-1: DMD2 SDXL (a real 1-step model, conditioned at t=399) + plus-face IP-Adapter + close-up prompt. 20 seeds.
# Baselines: 1-step (1-step UNet, t=[399]); 4-step ceiling (4-step UNet, t=[999,749,499,249]). Then AD5 on the 1-step UNet.
import sys, os, json, glob; sys.path.insert(0, '/workspace/exp'); sys.path.insert(0, '/workspace/exp/AD')
import torch, numpy as np, cv2
from PIL import Image
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, UNet2DConditionModel, LCMScheduler
from transformers import CLIPModel, CLIPProcessor
import common
from common import *
from run import rotate_lowfreq, encode as _enc
common.SEEDS = list(range(42, 62))
PROMPT_CU = "a close-up portrait photo of a woman wearing sunglasses, on a beach"
BASE = "stabilityai/stable-diffusion-xl-base-1.0"; OUT = '/workspace/exp/AE1'
def build(unet_file, safetensors):
    unet = UNet2DConditionModel.from_config(UNet2DConditionModel.load_config(BASE, subfolder="unet")).to('cuda', torch.float16)
    path = hf_hub_download("tianweiy/DMD2", unet_file)
    if safetensors:
        from safetensors.torch import load_file; sd = load_file(path)
    else:
        sd = torch.load(path, map_location='cpu')
    missing, unexpected = unet.load_state_dict(sd, strict=False); assert len(missing) == 0, f"missing keys {len(missing)}"
    pipe = DiffusionPipeline.from_pretrained(BASE, unet=unet, dtype=torch.float16, variant="fp16", use_safetensors=True).to('cuda')
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder='sdxl_models', weight_name='ip-adapter-plus-face_sdxl_vit-h.safetensors', image_encoder_folder='models/image_encoder')
    pipe.set_ip_adapter_scale(0.6); pipe.set_progress_bar_config(disable=True); return pipe
cm = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").cuda().eval(); cp = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
def pclip(paths, caps):
    with torch.no_grad(): out = cm(**cp(text=caps, images=[Image.open(x).convert('RGB') for x in paths], return_tensors='pt', padding=True).to('cuda'))
    return out.logits_per_image.softmax(-1)[:, 0].cpu().numpy()
SG = ["a photo of a woman wearing sunglasses", "a photo of a woman not wearing sunglasses"]; BG = ["a photo of a woman on a beach", "a photo of a woman in front of a plain wall"]
def report(tag, paths):
    sc = [score(p) for p in paths]; a = np.array([s[1] for s in sc]); d = np.array([s[0] for s in sc])
    r = dict(arcface=dict(mean=float(np.nanmean(a)), std=float(np.nanstd(a, ddof=1)), n_nan=int(np.isnan(a).sum())), dino=float(d.mean()),
             p_sunglasses=float(pclip(paths, SG).mean()), p_beach=float(pclip(paths, BG).mean()), per_seed_arcface=[round(float(x), 3) for x in a])
    print(f"{tag:26s} arcface {r['arcface']['mean']:+.3f}±{r['arcface']['std']:.3f} nan={r['arcface']['n_nan']} | dino {r['dino']:.3f} | P(sg) {r['p_sunglasses']:.2f} P(beach) {r['p_beach']:.2f}", flush=True)
    return r
summary = {}
# ---- 4-step ceiling ----
pipe4 = build("dmd2_sdxl_4step_unet_fp16.safetensors", True)
paths = []
for s in common.SEEDS:
    g = torch.Generator('cuda').manual_seed(s)
    img = pipe4(prompt=PROMPT_CU, timesteps=[999, 749, 499, 249], guidance_scale=0, ip_adapter_image=REF, generator=g).images[0]
    check_image(img); p = f'{OUT}/dmd2_4step_seed{s}.png'; img.save(p); paths.append(p)
summary['dmd2_4step'] = report('DMD2 4-step (4step UNet)', paths)
del pipe4; torch.cuda.empty_cache()
# ---- 1-step model ----
pipe1 = build("dmd2_sdxl_1step_unet_fp16.bin", False)
paths = []
for s in common.SEEDS:
    g = torch.Generator('cuda').manual_seed(s)
    img = pipe1(prompt=PROMPT_CU, timesteps=[399], guidance_scale=0, ip_adapter_image=REF, latents=noise_for(s), generator=g).images[0]
    check_image(img); p = f'{OUT}/dmd2_1step_seed{s}.png'; img.save(p); paths.append(p)
summary['dmd2_1step'] = report('DMD2 1-step (t=399)', paths)
# ---- AD5 on the 1-step model: aligned source (recompute alignment from DMD2's own 1-step face boxes) ----
boxes = np.array([b for b in (m.face_bbox(cv2.imread(p)) for p in paths) if b is not None]); tb = boxes.mean(0)
rb = m.face_bbox(cv2.imread('/workspace/pilot/refs/person.jpeg')).astype(float)
s_ = float((((tb[2]-tb[0])/(rb[2]-rb[0])) * ((tb[3]-tb[1])/(rb[3]-rb[1]))) ** 0.5)
ref = REF.resize((int(1024*s_), int(1024*s_)), Image.LANCZOS); rcx, rcy = (rb[0]+rb[2])/2*s_, (rb[1]+rb[3])/2*s_; tcx, tcy = (tb[0]+tb[2])/2, (tb[1]+tb[3])/2
canvas = Image.new('RGB', (1024, 1024), (128, 128, 128)); canvas.paste(ref, (int(tcx-rcx), int(tcy-rcy))); canvas.save(f'{OUT}/src_aligned_full.png')
print("DMD2 1-step mean face box", tb.round(0).tolist(), "| scale", round(s_, 3), flush=True)
z = _enc(pipe1, canvas)
for theta in [15, 20, 25, 30, 45]:
    paths = []
    for s in common.SEEDS:
        g = torch.Generator('cuda').manual_seed(s)
        img = pipe1(prompt=PROMPT_CU, timesteps=[399], guidance_scale=0, ip_adapter_image=REF, latents=rotate_lowfreq(noise_for(s), z, theta), generator=g).images[0]
        check_image(img); p = f'{OUT}/dmd2_1step_AD5_th{theta}_seed{s}.png'; img.save(p); paths.append(p)
    summary[f'dmd2_1step_AD5_th{theta}'] = report(f'DMD2 1-step + AD5 θ{theta}', paths)
summary['vram_gb'] = vram_check(); json.dump(summary, open(f'{OUT}/results.json', 'w'), indent=2); print("DONE", flush=True)
