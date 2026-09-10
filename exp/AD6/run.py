# AD6 (DMD2 1-step, 20 seeds): decompose the AD5 source. (b) eye region -> gray (sunglasses-neutral);
# (c) background -> prompt-consistent beach (text-only DMD2 4-step, no person, computed once); (bc) both.
import sys, os, json; sys.path.insert(0, '/workspace/exp'); sys.path.insert(0, '/workspace/exp/AD')
import torch, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFilter
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, UNet2DConditionModel, LCMScheduler
from transformers import CLIPModel, CLIPProcessor
import common
from common import *
from run import rotate_lowfreq, encode as _enc
common.SEEDS = list(range(42, 62)); OUT = '/workspace/exp/AD6'
PROMPT_CU = "a close-up portrait photo of a woman wearing sunglasses, on a beach"; BASE = "stabilityai/stable-diffusion-xl-base-1.0"
def build(unet_file, safetensors):
    unet = UNet2DConditionModel.from_config(UNet2DConditionModel.load_config(BASE, subfolder="unet")).to('cuda', torch.float16)
    path = hf_hub_download("tianweiy/DMD2", unet_file)
    sd = __import__('safetensors.torch', fromlist=['load_file']).load_file(path) if safetensors else torch.load(path, map_location='cpu')
    missing, _ = unet.load_state_dict(sd, strict=False); assert len(missing) == 0
    pipe = DiffusionPipeline.from_pretrained(BASE, unet=unet, dtype=torch.float16, variant="fp16", use_safetensors=True).to('cuda')
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder='sdxl_models', weight_name='ip-adapter-plus-face_sdxl_vit-h.safetensors', image_encoder_folder='models/image_encoder')
    pipe.set_ip_adapter_scale(0.6); pipe.set_progress_bar_config(disable=True); return pipe
# ---- offline background: text-only DMD2 4-step "beach, no people" (once per prompt) ----
pipe4 = build("dmd2_sdxl_4step_unet_fp16.safetensors", True); pipe4.set_ip_adapter_scale(0.0)
g = torch.Generator('cuda').manual_seed(777)
beach = pipe4(prompt="a photo of an empty beach, sand and sea, no people", timesteps=[999, 749, 499, 249], guidance_scale=0, ip_adapter_image=REF, generator=g).images[0]
beach.save(f'{OUT}/bg_beach.png'); assert m.face_bbox(cv2.cvtColor(np.asarray(beach), cv2.COLOR_RGB2BGR)) is None, "background contains a face"
del pipe4; torch.cuda.empty_cache()
pipe1 = build("dmd2_sdxl_1step_unet_fp16.bin", False)
# ---- aligned reference (same alignment as AE-1) ----
tb = np.array([501.0, 153.0, 821.0, 560.0]); rb = m.face_bbox(cv2.imread('/workspace/pilot/refs/person.jpeg')).astype(float)
s_ = float((((tb[2]-tb[0])/(rb[2]-rb[0])) * ((tb[3]-tb[1])/(rb[3]-rb[1]))) ** 0.5)
ref = REF.resize((int(1024*s_), int(1024*s_)), Image.LANCZOS); off = (int((tb[0]+tb[2])/2 - (rb[0]+rb[2])/2*s_), int((tb[1]+tb[3])/2 - (rb[1]+rb[3])/2*s_))
def compose(bg, eyes_gray):
    im = bg.copy(); im.paste(ref, off)
    if eyes_gray:                                     # eye band: upper-middle third of the face box, scaled reference coords
        fx1, fy1, fx2, fy2 = tb; ey1, ey2 = fy1 + 0.28*(fy2-fy1), fy1 + 0.52*(fy2-fy1)
        ImageDraw.Draw(im).rectangle([fx1, ey1, fx2, ey2], fill=(128, 128, 128))
    return im
gray = Image.new('RGB', (1024, 1024), (128, 128, 128)); beach_bg = beach.filter(ImageFilter.GaussianBlur(6))
SRC = {'a_gray': compose(gray, False), 'b_gray_eyesgray': compose(gray, True), 'c_beach': compose(beach_bg, False), 'bc_beach_eyesgray': compose(beach_bg, True)}
for k, im in SRC.items(): im.save(f'{OUT}/src_{k}.png')
Z = {k: _enc(pipe1, im) for k, im in SRC.items()}
cm = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").cuda().eval(); cp = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
def pclip(paths, caps):
    with torch.no_grad(): out = cm(**cp(text=caps, images=[Image.open(x).convert('RGB') for x in paths], return_tensors='pt', padding=True).to('cuda'))
    return out.logits_per_image.softmax(-1)[:, 0].cpu().numpy()
SG = ["a photo of a woman wearing sunglasses", "a photo of a woman not wearing sunglasses"]; BG = ["a photo of a woman on a beach", "a photo of a woman in front of a plain wall"]
summary = {}
for k in SRC:
    for theta in [30, 45]:
        paths = []
        for s in common.SEEDS:
            g = torch.Generator('cuda').manual_seed(s)
            img = pipe1(prompt=PROMPT_CU, timesteps=[399], guidance_scale=0, ip_adapter_image=REF, latents=rotate_lowfreq(noise_for(s), Z[k], theta), generator=g).images[0]
            check_image(img); p = f'{OUT}/{k}_th{theta}_seed{s}.png'; img.save(p); paths.append(p)
        sc = [score(p) for p in paths]; a = np.array([x[1] for x in sc]); d = np.array([x[0] for x in sc])
        summary[f'{k}_th{theta}'] = dict(arcface=dict(mean=float(np.nanmean(a)), std=float(np.nanstd(a, ddof=1)), n_nan=int(np.isnan(a).sum())), dino=float(d.mean()),
                                        p_sunglasses=float(pclip(paths, SG).mean()), p_beach=float(pclip(paths, BG).mean()))
        r = summary[f'{k}_th{theta}']
        print(f"{k:18s} θ{theta}  arcface {r['arcface']['mean']:+.3f}±{r['arcface']['std']:.3f} nan={r['arcface']['n_nan']} | dino {r['dino']:.3f} | P(sg) {r['p_sunglasses']:.2f} P(beach) {r['p_beach']:.2f}", flush=True)
json.dump(summary, open(f'{OUT}/results.json', 'w'), indent=2); print("DONE", flush=True)
