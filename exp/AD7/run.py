# AD7 robustness (DMD2 1-step, 20 seeds): (1) second identity; (2) non-eye attribute "red beanie hat" (top region gray);
# (3) "smiling" (mouth region gray). Each: 1-step baseline, 4-step ceiling, source a (plain aligned), source edited (+beach bg), theta 45.
import sys, os, json, urllib.request; sys.path.insert(0, '/workspace/exp'); sys.path.insert(0, '/workspace/exp/AD')
import torch, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFilter
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, UNet2DConditionModel, LCMScheduler
from transformers import CLIPModel, CLIPProcessor
import common
from common import *
from run import rotate_lowfreq, encode as _enc
SEEDS = list(range(42, 62)); OUT = '/workspace/exp/AD7'; BASE = "stabilityai/stable-diffusion-xl-base-1.0"
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
def pclip(paths, caps):
    with torch.no_grad(): out = cm(**cp(text=caps, images=[Image.open(x).convert('RGB') for x in paths], return_tensors='pt', padding=True).to('cuda'))
    return out.logits_per_image.softmax(-1)[:, 0].cpu().numpy()
BG = ["a photo of a woman on a beach", "a photo of a woman in front of a plain wall"]
# ---- second identity: download with UA, else generate a different woman text-only (DMD2 4-step) ----
ref2_path = f'{OUT}/ref2.jpg'
pipe4 = build("dmd2_sdxl_4step_unet_fp16.safetensors", True)
if not os.path.exists(ref2_path):
    if True:
        print('ref2: generating text-only (download path disabled)', flush=True)
        pipe4.set_ip_adapter_scale(0.0); g = torch.Generator('cuda').manual_seed(4242)
        pipe4(prompt="a close-up studio portrait photo of a 35 year old woman with short dark hair, neutral expression, plain background",
              timesteps=[999, 749, 499, 249], guidance_scale=0, ip_adapter_image=REF, generator=g).images[0].save(ref2_path)
        pipe4.set_ip_adapter_scale(0.6)
ref2 = Image.open(ref2_path).convert('RGB').resize((1024, 1024))
e1 = m.arc_embed(cv2.imread('/workspace/pilot/refs/person.jpeg')); e2 = m.arc_embed(cv2.cvtColor(np.asarray(ref2), cv2.COLOR_RGB2BGR))
assert e2 is not None and float(e1 @ e2) < 0.3, f"ref2 invalid or same identity ({None if e2 is None else float(e1@e2):.2f})"
print("ref2 ok; identity cosine vs ref1 = %.3f" % float(e1 @ e2), flush=True)
beach_bg = Image.open('/workspace/exp/AD6/bg_beach.png').convert('RGB').filter(ImageFilter.GaussianBlur(6))
def score_vs(path, ref_emb):
    bgr = cv2.imread(path); e = m.arc_embed(bgr); d = m.dino_embed(Image.open(path).convert('RGB'))
    return (float('nan') if e is None else float(e @ ref_emb)), d
def report(tag, paths, ref_emb, ref_dino, attr_caps):
    sc = [score_vs(p, ref_emb) for p in paths]; a = np.array([s[0] for s in sc]); d = np.array([float(s[1] @ ref_dino) for s in sc])
    r = dict(arcface=dict(mean=float(np.nanmean(a)), std=float(np.nanstd(a, ddof=1)), n_nan=int(np.isnan(a).sum())), dino=float(d.mean()),
             p_attr=float(pclip(paths, attr_caps).mean()), p_beach=float(pclip(paths, BG).mean()))
    print(f"{tag:34s} arcface {r['arcface']['mean']:+.3f}±{r['arcface']['std']:.3f} nan={r['arcface']['n_nan']} | dino {r['dino']:.3f} | P(attr) {r['p_attr']:.2f} P(beach) {r['p_beach']:.2f}", flush=True)
    return r
def gen(pipe, prompt, ref_img, latents, seed, ts):
    g = torch.Generator('cuda').manual_seed(seed)
    return pipe(prompt=prompt, timesteps=ts, guidance_scale=0, ip_adapter_image=ref_img, latents=latents, generator=g).images[0]
def align(ref_img, boxes):
    tb = np.array(boxes).mean(0); rb = m.face_bbox(cv2.cvtColor(np.asarray(ref_img), cv2.COLOR_RGB2BGR)).astype(float)
    s_ = float((((tb[2]-tb[0])/(rb[2]-rb[0])) * ((tb[3]-tb[1])/(rb[3]-rb[1]))) ** 0.5)
    rs = ref_img.resize((int(1024*s_), int(1024*s_)), Image.LANCZOS); off = (int((tb[0]+tb[2])/2 - (rb[0]+rb[2])/2*s_), int((tb[1]+tb[3])/2 - (rb[1]+rb[3])/2*s_))
    return rs, off, tb
def compose(bg, rs, off, tb, region):
    im = bg.copy(); im.paste(rs, off); x1, y1, x2, y2 = tb; h = y2 - y1
    if region == 'eyes':  ImageDraw.Draw(im).rectangle([x1, y1 + 0.28*h, x2, y1 + 0.52*h], fill=(128, 128, 128))
    if region == 'top':   ImageDraw.Draw(im).rectangle([x1 - 0.2*(x2-x1), 0, x2 + 0.2*(x2-x1), y1 + 0.30*h], fill=(128, 128, 128))
    if region == 'mouth': ImageDraw.Draw(im).rectangle([x1 + 0.15*(x2-x1), y1 + 0.62*h, x2 - 0.15*(x2-x1), y1 + 0.95*h], fill=(128, 128, 128))
    return im
CASES = [  # (case, ref_img, prompt, attr caps, edit region)
    ('id2_sunglasses', ref2, "a close-up portrait photo of a woman wearing sunglasses, on a beach", ["a photo of a woman wearing sunglasses", "a photo of a woman not wearing sunglasses"], 'eyes'),
    ('id1_redhat',     REF,  "a close-up portrait photo of a woman wearing a red beanie hat, on a beach", ["a photo of a woman wearing a red beanie hat", "a photo of a woman without a hat"], 'top'),
    ('id1_smiling',    REF,  "a close-up portrait photo of a woman smiling, on a beach", ["a photo of a smiling woman", "a photo of a woman with a neutral expression"], 'mouth'),
]
summary = {}
for case, rimg, prompt, caps, region in CASES:
    remb = m.arc_embed(cv2.cvtColor(np.asarray(rimg), cv2.COLOR_RGB2BGR)); rdino = m.dino_embed(rimg)
    paths = []
    for s in SEEDS:
        img = gen(pipe4, prompt, rimg, None, s, [999, 749, 499, 249]); p = f'{OUT}/{case}_4step_seed{s}.png'; img.save(p); paths.append(p)
    summary[f'{case}_4step'] = report(f'{case} 4-step', paths, remb, rdino, caps)
del pipe4; torch.cuda.empty_cache()
pipe1 = build("dmd2_sdxl_1step_unet_fp16.bin", False)
for case, rimg, prompt, caps, region in CASES:
    remb = m.arc_embed(cv2.cvtColor(np.asarray(rimg), cv2.COLOR_RGB2BGR)); rdino = m.dino_embed(rimg)
    paths = []
    for s in SEEDS:
        img = gen(pipe1, prompt, rimg, noise_for(s), s, [399]); p = f'{OUT}/{case}_1step_seed{s}.png'; img.save(p); paths.append(p)
    summary[f'{case}_1step'] = report(f'{case} 1-step', paths, remb, rdino, caps)
    boxes = [b for b in (m.face_bbox(cv2.imread(p)) for p in paths) if b is not None]
    rs, off, tb = align(rimg, boxes)
    srcs = {'a_gray': compose(Image.new('RGB', (1024, 1024), (128, 128, 128)), rs, off, tb, None),
            'edit_beach': compose(beach_bg, rs, off, tb, region)}
    for k, im in srcs.items():
        im.save(f'{OUT}/src_{case}_{k}.png'); z = _enc(pipe1, im); paths = []
        for s in SEEDS:
            img = gen(pipe1, prompt, rimg, rotate_lowfreq(noise_for(s), z, 45), s, [399]); p = f'{OUT}/{case}_{k}_th45_seed{s}.png'; img.save(p); paths.append(p)
        summary[f'{case}_{k}_th45'] = report(f'{case} {k} θ45', paths, remb, rdino, caps)
json.dump(summary, open(f'{OUT}/results.json', 'w'), indent=2); print("DONE", flush=True)
