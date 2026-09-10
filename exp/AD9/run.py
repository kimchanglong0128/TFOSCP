# AD9: multi-identity evaluation on DMD2 (30 real faces from FFHQ/CelebA-HQ mirrors), fixed hyperparameters
# (theta=45, rho=0.15), automatic conflict-region from the prompt, offline prompt background. 20 seeds per identity.
import sys, os, json, glob, io; sys.path.insert(0, '/workspace/exp'); sys.path.insert(0, '/workspace/exp/AD')
import torch, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFilter
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, UNet2DConditionModel, LCMScheduler
from transformers import CLIPModel, CLIPProcessor
import common
from common import *
from run import rotate_lowfreq, encode as _enc
OUT = '/workspace/exp/AD9'; SEEDS = list(range(42, 62)); N_ID = 30; THETA = 45; BASE = "stabilityai/stable-diffusion-xl-base-1.0"
SUBJ = 'person'; PROMPT = f"a close-up portrait photo of a {SUBJ} wearing sunglasses, on a beach"
# ---- automatic conflict region from the prompt ----
REGION_MAP = {'sunglasses': 'eyes', 'glasses': 'eyes', 'hat': 'top', 'beanie': 'top', 'cap': 'top', 'beard': 'chin', 'mustache': 'mouth', 'lipstick': 'mouth'}
def region_for(prompt):
    for k, v in REGION_MAP.items():
        if k in prompt.lower(): return v
    return None
REGION = region_for(PROMPT); print("conflict region for prompt:", REGION, flush=True)
# ---- identities: pull from HF mirrors until 30 single-face images >= 512px ----
def collect():
    from datasets import load_dataset
    paths = sorted(glob.glob(f'{OUT}/ids/id*.png'))
    if len(paths) >= N_ID: return paths[:N_ID]
    for name, split in [("Ryan-sjtu/ffhq512-caption", "train"), ("PhilSad/celeba-hq-1.5k", "train"), ("mattymchen/celeba-hq", "train")]:
        try:
            ds = load_dataset(name, split=split, streaming=True)
        except Exception as e:
            print("dataset", name, "failed:", str(e)[:120], flush=True); continue
        n = len(paths); rng = np.random.RandomState(0)
        for i, ex in enumerate(ds):
            if n >= N_ID: break
            img = ex.get('image') or ex.get('img') or next((v for v in ex.values() if isinstance(v, Image.Image)), None)
            if img is None: continue
            if rng.rand() > 0.3: continue                              # subsample for variety
            img = img.convert('RGB')
            if min(img.size) < 512: continue
            bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR); faces = m._app.get(bgr)
            if len(faces) != 1 or faces[0].det_score < 0.7: continue
            x1, y1, x2, y2 = faces[0].bbox
            if (x2 - x1) * (y2 - y1) < 0.08 * img.size[0] * img.size[1]: continue
            p = f'{OUT}/ids/id{n:02d}.png'; img.save(p); paths.append(p); n += 1
        print("dataset", name, "-> collected", n, flush=True)
        if n >= N_ID: break
    assert len(paths) >= N_ID, f"only {len(paths)} identities"
    return paths[:N_ID]
ID_PATHS = collect(); print("identities:", len(ID_PATHS), flush=True)
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
def gen(pipe, ref_img, latents, seed, ts):
    g = torch.Generator('cuda').manual_seed(seed)
    return pipe(prompt=PROMPT, timesteps=ts, guidance_scale=0, ip_adapter_image=ref_img, latents=latents, generator=g).images[0]
def make_source(ref_img, boxes, region):
    tb = np.array(boxes).mean(0); rb = m.face_bbox(cv2.cvtColor(np.asarray(ref_img), cv2.COLOR_RGB2BGR)).astype(float)
    s_ = float((((tb[2]-tb[0])/(rb[2]-rb[0])) * ((tb[3]-tb[1])/(rb[3]-rb[1]))) ** 0.5)
    rs = ref_img.resize((max(8, int(ref_img.size[0]*s_)), max(8, int(ref_img.size[1]*s_))), Image.LANCZOS)
    off = (int((tb[0]+tb[2])/2 - (rb[0]+rb[2])/2*s_), int((tb[1]+tb[3])/2 - (rb[1]+rb[3])/2*s_))
    im = beach_bg.copy(); im.paste(rs, off); x1, y1, x2, y2 = tb; h = y2 - y1; w = x2 - x1; dr = ImageDraw.Draw(im)
    if region == 'eyes':  dr.rectangle([x1, y1 + 0.28*h, x2, y1 + 0.52*h], fill=(128, 128, 128))
    if region == 'top':   dr.rectangle([x1 - 0.2*w, 0, x2 + 0.2*w, y1 + 0.30*h], fill=(128, 128, 128))
    if region == 'chin':  dr.rectangle([x1, y1 + 0.70*h, x2, y2 + 0.15*h], fill=(128, 128, 128))
    if region == 'mouth': dr.rectangle([x1 + 0.15*w, y1 + 0.62*h, x2 - 0.15*w, y1 + 0.95*h], fill=(128, 128, 128))
    return im
refs = {p: Image.open(p).convert('RGB').resize((1024, 1024), Image.LANCZOS) for p in ID_PATHS}
results = {os.path.basename(p): {} for p in ID_PATHS}
# ---- phase A: 4-step ceiling ----
pipe4 = build("dmd2_sdxl_4step_unet_fp16.safetensors", True)
for p in ID_PATHS:
    idn = os.path.basename(p)[:-4]; rimg = refs[p]; remb = m.arc_embed(cv2.cvtColor(np.asarray(rimg), cv2.COLOR_RGB2BGR)); rdino = m.dino_embed(rimg)
    paths = []
    for s in SEEDS:
        img = gen(pipe4, rimg, None, s, [999, 749, 499, 249]); q = f'{OUT}/gen/{idn}_4step_s{s}.png'; os.makedirs(os.path.dirname(q), exist_ok=True); img.save(q); paths.append(q)
    results[idn + '.png']['step4'] = metrics(paths, remb, rdino); print(f"{idn} 4-step arcface {results[idn+'.png']['step4']['arcface']:+.3f}", flush=True)
del pipe4; torch.cuda.empty_cache()
# ---- phase B: 1-step baseline, then method ----
pipe1 = build("dmd2_sdxl_1step_unet_fp16.bin", False)
for p in ID_PATHS:
    idn = os.path.basename(p)[:-4]; rimg = refs[p]; remb = m.arc_embed(cv2.cvtColor(np.asarray(rimg), cv2.COLOR_RGB2BGR)); rdino = m.dino_embed(rimg)
    paths = []
    for s in SEEDS:
        img = gen(pipe1, rimg, noise_for(s), s, [399]); q = f'{OUT}/gen/{idn}_1step_s{s}.png'; img.save(q); paths.append(q)
    results[idn + '.png']['step1'] = metrics(paths, remb, rdino)
    boxes = [b for b in (m.face_bbox(cv2.imread(q)) for q in paths) if b is not None]
    if len(boxes) < 5: results[idn + '.png']['method'] = None; print(f"{idn}: too few faces at 1-step ({len(boxes)}), skipped", flush=True); continue
    src = make_source(rimg, boxes, REGION); src.save(f'{OUT}/gen/{idn}_src.png'); z = _enc(pipe1, src); paths = []
    for s in SEEDS:
        img = gen(pipe1, rimg, rotate_lowfreq(noise_for(s), z, THETA), s, [399]); q = f'{OUT}/gen/{idn}_method_s{s}.png'; img.save(q); paths.append(q)
    results[idn + '.png']['method'] = metrics(paths, remb, rdino)
    r = results[idn + '.png']; print(f"{idn} 1-step {r['step1']['arcface']:+.3f} -> method {r['method']['arcface']:+.3f} (4-step {r['step4']['arcface']:+.3f}) | dino {r['method']['dino']:.2f} | sg {r['step1']['p_sunglasses']:.2f}->{r['method']['p_sunglasses']:.2f} | beach {r['step1']['p_beach']:.2f}->{r['method']['p_beach']:.2f}", flush=True)
    json.dump(results, open(f'{OUT}/results.json', 'w'), indent=1)
json.dump(results, open(f'{OUT}/results.json', 'w'), indent=1); print("DONE", flush=True)
