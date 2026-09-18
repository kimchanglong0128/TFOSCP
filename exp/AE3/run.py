# AE-3: second 1-step backbone — Hyper-SD (ByteDance). 1-step: Hyper-SDXL-1step UNet, LCMScheduler, t=[800];
# 4-step ceiling: base SDXL + Hyper-SDXL-4steps LoRA (fused), DDIM trailing, eta=1, 4 steps. FaceID-PlusV2 adapter (no LoRA).
# Phase 0 sanity (id1, 5 seeds) -> phase 1 id1 20 seeds -> phase 2 30 FFHQ ids.
import sys, os, json, glob; sys.path.insert(0, '/workspace/exp'); sys.path.insert(0, '/workspace/exp/AE2')
import torch, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFilter
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, UNet2DConditionModel, LCMScheduler, DDIMScheduler
from transformers import CLIPVisionModelWithProjection, CLIPModel, CLIPProcessor
import importlib.util as _ilu
_s = _ilu.spec_from_file_location('adrun', '/workspace/exp/AD/run.py'); adrun = _ilu.module_from_spec(_s); _s.loader.exec_module(adrun); rotate_lowfreq, _enc = adrun.rotate_lowfreq, adrun.encode
from common import REF, noise_for, check_image
from faceid import face_embeds
import idmetrics as m
OUT = '/workspace/exp/AE3'; SEEDS = list(range(42, 62)); SUBJ = 'person'; BASE = "stabilityai/stable-diffusion-xl-base-1.0"
PROMPT = f"a close-up portrait photo of a {SUBJ} wearing sunglasses, on a beach"
def attach_faceid(pipe):
    pipe.load_ip_adapter("h94/IP-Adapter-FaceID", subfolder=None, weight_name="ip-adapter-faceid-plusv2_sdxl.bin", image_encoder_folder=None); pipe.set_ip_adapter_scale(0.6); pipe.set_progress_bar_config(disable=True); return pipe
def build_1step():
    unet = UNet2DConditionModel.from_config(UNet2DConditionModel.load_config(BASE, subfolder="unet")).to('cuda', torch.float16)
    from safetensors.torch import load_file
    sd = load_file(hf_hub_download("ByteDance/Hyper-SD", "Hyper-SDXL-1step-Unet.safetensors")); missing, _ = unet.load_state_dict(sd, strict=False); assert len(missing) == 0, missing[:3]
    enc = CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter", subfolder="models/image_encoder", torch_dtype=torch.float16)
    pipe = DiffusionPipeline.from_pretrained(BASE, unet=unet, image_encoder=enc, dtype=torch.float16, variant="fp16", use_safetensors=True).to('cuda')
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config); return attach_faceid(pipe)
def build_4step():
    enc = CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter", subfolder="models/image_encoder", torch_dtype=torch.float16)
    pipe = DiffusionPipeline.from_pretrained(BASE, image_encoder=enc, dtype=torch.float16, variant="fp16", use_safetensors=True).to('cuda')
    pipe.load_lora_weights(hf_hub_download("ByteDance/Hyper-SD", "Hyper-SDXL-4steps-lora.safetensors")); pipe.fuse_lora()
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing"); return attach_faceid(pipe)
def gen1(pipe, emb, latents, seed):
    g = torch.Generator('cuda').manual_seed(seed); return pipe(prompt=PROMPT, timesteps=[800], guidance_scale=0, ip_adapter_image_embeds=emb, latents=latents, generator=g).images[0]
def gen4(pipe, emb, seed):
    g = torch.Generator('cuda').manual_seed(seed); return pipe(prompt=PROMPT, num_inference_steps=4, guidance_scale=0, eta=1.0, ip_adapter_image_embeds=emb, generator=g).images[0]
cm = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").cuda().eval(); cp = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
def clip_probs(paths, caps):
    out = []
    for i in range(0, len(paths), 10):
        with torch.no_grad(): o = cm(**cp(text=caps, images=[Image.open(x).convert('RGB') for x in paths[i:i+10]], return_tensors='pt', padding=True).to('cuda'))
        out.append(o.logits_per_image.softmax(-1).cpu().numpy())
    return np.concatenate(out).mean(0)
C3 = [f"a photo of a {SUBJ} wearing dark sunglasses", f"a photo of a {SUBJ} wearing clear eyeglasses", f"a photo of a {SUBJ} without glasses"]; CB = [f"a photo of a {SUBJ} on a beach", f"a photo of a {SUBJ} in front of a plain wall"]
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
    im = beach_bg.copy(); im.paste(rs, off); x1, y1, x2, y2 = tb; h = y2 - y1; ImageDraw.Draw(im).rectangle([x1, y1 + 0.28*h, x2, y1 + 0.52*h], fill=(128, 128, 128)); return im
def rep(k, key, r): print(f"{k} {key:8s} arcface {r['arcface']:+.3f}±{r['arcface_std']:.3f} nan={r['n_nan']} dino {r['dino']:.2f} sg {r['p_sunglasses']:.2f} beach {r['p_beach']:.2f}", flush=True)
ids = {'id1': REF}
for p in sorted(glob.glob('/workspace/exp/AD9/ids/id*.png')): ids[os.path.basename(p)[:-4]] = Image.open(p).convert('RGB').resize((1024, 1024), Image.LANCZOS)
results = {k: {} for k in ids}
# ---- phase 0+1+2 for 4-step ----
pipe4 = build_4step()
for k, rimg in ids.items():
    remb = m.arc_embed(cv2.cvtColor(np.asarray(rimg), cv2.COLOR_RGB2BGR)); rdino = m.dino_embed(rimg); emb = face_embeds(pipe4, rimg); paths = []
    for s in SEEDS:
        img = gen4(pipe4, emb, s); q = f'{OUT}/gen/{k}_4step_s{s}.png'; os.makedirs(os.path.dirname(q), exist_ok=True); img.save(q); check_image(img); paths.append(q)
    results[k]['step4'] = metrics(paths, remb, rdino); rep(k, 'step4', results[k]['step4'])
    if k == 'id1': print("SANITY 4-step ok", flush=True)
del pipe4; torch.cuda.empty_cache()
pipe1 = build_1step()
for k, rimg in ids.items():
    remb = m.arc_embed(cv2.cvtColor(np.asarray(rimg), cv2.COLOR_RGB2BGR)); rdino = m.dino_embed(rimg); emb = face_embeds(pipe1, rimg); paths = []
    for s in SEEDS:
        img = gen1(pipe1, emb, noise_for(s), s); q = f'{OUT}/gen/{k}_1step_s{s}.png'; img.save(q); check_image(img); paths.append(q)
    results[k]['step1'] = metrics(paths, remb, rdino); rep(k, 'step1', results[k]['step1'])
    boxes = [b for b in (m.face_bbox(cv2.imread(q)) for q in paths) if b is not None]
    if len(boxes) < 5: results[k]['method'] = None; print(k, "skipped", flush=True); continue
    src = make_source(rimg, boxes); src.save(f'{OUT}/gen/{k}_src.png'); z = _enc(pipe1, src); paths = []
    for s in SEEDS:
        img = gen1(pipe1, emb, rotate_lowfreq(noise_for(s), z, 45), s); q = f'{OUT}/gen/{k}_method_s{s}.png'; img.save(q); paths.append(q)
    results[k]['method'] = metrics(paths, remb, rdino); rep(k, 'method', results[k]['method'])
    json.dump(results, open(f'{OUT}/results.json', 'w'), indent=1)
json.dump(results, open(f'{OUT}/results.json', 'w'), indent=1); print("DONE", flush=True)
