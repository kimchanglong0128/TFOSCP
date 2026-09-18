# Shared library for the next-phase cards (B1, B2, T1, ...). Wraps the AE-2 protocol so each card only adds its variable.
import sys, os, json, glob; sys.path.insert(0, '/workspace/exp'); sys.path.insert(0, '/workspace/exp/AE2'); sys.path.insert(0, '/workspace/pilot')
import torch, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from transformers import CLIPModel, CLIPProcessor
from scipy import stats
import importlib.util as _ilu
_s = _ilu.spec_from_file_location('adrun', '/workspace/exp/AD/run.py'); adrun = _ilu.module_from_spec(_s); _s.loader.exec_module(adrun)
rotate_lowfreq, encode, split = adrun.rotate_lowfreq, adrun.encode, adrun.split
from common import REF, noise_for, check_image
from faceid import build_faceid, face_embeds
import idmetrics as m
SEEDS = list(range(42, 62)); SUBJ = 'person'
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
def arc_of(path, remb, pad_recover=False):
    e = m.arc_embed(cv2.imread(path))
    if e is None and pad_recover:
        pad = ImageOps.expand(Image.open(path).convert('RGB'), border=int(0.3*1024), fill=(128,128,128)); e = m.arc_embed(cv2.cvtColor(np.asarray(pad), cv2.COLOR_RGB2BGR))
    return float('nan') if e is None else float(e @ remb)
def metrics(paths, remb, rdino, pad_recover=False):
    a = np.array([arc_of(p, remb, pad_recover) for p in paths]); d = [float(m.dino_embed(Image.open(p).convert('RGB')) @ rdino) for p in paths]
    p3 = clip_probs(paths, C3); pb = clip_probs(paths, CB)[0]
    return dict(arcface=float(np.nanmean(a)), arcface_std=float(np.nanstd(a, ddof=1)), n_nan=int(np.isnan(a).sum()), dino=float(np.mean(d)),
                p_sunglasses=float(p3[0]), p_clear=float(p3[1]), p_none=float(p3[2]), p_beach=float(pb), per_seed=[None if np.isnan(x) else float(x) for x in a])
def ref_box(ref_img):
    return m.face_bbox(cv2.cvtColor(np.asarray(ref_img), cv2.COLOR_RGB2BGR)).astype(float)
def paste_aligned(ref_img, rb, tb, bg=None):
    """scale+translate ref_img so its face box rb lands on target box tb; paste on bg (beach)."""
    s_ = float((((tb[2]-tb[0])/(rb[2]-rb[0])) * ((tb[3]-tb[1])/(rb[3]-rb[1]))) ** 0.5)
    rs = ref_img.resize((max(8, int(ref_img.size[0]*s_)), max(8, int(ref_img.size[1]*s_))), Image.LANCZOS)
    off = (int((tb[0]+tb[2])/2 - (rb[0]+rb[2])/2*s_), int((tb[1]+tb[3])/2 - (rb[1]+rb[3])/2*s_))
    im = (bg or beach_bg).copy(); im.paste(rs, off); return im
def gray_region(im, tb, region='eyes'):
    x1, y1, x2, y2 = tb; h = y2 - y1; w = x2 - x1; d = ImageDraw.Draw(im); g = (128, 128, 128)
    if region == 'eyes': d.rectangle([x1, y1 + 0.28*h, x2, y1 + 0.52*h], fill=g)
    elif region == 'top': d.rectangle([x1 - 0.15*w, y1 - 0.45*h, x2 + 0.15*w, y1 + 0.22*h], fill=g)
    elif region == 'chin': d.rectangle([x1, y1 + 0.68*h, x2, y2 + 0.10*h], fill=g)
    elif region == 'mouth': d.rectangle([x1 + 0.2*w, y1 + 0.62*h, x2 - 0.2*w, y1 + 0.85*h], fill=g)
    return im
def make_source(ref_img, boxes, region='eyes', rb=None):
    tb = np.array(boxes).mean(0); rb = ref_box(ref_img) if rb is None else rb
    return gray_region(paste_aligned(ref_img, rb, tb), tb, region)
def load_ids(with_id1=False):
    ids = {'id1': REF} if with_id1 else {}
    for p in sorted(glob.glob('/workspace/exp/AD9/ids/id*.png')): ids[os.path.basename(p)[:-4]] = Image.open(p).convert('RGB').resize((1024, 1024), Image.LANCZOS)
    return ids
def gen(pipe, emb, latents, seed, ts, prompt=None):
    g = torch.Generator('cuda').manual_seed(seed)
    return pipe(prompt=prompt or PROMPT, timesteps=ts, guidance_scale=0, ip_adapter_image_embeds=emb, latents=latents, generator=g).images[0]
def rep(k, key, r): print(f"{k} {key:14s} arcface {r['arcface']:+.3f}±{r['arcface_std']:.3f} nan={r['n_nan']} dino {r['dino']:.2f} sg {r['p_sunglasses']:.2f} beach {r['p_beach']:.2f}", flush=True)
def paired(R, ids, a, b, key='arcface'):
    x = np.array([R[k][a][key] for k in ids]); y = np.array([R[k][b][key] for k in ids]); d = x - y
    out = dict(mean_a=float(x.mean()), mean_b=float(y.mean()), delta=float(d.mean()), delta_se=float(d.std(ddof=1)/np.sqrt(len(d))), positive=int((d > 0).sum()), n=len(d),
               p_t=float(stats.ttest_rel(x, y).pvalue))
    try: out['p_wilcoxon'] = float(stats.wilcoxon(x, y).pvalue)
    except ValueError: out['p_wilcoxon'] = float('nan')
    return out
def lowfreq_cos(z1, z2, rho=0.15):
    a, _ = split(z1, rho); b, _ = split(z2, rho); return float((a*b).sum() / (a.norm()*b.norm()))
# ---- Hyper-SD 1-step backbone (from AE-3) ----
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, UNet2DConditionModel, LCMScheduler, DDIMScheduler
from transformers import CLIPVisionModelWithProjection
BASE = "stabilityai/stable-diffusion-xl-base-1.0"
def attach_faceid(pipe, scale=0.6):
    pipe.load_ip_adapter("h94/IP-Adapter-FaceID", subfolder=None, weight_name="ip-adapter-faceid-plusv2_sdxl.bin", image_encoder_folder=None); pipe.set_ip_adapter_scale(scale); pipe.set_progress_bar_config(disable=True); return pipe
def build_hyper_1step():
    unet = UNet2DConditionModel.from_config(UNet2DConditionModel.load_config(BASE, subfolder="unet")).to('cuda', torch.float16)
    from safetensors.torch import load_file
    sd = load_file(hf_hub_download("ByteDance/Hyper-SD", "Hyper-SDXL-1step-Unet.safetensors")); missing, _ = unet.load_state_dict(sd, strict=False); assert len(missing) == 0, missing[:3]
    enc = CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter", subfolder="models/image_encoder", torch_dtype=torch.float16)
    pipe = DiffusionPipeline.from_pretrained(BASE, unet=unet, image_encoder=enc, dtype=torch.float16, variant="fp16", use_safetensors=True).to('cuda')
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config); return attach_faceid(pipe)
def build_hyper_lora(steps_file="Hyper-SDXL-1step-lora.safetensors"):
    """base SDXL + Hyper-SD N-step LoRA fused (IP-Adapter attached first so fuse is not undone), LCM scheduler (official 1-step LoRA usage)."""
    enc = CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter", subfolder="models/image_encoder", torch_dtype=torch.float16)
    pipe = DiffusionPipeline.from_pretrained(BASE, image_encoder=enc, dtype=torch.float16, variant="fp16", use_safetensors=True).to('cuda')
    attach_faceid(pipe); pipe.load_lora_weights(hf_hub_download("ByteDance/Hyper-SD", steps_file)); pipe.fuse_lora()
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config); return pipe
