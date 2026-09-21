# Shared library for card G1 (general subjects, DreamBench). No face code is imported here.
#   build_source(ref, subject_mode=...) : source image whose VAE latent gives the rotation target
#   two-band norm-preserving rotation (MB2 default: low 45 deg @ |f|<=0.15, mid 30 deg @ 0.15-0.35)
#   DreamBench metrics: DINO ViT-S/16 (CLS), CLIP-I and CLIP-T with CLIP ViT-B/32 — the standard choices, so numbers are comparable to the literature
import sys, os, json, glob, math; sys.path.insert(0, '/workspace/exp')
import torch, numpy as np
import torch.nn.functional as F
from PIL import Image
from scipy import stats
import importlib.util as _ilu
_s = _ilu.spec_from_file_location('adrun', '/workspace/exp/AD/run.py'); adrun = _ilu.module_from_spec(_s); _s.loader.exec_module(adrun)
rotate_lowfreq, encode, split = adrun.rotate_lowfreq, adrun.encode, adrun.split
from common import noise_for, check_image
DB = '/workspace/data/dreambooth/dataset'; BASE = "stabilityai/stable-diffusion-xl-base-1.0"
LIVE = {'cat', 'cat2', 'dog', 'dog2', 'dog3', 'dog5', 'dog6', 'dog7', 'dog8'}
PROMPTS = ['a {} in the jungle', 'a {} in the snow', 'a {} on the beach', 'a {} on a cobblestone street', 'a {} on top of pink fabric']   # first 5 of both DreamBench lists, unique token dropped (adapter setting)
def load_classes():
    out, on = {}, False
    for L in open(f'{DB}/prompts_and_classes.txt'):
        L = L.strip()
        if L == 'subject_name,class': on = True; continue
        if on and not L: break
        if on: k, c = L.split(','); out[k] = c
    assert len(out) == 30, len(out); return out
def square(im, size=1024):
    w, h = im.size; s = min(w, h); im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)); return im.resize((size, size), Image.LANCZOS)
def load_refs(subject): return [square(Image.open(p).convert('RGB')) for p in sorted(glob.glob(f'{DB}/{subject}/*.jpg'))]
def build_source(ref_img, subject_mode='ref_direct', **kw):
    """source image for the low/mid-frequency rotation target.
       'ref_direct' (G1-a): the reference itself, centre-cropped to 1024x1024 — no alignment, no grayed region, no pasted background.
       'class_box'  (G1-b): subject cut out (U^2-Net) and roughly aligned to the mean box of the 1-step baseline outputs, on a scene background `bg`
       'ellipse'    (G1-c): class-agnostic centre ellipse prior on `bg`
       'face': the face protocol lives in cards/lib.py (make_source)."""
    if subject_mode == 'ref_direct': return square(ref_img)
    if subject_mode == 'class_box':                      # kw: tb (target box, px), bg (PIL 1024x1024). subject cut out by saliency mask, scaled+shifted so its box lands on tb
        ref = square(ref_img); tb, bg = kw['tb'], kw['bg']; mk = subject_mask(ref); rb = mask_box(mk); assert rb is not None, "no salient subject in reference"
        s_ = float((((tb[2]-tb[0]) / (rb[2]-rb[0])) * ((tb[3]-tb[1]) / (rb[3]-rb[1]))) ** 0.5); n = max(8, int(1024 * s_))
        rs = ref.resize((n, n), Image.LANCZOS); al = Image.fromarray((np.clip(mk, 0, 1) * 255).astype(np.uint8)).resize((n, n), Image.BILINEAR)
        off = (int((tb[0]+tb[2]) / 2 - (rb[0]+rb[2]) / 2 * s_), int((tb[1]+tb[3]) / 2 - (rb[1]+rb[3]) / 2 * s_)); im = bg.copy(); im.paste(rs, off, al); return im
    if subject_mode == 'ellipse':                        # kw: bg. class-agnostic prior: gray ellipse in the image centre (half the width/height), nothing from the reference
        from PIL import ImageDraw, ImageFilter
        im = kw['bg'].copy(); m_ = Image.new('L', im.size, 0); ImageDraw.Draw(m_).ellipse([256, 256, 768, 768], fill=255); m_ = m_.filter(ImageFilter.GaussianBlur(12))
        im.paste(Image.new('RGB', im.size, (128, 128, 128)), (0, 0), m_); return im
    if subject_mode == 'face': raise ValueError("use cards/lib.make_source for faces")
    raise ValueError(subject_mode)
def band(z, lo, hi): return split(z, hi)[0] - split(z, lo)[0]
def rotate_two_band(eps, src, th_lo=45, rho_lo=0.15, th_mid=30, rho_hi=0.35):
    e = eps.float(); out = rotate_lowfreq(eps, src, th_lo, rho_lo).float()
    if th_mid != 0:
        eM = band(e, rho_lo, rho_hi); sM = band(src.float(), rho_lo, rho_hi); r = eM.norm(); ehat = eM / r
        sp = sM - (sM * ehat).sum() * ehat; sp = sp / sp.norm().clamp_min(1e-8); t = math.radians(th_mid); out = out - eM + r * (math.cos(t) * ehat + math.sin(t) * sp)
    assert abs(float(out.norm()) - float(e.norm())) / float(e.norm()) < 2e-3; return out.to(eps.dtype)
# ---------------- generator: DMD2 1-step + IP-Adapter Plus (ViT-H), non-face ----------------
def build_dmd2_ipplus(scale=0.6):
    from huggingface_hub import hf_hub_download
    from diffusers import DiffusionPipeline, UNet2DConditionModel, LCMScheduler
    unet = UNet2DConditionModel.from_config(UNet2DConditionModel.load_config(BASE, subfolder="unet")).to('cuda', torch.float16)
    sd = torch.load(hf_hub_download("tianweiy/DMD2", "dmd2_sdxl_1step_unet_fp16.bin"), map_location='cpu'); missing, _ = unet.load_state_dict(sd, strict=False); assert len(missing) == 0
    pipe = DiffusionPipeline.from_pretrained(BASE, unet=unet, dtype=torch.float16, variant="fp16", use_safetensors=True).to('cuda')
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter-plus_sdxl_vit-h.safetensors", image_encoder_folder="models/image_encoder")   # ViT-H encoder
    pipe.set_ip_adapter_scale(scale); pipe.set_progress_bar_config(disable=True); return pipe
def ip_embeds(pipe, ref_img): return pipe.prepare_ip_adapter_image_embeds([ref_img], None, 'cuda', 1, False)
def gen(pipe, emb, latents, seed, prompt, ts=(399,)):
    g = torch.Generator('cuda').manual_seed(seed)
    return pipe(prompt=prompt, timesteps=list(ts), guidance_scale=0, ip_adapter_image_embeds=emb, latents=latents, generator=g).images[0]
# ---------------- DreamBench metrics ----------------
class Metrics:
    def __init__(self):
        from transformers import CLIPModel, CLIPProcessor, ViTModel
        from torchvision import transforms as T
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda().eval(); self.cp = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.dino = ViTModel.from_pretrained("facebook/dino-vits16", add_pooling_layer=False).cuda().eval()
        self.dt = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    @staticmethod
    def _t(o): return o if torch.is_tensor(o) else o.pooler_output          # newer transformers return an output object from get_*_features
    @torch.no_grad()
    def dino_emb(self, ims): x = torch.stack([self.dt(i) for i in ims]).cuda(); return F.normalize(self.dino(pixel_values=x).last_hidden_state[:, 0], dim=-1)
    @torch.no_grad()
    def clip_img(self, ims): return F.normalize(self._t(self.clip.get_image_features(**self.cp(images=ims, return_tensors='pt').to('cuda'))), dim=-1)
    @torch.no_grad()
    def clip_txt(self, texts): return F.normalize(self._t(self.clip.get_text_features(**self.cp(text=texts, return_tensors='pt', padding=True).to('cuda'))), dim=-1)
    def score(self, ims, refs_d, refs_c, txt, used=0):
        """ims: generated PIL list for ONE prompt. refs_*: (n_ref, d) embeddings of all real images; `used` = index of the reference fed to the adapter/source.
           returns per-image DINO / CLIP-I against all refs (standard), against the used ref only, and against the HELD-OUT refs only (copy-proof), plus CLIP-T."""
        d = self.dino_emb(ims) @ refs_d.T; c = self.clip_img(ims); ci = c @ refs_c.T; ct = (c @ txt.T).squeeze(-1); ho = [j for j in range(refs_d.shape[0]) if j != used]
        return dict(dino=d.mean(1).tolist(), dino_used=d[:, used].tolist(), dino_heldout=d[:, ho].mean(1).tolist(), clip_i=ci.mean(1).tolist(), clip_i_heldout=ci[:, ho].mean(1).tolist(), clip_t=ct.tolist())
def paired(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); d = x - y
    out = dict(mean_a=float(x.mean()), mean_b=float(y.mean()), delta=float(d.mean()), delta_se=float(d.std(ddof=1) / np.sqrt(len(d))), positive=int((d > 0).sum()), n=len(d), p_t=float(stats.ttest_rel(x, y).pvalue))
    try: out['p_wilcoxon'] = float(stats.wilcoxon(x, y).pvalue)
    except ValueError: out['p_wilcoxon'] = float('nan')
    return out
# ---------------- subject segmentation (U^2-Net via onnxruntime; no extra python deps) ----------------
_U2 = None
def subject_mask(im):
    """salient-object mask (H,W) in [0,1] at the image's resolution."""
    global _U2
    if _U2 is None:
        import onnxruntime as ort; _U2 = ort.InferenceSession('/workspace/hf_cache/u2net/u2net.onnx', providers=['CPUExecutionProvider'])
    x = np.asarray(im.convert('RGB').resize((320, 320), Image.LANCZOS), dtype=np.float32) / 255.0; x = (x / max(x.max(), 1e-6) - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    y = _U2.run(None, {_U2.get_inputs()[0].name: x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0, 0]; y = (y - y.min()) / max(y.max() - y.min(), 1e-6)
    return np.asarray(Image.fromarray((y * 255).astype(np.uint8)).resize(im.size, Image.BILINEAR), dtype=np.float32) / 255.0
def mask_box(mask, thr=0.5):
    ys, xs = np.where(mask > thr)
    return None if len(xs) < 50 else np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=float)
