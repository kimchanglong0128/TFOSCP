# F1: identity-orthogonal projection of the text (attribute) branch inside IP-Adapter cross-attention.
# Per pixel: T <- T - beta * mask * (<T,I>/|I|^2) I   where T = text-branch output, I = IP-branch output.
# Derived from theory branch 2 (guidance component parallel to identity cannot be undone in 1 step).
# Controls: beta=0 (must equal baseline), random projection direction, inverted mask.
import sys, os, math; sys.path.insert(0, '/workspace/pilot')
import torch, torch.nn.functional as F, pandas as pd, numpy as np, cv2
from PIL import Image, ImageDraw
from diffusers import DiffusionPipeline, LCMScheduler
from diffusers.models.attention_processor import IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0
import idmetrics as m

OUT = '/workspace/outputs/f1'; os.makedirs(OUT, exist_ok=True)
REF = Image.open('/workspace/pilot/refs/person.jpeg').convert('RGB')
PROMPT = "a photo of a woman wearing sunglasses, on a beach"
SEEDS = [42, 43, 44]; H = W = 1024

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                         dtype=torch.float16, variant="fp16", use_safetensors=True).to('cuda')
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
pipe.load_ip_adapter("h94/IP-Adapter", subfolder='sdxl_models', weight_name='ip-adapter_sdxl.bin')
pipe.set_ip_adapter_scale(0.6)

# classic processors (so batch_to_head_dim is called: 1st = text branch T, 2nd = IP branch I)
procs = {}
for name, p in pipe.unet.attn_processors.items():
    if isinstance(p, IPAdapterAttnProcessor2_0):
        q = IPAdapterAttnProcessor(hidden_size=p.hidden_size, cross_attention_dim=p.cross_attention_dim,
                                   num_tokens=p.num_tokens, scale=p.scale).to(pipe.unet.device, dtype=pipe.unet.dtype)
        q.load_state_dict(p.state_dict()); procs[name] = q
    else:
        procs[name] = p
pipe.unet.set_attn_processor(procs)

CFG = dict(beta=0.0, mask=None, mode='proj')      # mode: 'proj' | 'negonly' | 'random'
RNG = torch.Generator('cuda').manual_seed(0)

def mask_for(hw, mask_1024):
    """downsample a [1,1,1024,1024] float mask to this layer's grid -> [1, hw, 1]"""
    side = int(math.sqrt(hw))
    return F.interpolate(mask_1024, size=(side, side), mode='area').reshape(1, hw, 1)

def hook_attn(attn):
    orig = attn.batch_to_head_dim; calls = []
    def patched(x):
        out = orig(x); calls.append(out)
        if len(calls) == 2:
            T, I = calls[0], out
            if CFG['beta'] > 0:
                Tf, If = T.float(), I.float()
                if CFG['mode'] == 'random':
                    If = torch.randn(If.shape, generator=RNG, device=If.device)
                dot = (Tf * If).sum(-1, keepdim=True)
                if CFG['mode'] == 'negonly':
                    dot = dot.clamp(max=0)                      # remove only the component opposing I
                proj = dot / (If * If).sum(-1, keepdim=True).clamp_min(1e-6) * If
                mk = 1.0 if CFG['mask'] is None else mask_for(T.shape[1], CFG['mask']).to(T.device)
                T.sub_((CFG['beta'] * mk * proj).to(T.dtype))   # in place: the processor still holds T
            calls.clear()
        return out
    attn.batch_to_head_dim = patched

for name, p in pipe.unet.attn_processors.items():
    if isinstance(p, IPAdapterAttnProcessor):
        hook_attn(pipe.unet.get_submodule(name.replace('.processor', '')))

def rect_mask(bbox, dilate=1.3):
    x1, y1, x2, y2 = bbox; cx, cy = (x1+x2)/2, (y1+y2)/2; w, h = (x2-x1)*dilate, (y2-y1)*dilate
    im = Image.new('L', (W, H), 0); ImageDraw.Draw(im).rectangle([cx-w/2, cy-h/2, cx+w/2, cy+h/2], fill=255)
    return torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)[None, None]
bboxes = {s: m.face_bbox(cv2.imread(f'/workspace/outputs/ipa_seeds/ref_woman/woman_step1_seed{s}.png')) for s in SEEDS}

CONFIGS = [  # (name, beta, use_mask, mode)
    ('beta0',            0.0, 'face', 'proj'),     # check: must equal baseline
    ('proj_face_b1',     1.0, 'face', 'proj'),
    ('proj_face_b0.5',   0.5, 'face', 'proj'),
    ('proj_all_b1',      1.0, None,   'proj'),
    ('negonly_face_b1',  1.0, 'face', 'negonly'),
    ('random_face_b1',   1.0, 'face', 'random'),   # control: random direction
    ('proj_inv_b1',      1.0, 'inv',  'proj'),     # control: outside the face
]
rows = []
for s in SEEDS:
    fm = rect_mask(bboxes[s])
    for name, beta, use_mask, mode in CONFIGS:
        CFG.update(beta=beta, mode=mode, mask=None if use_mask is None else (fm if use_mask == 'face' else 1.0 - fm))
        g = torch.Generator('cuda').manual_seed(s)
        img = pipe(prompt=PROMPT, num_inference_steps=1, guidance_scale=1, ip_adapter_image=REF, generator=g).images[0]
        path = f'{OUT}/woman_{name}_seed{s}.png'; img.save(path)
        rows.append(dict(cfg=name, beta=beta, mask=str(use_mask), mode=mode, seed=s, path=path))
pd.DataFrame(rows).to_csv(f'{OUT}/runs.csv', index=False)

df = pd.read_csv(f'{OUT}/runs.csv')
df[['dino', 'arcface']] = [m.score(p, 'woman') for p in df.path]
df.to_csv(f'{OUT}/eval_f1.csv', index=False)
for s in SEEDS:
    a = np.asarray(Image.open(f'{OUT}/woman_beta0_seed{s}.png')).astype(float)
    b = np.asarray(Image.open(f'/workspace/outputs/ipa_seeds/ref_woman/woman_step1_seed{s}.png')).astype(float)
    print(f"check beta0 vs baseline seed{s}: mean|diff| {np.abs(a-b).mean():.3f} (classic-vs-SDPA fp16 noise expected ~0.2)")
print(df.groupby('cfg', sort=False)[['dino', 'arcface']].agg(['mean', 'std']).round(3))
