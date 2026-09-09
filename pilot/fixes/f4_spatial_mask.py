# F4: spatially gate the IP-Adapter injection with a face-region mask (diffusers ip_adapter_masks).
# Derived from H1 (injection fails to localize). Masks: full (check), oracle face bbox from the baseline
# 1-step image, inverted oracle (negative control), generic centered ellipse (portrait prior).
import sys, os, json; sys.path.insert(0, '/workspace/pilot')
import torch, pandas as pd, numpy as np, cv2
from PIL import Image, ImageDraw
from diffusers import DiffusionPipeline, LCMScheduler
from diffusers.image_processor import IPAdapterMaskProcessor
import idmetrics as m

OUT = '/workspace/outputs/f4'; os.makedirs(OUT, exist_ok=True)
REF = Image.open('/workspace/pilot/refs/person.jpeg').convert('RGB')
PROMPT = "a photo of a woman wearing sunglasses, on a beach"
SEEDS = [42, 43, 44]; H = W = 1024

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                         dtype=torch.float16, variant="fp16", use_safetensors=True).to('cuda')
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
pipe.load_ip_adapter("h94/IP-Adapter", subfolder='sdxl_models', weight_name='ip-adapter_sdxl.bin')
mp = IPAdapterMaskProcessor()

def rect_mask(bbox, dilate=1.3):
    x1, y1, x2, y2 = bbox; cx, cy = (x1+x2)/2, (y1+y2)/2; w, h = (x2-x1)*dilate, (y2-y1)*dilate
    im = Image.new('L', (W, H), 0); ImageDraw.Draw(im).rectangle([cx-w/2, cy-h/2, cx+w/2, cy+h/2], fill=255); return im
def ellipse_mask():
    im = Image.new('L', (W, H), 0); ImageDraw.Draw(im).ellipse([W*0.22, H*0.10, W*0.78, H*0.80], fill=255); return im
def invert(im): return Image.fromarray(255 - np.asarray(im))
def to_tensor(im): return [mp.preprocess([im], height=H, width=W)]           # list of [1,1,H,W]

# oracle bboxes from the baseline 1-step images
bboxes = {s: m.face_bbox(cv2.imread(f'/workspace/outputs/ipa_seeds/ref_woman/woman_step1_seed{s}.png')) for s in SEEDS}
print("oracle bboxes:", {s: (None if b is None else b.tolist()) for s, b in bboxes.items()})
for s, b in bboxes.items():
    rect_mask(b).save(f'{OUT}/mask_oracle_seed{s}.png')
ellipse_mask().save(f'{OUT}/mask_ellipse.png')

rows = []
def run(cfg, seed, mask_im, scale):
    pipe.set_ip_adapter_scale(scale)
    g = torch.Generator('cuda').manual_seed(seed)
    kw = {} if mask_im is None else dict(cross_attention_kwargs={"ip_adapter_masks": to_tensor(mask_im)})
    img = pipe(prompt=PROMPT, num_inference_steps=1, guidance_scale=1, ip_adapter_image=REF, generator=g, **kw).images[0]
    path = f'{OUT}/woman_{cfg}_seed{seed}.png'; img.save(path)
    rows.append(dict(cfg=cfg, seed=seed, scale=scale, path=path))

for s in SEEDS:
    run('noise',            s, None,                              0.6)   # baseline
    run('full_mask',        s, Image.new('L', (W, H), 255),       0.6)   # check: must equal baseline
    run('oracle_face',      s, rect_mask(bboxes[s]),              0.6)
    run('oracle_face_s0.9', s, rect_mask(bboxes[s]),              0.9)
    run('oracle_inv',       s, invert(rect_mask(bboxes[s])),      0.6)   # negative control
    run('ellipse',          s, ellipse_mask(),                    0.6)
pd.DataFrame(rows).to_csv(f'{OUT}/runs.csv', index=False)

df = pd.read_csv(f'{OUT}/runs.csv')
df[['dino', 'arcface']] = [m.score(p, 'woman') for p in df.path]
df['face_area'] = [(lambda b: None if b is None else int((b[2]-b[0])*(b[3]-b[1])))(m.face_bbox(cv2.imread(p))) for p in df.path]
df.to_csv(f'{OUT}/eval_f4.csv', index=False)
for s in SEEDS:
    a = np.asarray(Image.open(f'{OUT}/woman_noise_seed{s}.png')); b = np.asarray(Image.open(f'{OUT}/woman_full_mask_seed{s}.png'))
    print(f"check full_mask == noise seed{s}:", np.array_equal(a, b), "| mean|diff|", np.abs(a.astype(float)-b).mean().round(3))
print(df.groupby('cfg', sort=False)[['dino', 'arcface', 'face_area']].agg(['mean', 'std']).round(3))
