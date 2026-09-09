# F3+F4: realistic 1+1 NFE pipeline. Pass 1 (text-only 1-step) gives a layout latent AND a face bbox;
# pass 2 = anchored init from the layout (t) + IP injection masked to the pass-1 face box.
import sys, os; sys.path.insert(0, '/workspace/pilot')
import torch, pandas as pd, numpy as np, cv2
from PIL import Image, ImageDraw
from diffusers import DiffusionPipeline, LCMScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.image_processor import IPAdapterMaskProcessor
import idmetrics as m
OUT='/workspace/outputs/f34'; os.makedirs(OUT, exist_ok=True)
REF=Image.open('/workspace/pilot/refs/person.jpeg').convert('RGB'); PROMPT="a photo of a woman wearing sunglasses, on a beach"
SEEDS=[42,43,44]; H=W=1024
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", dtype=torch.float16, variant="fp16", use_safetensors=True).to('cuda')
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
pipe.load_ip_adapter("h94/IP-Adapter", subfolder='sdxl_models', weight_name='ip-adapter_sdxl.bin')
mp = IPAdapterMaskProcessor()
def rect_mask(bbox, dilate=1.3):
    x1,y1,x2,y2=bbox; cx,cy=(x1+x2)/2,(y1+y2)/2; w,h=(x2-x1)*dilate,(y2-y1)*dilate
    im=Image.new('L',(W,H),0); ImageDraw.Draw(im).rectangle([cx-w/2,cy-h/2,cx+w/2,cy+h/2],fill=255); return im
def decode(z):
    pipe.vae.to(torch.float32)
    with torch.no_grad(): img = pipe.vae.decode(z.to(torch.float32)/pipe.vae.config.scaling_factor).sample
    pipe.vae.to(torch.float16); return pipe.image_processor.postprocess(img, output_type='pil')[0]
rows=[]
for s in SEEDS:
    # pass 1: text-only layout + its face box
    pipe.set_ip_adapter_scale(0.0); g=torch.Generator('cuda').manual_seed(s)
    zL = pipe(prompt=PROMPT, num_inference_steps=1, guidance_scale=1, ip_adapter_image=REF, generator=g, output_type='latent').images
    lay = decode(zL); lay.save(f'{OUT}/layout_seed{s}.png')
    bb = m.face_bbox(cv2.cvtColor(np.asarray(lay), cv2.COLOR_RGB2BGR))
    print(f"seed{s} layout face bbox:", None if bb is None else bb.tolist())
    if bb is None: continue
    mask = [mp.preprocess([rect_mask(bb)], height=H, width=W)]
    pipe.set_ip_adapter_scale(0.6)
    g=torch.Generator('cuda').manual_seed(s); noise = randn_tensor(zL.shape, generator=g, device='cuda', dtype=torch.float16)
    for tag, t, use_mask in [('f3_t999',999,False), ('f4only_layoutmask',None,True), ('f34_t999',999,True), ('f34_t959',959,True)]:
        g=torch.Generator('cuda').manual_seed(s)
        lat = noise if t is None else pipe.scheduler.add_noise(zL, noise, torch.tensor([t],device='cuda'))
        kw = dict(timesteps=[t]) if t else dict(num_inference_steps=1)
        if use_mask: kw['cross_attention_kwargs']={"ip_adapter_masks": mask}
        img = pipe(prompt=PROMPT, guidance_scale=1, ip_adapter_image=REF, latents=lat, generator=g, **kw).images[0]
        p=f'{OUT}/woman_{tag}_seed{s}.png'; img.save(p); rows.append(dict(cfg=tag, seed=s, path=p))
df=pd.DataFrame(rows); df[['dino','arcface']]=[m.score(p,'woman') for p in df.path]; df.to_csv(f'{OUT}/eval_f34.csv', index=False)
print(df.groupby('cfg', sort=False)[['dino','arcface']].agg(['mean','std']).round(3))
