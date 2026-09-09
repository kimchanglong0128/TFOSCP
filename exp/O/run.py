# Scheme O: sharp NON-FACE offline latent as start-point anchor (identity leakage impossible). 1 NFE.
# anchors: (a) VAE latent of refs/dog.jpg (replicates H2 control under the strict harness),
#          (b) text-only 8-step latent of "a photo of a dog on a beach" (seed 200), (c) "a photo of a beach" (seed 201).
import sys, os; sys.path.insert(0, '/workspace/exp')
import torch, json, numpy as np, cv2
from PIL import Image
from common import *
pipe = load_pipe()
def decode(z):
    pipe.vae.to(torch.float32)
    with torch.no_grad(): img = pipe.vae.decode(z.to(torch.float32) / pipe.vae.config.scaling_factor).sample
    pipe.vae.to(torch.float16); return pipe.image_processor.postprocess(img, output_type='pil')[0]
def encode(pil):
    x = pipe.image_processor.preprocess(pil, height=1024, width=1024).to('cuda', torch.float32)
    pipe.vae.to(torch.float32)
    with torch.no_grad(): z = pipe.vae.encode(x).latent_dist.mean * pipe.vae.config.scaling_factor
    pipe.vae.to(torch.float16); return z
anchors = {}
anchors['dogimg'] = encode(Image.open('/workspace/pilot/refs/dog.jpg').convert('RGB'))
pipe.set_ip_adapter_scale(0.0)
for tag, prompt, ps in [('dogtxt', "a photo of a dog on a beach", 200), ('beach', "a photo of a beach", 201)]:
    path = f'/workspace/exp/O/anchor_{tag}.pt'
    if os.path.exists(path): z = torch.load(path).cuda()
    else:
        g = torch.Generator('cuda').manual_seed(ps)
        z = pipe(prompt=prompt, num_inference_steps=8, guidance_scale=1, ip_adapter_image=REF, generator=g, output_type='latent').images.float()
        torch.save(z.cpu(), path)
    anchors[tag] = z
pipe.set_ip_adapter_scale(0.6)
for tag, z in anchors.items():
    assert torch.isfinite(z).all() and tuple(z.shape) == LATENT_SHAPE
    decode(z).save(f'/workspace/exp/O/anchor_{tag}.png')
    print(f"anchor {tag}: face detected? {m.face_bbox(cv2.imread(f'/workspace/exp/O/anchor_{tag}.png')) is not None} | vs ref dino/arcface {m.score(f'/workspace/exp/O/anchor_{tag}.png','woman')}")
assert pipe.scheduler.alphas_cumprod.device.type == 'cpu'
results = {}
for tag, t in [('dogimg', 999), ('dogimg', 959), ('dogtxt', 999), ('beach', 999)]:
    name = f"O_{tag}_t{t}"
    results[name] = run_scheme(name, pipe, lambda s, z=anchors[tag], t=t: anchor(pipe, z, noise_for(s), t), f'/workspace/exp/O/{name}',
                               timesteps=[t], note=f"non-face anchor '{tag}' at t={t}")
json.dump({k: dict(passed=v['passed'], summary=v['summary'], delta=v['delta_vs_baseline']) for k, v in results.items()},
          open('/workspace/exp/O/results.json', 'w'), indent=2)
print("ANY_PASSED:", any(v['passed'] for v in results.values()))
