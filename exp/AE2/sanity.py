import sys; sys.path.insert(0, '/workspace/exp'); sys.path.insert(0, '/workspace/exp/AE2')
import torch, numpy as np, cv2
from PIL import Image
from common import REF, noise_for, check_image
from faceid import build_faceid, face_embeds
import idmetrics as m
PROMPT = "a close-up portrait photo of a woman wearing sunglasses, on a beach"
r = m.arc_embed(cv2.imread('/workspace/pilot/refs/person.jpeg'))
for unet_file, st, ts, tag in [("dmd2_sdxl_4step_unet_fp16.safetensors", True, [999, 749, 499, 249], "4-step"), ("dmd2_sdxl_1step_unet_fp16.bin", False, [399], "1-step")]:
    for lora in [True, False]:
        pipe = build_faceid(unet_file, st, lora=lora); emb = face_embeds(pipe, REF); vals = []
        for s in [42, 43, 44, 45, 46]:
            g = torch.Generator('cuda').manual_seed(s)
            img = pipe(prompt=PROMPT, timesteps=ts, guidance_scale=0, ip_adapter_image_embeds=emb, latents=noise_for(s), generator=g).images[0]
            check_image(img); p = f'/workspace/exp/AE2/sanity_{tag}_lora{int(lora)}_s{s}.png'; img.save(p)
            e = m.arc_embed(cv2.imread(p)); vals.append(float('nan') if e is None else float(e @ r))
        v = np.array(vals); print(f"FaceID-PlusV2 {tag} lora={lora}: arcface {np.nanmean(v):+.3f}±{np.nanstd(v):.3f} nan={np.isnan(v).sum()} per-seed {np.round(v,3).tolist()}", flush=True)
        del pipe; torch.cuda.empty_cache()
print("DONE")
