# AE-3 fix: Hyper-SD 4-step ceiling was generated with the LoRA UNMERGED (load_ip_adapter after fuse_lora unmerges it) -> gray images.
# Fix: attach FaceID first, then load+fuse the 4-step LoRA; DDIM trailing, no eta (official usage). Regenerate 4-step only, rescore, re-summarize.
import sys, os, json, shutil; sys.path.insert(0, '/workspace/exp/AE3')
import fix4_defs as d
import torch, numpy as np, cv2
from PIL import Image, ImageOps
from scipy import stats
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, DDIMScheduler
from transformers import CLIPVisionModelWithProjection
m = d.m; OUT = d.OUT; SEEDS = d.SEEDS
def build_4step_fixed():
    enc = CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter", subfolder="models/image_encoder", torch_dtype=torch.float16)
    pipe = DiffusionPipeline.from_pretrained(d.BASE, image_encoder=enc, dtype=torch.float16, variant="fp16", use_safetensors=True).to('cuda')
    d.attach_faceid(pipe)                                   # IP-Adapter FIRST
    pipe.load_lora_weights(hf_hub_download("ByteDance/Hyper-SD", "Hyper-SDXL-4steps-lora.safetensors")); pipe.fuse_lora()
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing"); return pipe
def gen4(pipe, emb, seed):
    g = torch.Generator('cuda').manual_seed(seed)
    return pipe(prompt=d.PROMPT, num_inference_steps=4, guidance_scale=0, ip_adapter_image_embeds=emb, generator=g).images[0]   # no eta
ids = {'id1': d.REF}
import glob
for p in sorted(glob.glob('/workspace/exp/AD9/ids/id*.png')): ids[os.path.basename(p)[:-4]] = Image.open(p).convert('RGB').resize((1024, 1024), Image.LANCZOS)
R = json.load(open(f'{OUT}/results.json'))
if not os.path.exists(f'{OUT}/results_step4_broken.json'): shutil.copy(f'{OUT}/results.json', f'{OUT}/results_step4_broken.json')
pipe4 = build_4step_fixed()
# sanity: image must not be gray mush, and a face must be detected
img = gen4(pipe4, d.face_embeds(pipe4, ids['id1']), 42); arr = np.asarray(img).astype(np.float32)
print("SANITY pixel std", round(float(arr.std()), 1), "face", m.face_bbox(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)) is not None, flush=True)
assert arr.std() > 35, "4-step still gray"; img.save(f'{OUT}/smoke/id1_4step_fixed_s42.png')
for k, rimg in ids.items():
    remb = m.arc_embed(cv2.cvtColor(np.asarray(rimg), cv2.COLOR_RGB2BGR)); rdino = m.dino_embed(rimg); emb = d.face_embeds(pipe4, rimg); paths = []
    for s in SEEDS:
        img = gen4(pipe4, emb, s); q = f'{OUT}/gen/{k}_4step_s{s}.png'; img.save(q); d.check_image(img); paths.append(q)
    R[k]['step4'] = d.metrics(paths, remb, rdino); d.rep(k, 'step4', R[k]['step4'])
    # padding recovery for undetected faces (extreme close-ups), as in AE-2
    rec = []
    for q in paths:
        e = m.arc_embed(cv2.imread(q))
        if e is None:
            pad = ImageOps.expand(Image.open(q).convert('RGB'), border=int(0.3*1024), fill=(128,128,128)); e2 = m.arc_embed(cv2.cvtColor(np.asarray(pad), cv2.COLOR_RGB2BGR))
            if e2 is not None: rec.append(float(e2 @ remb))
    allv = [float(m.arc_embed(cv2.imread(q)) @ remb) for q in paths if m.arc_embed(cv2.imread(q)) is not None] + rec
    R[k]['step4']['arcface_padded'] = float(np.mean(allv)); R[k]['step4']['n_recovered'] = len(rec)
    json.dump(R, open(f'{OUT}/results.json', 'w'), indent=1)
del pipe4; torch.cuda.empty_cache()
ids30 = sorted(k for k in R if k != 'id1' and R[k].get('method'))
a = {c: np.array([R[k][c]['arcface'] for k in ids30]) for c in ('step1', 'method', 'step4')}; a4p = np.array([R[k]['step4']['arcface_padded'] for k in ids30])
dl = a['method'] - a['step1']
S = dict(n=len(ids30), step1=float(a['step1'].mean()), method=float(a['method'].mean()), step4=float(np.nanmean(a['step4'])), step4_padded=float(a4p.mean()),
         delta=float(dl.mean()), positive=int((dl > 0).sum()), p_t=float(stats.ttest_rel(a['method'], a['step1']).pvalue),
         method_vs_step4_padded_p=float(stats.ttest_rel(a['method'], a4p).pvalue), method_ge_step4_padded=int((a['method'] >= a4p).sum()),
         step4_nan_total=int(sum(R[k]['step4']['n_nan'] for k in ids30)), step4_recovered=int(sum(R[k]['step4']['n_recovered'] for k in ids30)))
for key in ('p_sunglasses', 'p_beach', 'dino'):
    v = {c: np.array([R[k][c][key] for k in ids30]) for c in ('step1', 'method', 'step4')}; S[key] = {c: float(v[c].mean()) for c in v}; S[key]['p_method_vs_step1'] = float(stats.ttest_rel(v['method'], v['step1']).pvalue)
S['id1'] = {c: R['id1'][c]['arcface'] for c in ('step1', 'method', 'step4')}
json.dump(S, open(f'{OUT}/summary_fixed.json', 'w'), indent=1); print(json.dumps(S, indent=1)); print("DONE", flush=True)
