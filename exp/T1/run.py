# Card T1: controlled t_start test. Same FaceID-PlusV2 adapter, same prompt, 30 FFHQ ids x 20 seeds; 1-step baseline and method (theta=45, rho=0.15) per generator:
#   hyper_lora_999 : base SDXL + Hyper-SDXL-1step LoRA (fused), LCM scheduler, t=[999]        (same distillation family as Hyper UNet-800, differs in t_start + capacity)
#   tcd_lora_999   : base SDXL + TCD-SDXL LoRA, TCDScheduler, 1 step (t=999), eta=0.3
#   lcm_lora_999   : base SDXL + LCM-LoRA, LCM scheduler, t=[999]
#   dmd2_1s_{600,800,999}: DMD2 1-step UNet (trained at 399) forced to t in {600,800,999}   (399 = card B1 step1/A_ref, reused)
#   dmd2_4s_{999,399}    : DMD2 4-step UNet run for a single step at t=999 (its own first step) and at t=399
#   hyper_unet_800 = card B2 step1/th45_rho0.15 (reused). Prediction (lit_4): identity transfer decreases with t_start.
import sys; sys.path.insert(0, '/workspace/exp/cards')
from lib import *
from diffusers import TCDScheduler
OUT = '/workspace/exp/T1'; os.makedirs(f'{OUT}/gen', exist_ok=True); THETA, RHO = 45, 0.15
ids = load_ids(); keys = list(ids)
def build_lora(repo, fname, sched):
    enc = CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter", subfolder="models/image_encoder", torch_dtype=torch.float16)
    pipe = DiffusionPipeline.from_pretrained(BASE, image_encoder=enc, dtype=torch.float16, variant="fp16", use_safetensors=True).to('cuda')
    attach_faceid(pipe); pipe.load_lora_weights(repo, weight_name=fname); pipe.fuse_lora()
    pipe.scheduler = (TCDScheduler if sched == 'tcd' else LCMScheduler).from_config(pipe.scheduler.config); return pipe
def build_dmd2(unet_file, safet): return build_faceid(unet_file, safet, lora=False)
GEN = [  # name, builder, list of (group_name, timesteps, extra_kw)
    ('hyper_lora', lambda: build_lora("ByteDance/Hyper-SD", "Hyper-SDXL-1step-lora.safetensors", 'lcm'), [('hyper_lora_999', [999], {})]),
    ('tcd_lora',   lambda: build_lora("h1t/TCD-SDXL-LoRA", "pytorch_lora_weights.safetensors", 'tcd'), [('tcd_lora_999', [999], dict(eta=0.3))]),
    ('lcm_lora',   lambda: build_lora("latent-consistency/lcm-lora-sdxl", "pytorch_lora_weights.safetensors", 'lcm'), [('lcm_lora_999', [999], {})]),
    ('dmd2_1s',    lambda: build_dmd2("dmd2_sdxl_1step_unet_fp16.bin", False), [(f'dmd2_1s_{t}', [t], {}) for t in (600, 800, 999)]),
    ('dmd2_4s',    lambda: build_dmd2("dmd2_sdxl_4step_unet_fp16.safetensors", True), [('dmd2_4s_999', [999], {}), ('dmd2_4s_399', [399], {})]),
]
def gen_kw(pipe, emb, latents, seed, ts, kw):
    g = torch.Generator('cuda').manual_seed(seed)
    return pipe(prompt=PROMPT, timesteps=ts, guidance_scale=0, ip_adapter_image_embeds=emb, latents=latents, generator=g, **kw).images[0]
R = json.load(open(f'{OUT}/results.json')) if os.path.exists(f'{OUT}/results.json') else {}
for gen_name, builder, groups in GEN:
    if all(all(g in R.get(k, {}) and 'method' in R[k][g] for k in keys) for g, _, _ in groups): print(gen_name, "already done", flush=True); continue
    pipe = builder()
    # smoke: 1 image must not be flat and must contain a face for the first group
    img = gen_kw(pipe, face_embeds(pipe, ids['id00']), noise_for(42), 42, groups[0][1], groups[0][2]); a = np.asarray(img).astype(np.float32)
    print(gen_name, "SMOKE std", round(float(a.std()), 1), "face", m.face_bbox(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)) is not None, flush=True); img.save(f'{OUT}/smoke_{gen_name}.png')
    for k, rimg in ids.items():
        remb = m.arc_embed(cv2.cvtColor(np.asarray(rimg), cv2.COLOR_RGB2BGR)); rdino = m.dino_embed(rimg); emb = face_embeds(pipe, rimg); R.setdefault(k, {})
        for gname, ts, kw in groups:
            if gname in R[k] and 'method' in R[k][gname]: continue
            paths = []
            for s in SEEDS:
                img = gen_kw(pipe, emb, noise_for(s), s, ts, kw); q = f'{OUT}/gen/{k}_{gname}_step1_s{s}.png'; img.save(q); check_image(img); paths.append(q)
            r1 = metrics(paths, remb, rdino, pad_recover=True); rep(k, gname + ' 1s', r1)
            boxes = [b for b in (m.face_bbox(cv2.imread(q)) for q in paths) if b is not None]
            if len(boxes) < 5: R[k][gname] = dict(step1=r1, method=None, note='few faces'); print(k, gname, "few faces -> method skipped", flush=True); continue
            src = make_source(rimg, boxes); z = encode(pipe, src)
            l0 = rotate_lowfreq(noise_for(42), z, 0, RHO); assert float((l0.float() - noise_for(42).float()).norm() / noise_for(42).float().norm()) < 2e-3
            paths = []
            for s in SEEDS:
                img = gen_kw(pipe, emb, rotate_lowfreq(noise_for(s), z, THETA, RHO), s, ts, kw); q = f'{OUT}/gen/{k}_{gname}_method_s{s}.png'; img.save(q); check_image(img); paths.append(q)
            rm_ = metrics(paths, remb, rdino, pad_recover=True); rep(k, gname + ' m', rm_); R[k][gname] = dict(step1=r1, method=rm_)
        json.dump(R, open(f'{OUT}/results.json', 'w'), indent=1)
    del pipe; torch.cuda.empty_cache()
# ---- summary incl. reused groups from B1 (dmd2_1s_399) and B2 (hyper_unet_800) ----
def reuse(path, s1, mk, name):
    d = json.load(open(path))['results']
    for k in keys:
        if k in d and mk in d[k]: R.setdefault(k, {})[name] = dict(step1=d[k][s1], method=d[k][mk])
reuse('/workspace/exp/B1/results.json', 'step1', 'A_ref', 'dmd2_1s_399'); reuse('/workspace/exp/B2/results.json', 'step1', 'th45_rho0.15', 'hyper_unet_800')
S = {}
for g in ['lcm_lora_999', 'tcd_lora_999', 'hyper_lora_999', 'hyper_unet_800', 'dmd2_4s_999', 'dmd2_4s_399', 'dmd2_1s_999', 'dmd2_1s_800', 'dmd2_1s_600', 'dmd2_1s_399']:
    ok = [k for k in keys if g in R.get(k, {}) and R[k][g].get('method')]
    if not ok: continue
    x = np.array([R[k][g]['method']['arcface'] for k in ok]); y = np.array([R[k][g]['step1']['arcface'] for k in ok]); d = x - y
    S[g] = dict(n=len(ok), step1=float(np.nanmean(y)), method=float(np.nanmean(x)), delta=float(np.nanmean(d)), positive=int((d > 0).sum()), p_t=float(stats.ttest_rel(x, y, nan_policy='omit').pvalue),
                nan_step1=int(sum(R[k][g]['step1']['n_nan'] for k in ok)), nan_method=int(sum(R[k][g]['method']['n_nan'] for k in ok)),
                sg_step1=float(np.mean([R[k][g]['step1']['p_sunglasses'] for k in ok])), sg_method=float(np.mean([R[k][g]['method']['p_sunglasses'] for k in ok])))
    print(f"{g:15s} n={S[g]['n']} step1 {S[g]['step1']:.3f} method {S[g]['method']:.3f} Δ{S[g]['delta']:+.3f} p={S[g]['p_t']:.3g} {S[g]['positive']}/{S[g]['n']} nan {S[g]['nan_step1']}/{S[g]['nan_method']}", flush=True)
json.dump(S, open(f'{OUT}/summary.json', 'w'), indent=1); print("DONE", flush=True)
