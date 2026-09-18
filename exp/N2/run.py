# Card N2: low-frequency / start-point baseline family, DMD2 1-step (t=399) + FaceID, 30 FFHQ ids x 20 seeds, one row each (NFE noted):
#   replace_g0.5, replace_g1.0 : Colorful-Noise-style band replacement eps_L <- g*z_L (no norm)                       1 NFE
#   blend0.3_raw / blend0.3_norm: eps' = 0.7 eps + 0.3 z_ref, un-normalised / rescaled to ||eps||                         1 NFE
#   patch_transplant           : copy the face-box region of eps from the seed with the best 1-step ArcFace into every other seed (Crystal-Ball-style trigger) 1 NFE (+ oracle choice)
#   sdedit_raw / sdedit_norm   : x = sqrt(abar_399) z_ref + sqrt(1-abar_399) eps, raw / rescaled to ||eps||               1 NFE
#   mask_gate_oracle           : ip_adapter_masks = dilated face box of the same-seed 1-step output (F4)                  1 NFE given the box
#   two_pass                   : pass 1 without identity (ip scale 0) -> box -> pass 2 with mask gate (F3+F4)              2 NFE
#   noise_select               : pick, from a library of 200 pre-sampled noises, the one whose low band is most aligned with z_L (NoiseQuery-style) 1 NFE
#   rot45 (ours) and step1 reused from B1.  AM1 scale-power is not run: on FaceID the IP map has 4 tokens and power == attention temperature, already negative (scheme S).
import sys; sys.path.insert(0, '/workspace/exp/cards')
from lib import *
from common import abar
OUT = '/workspace/exp/N2'; os.makedirs(f'{OUT}/gen', exist_ok=True); T = [399]; RHO = 0.15
ids = load_ids(); keys = list(ids); B1 = json.load(open('/workspace/exp/B1/results.json'))['results']
pipe = build_faceid("dmd2_sdxl_1step_unet_fp16.bin", False, lora=False); A399 = abar(pipe, 399); print("abar_399 =", A399, flush=True)
LIB = [noise_for(10000 + i) for i in range(200)]; LIB_L = [split(e, RHO)[0] for e in LIB]
def box_mask(b, dil=0.15):
    x1, y1, x2, y2 = b; w, h = x2 - x1, y2 - y1; mk = np.zeros((1024, 1024), np.float32); mk[max(0, int(y1 - dil*h)):int(y2 + dil*h), max(0, int(x1 - dil*w)):int(x2 + dil*w)] = 1
    return pipe.mask_processor if False else Image.fromarray((mk * 255).astype(np.uint8))
from diffusers.image_processor import IPAdapterMaskProcessor
mp = IPAdapterMaskProcessor()
def gen_mask(emb, lat, s, mask_pil, scale=0.6):
    masks = mp.preprocess([mask_pil], height=1024, width=1024); g = torch.Generator('cuda').manual_seed(s)
    return pipe(prompt=PROMPT, timesteps=T, guidance_scale=0, ip_adapter_image_embeds=emb, latents=lat, generator=g, cross_attention_kwargs={"ip_adapter_masks": [masks]}).images[0]
R = json.load(open(f'{OUT}/results.json')) if os.path.exists(f'{OUT}/results.json') else {}
for k, rimg in ids.items():
    if k in R and 'noise_select' in R[k]: continue
    remb = m.arc_embed(cv2.cvtColor(np.asarray(rimg), cv2.COLOR_RGB2BGR)); rdino = m.dino_embed(rimg); emb = face_embeds(pipe, rimg); R[k] = {}
    base_paths = [f'/workspace/exp/B1/gen/{k}_step1_s{s}.png' for s in SEEDS]; bboxes = {s: m.face_bbox(cv2.imread(p)) for s, p in zip(SEEDS, base_paths)}
    boxes = [b for b in bboxes.values() if b is not None]
    if len(boxes) < 5: print(k, "few faces, skipped", flush=True); continue
    src = make_source(rimg, boxes); z = encode(pipe, src); zL = split(z, RHO)[0]; tb = np.array(boxes).mean(0)
    per = B1[k]['step1']['per_seed']; best_s = SEEDS[int(np.nanargmax(per))]; eb = noise_for(best_s)
    def transplant(s):
        e = noise_for(s).clone(); x1, y1, x2, y2 = [int(v / 8) for v in tb]; e[:, :, y1:y2, x1:x2] = eb[:, :, y1:y2, x1:x2]; return e
    def sdedit(s, norm):
        e = noise_for(s); x = (A399 ** 0.5) * z.to(e.dtype) + ((1 - A399) ** 0.5) * e; return (x * (e.float().norm() / x.float().norm())).to(e.dtype) if norm else x
    def blend(s, norm):
        e = noise_for(s); x = 0.7 * e.float() + 0.3 * z.float(); return (x * (e.float().norm() / x.norm())).to(e.dtype) if norm else x.to(e.dtype)
    def replace(s, g_):
        e = noise_for(s); return (split(e, RHO)[1] + g_ * zL).to(e.dtype)
    def select(s):
        # seed-dependent library slice of 20 candidates so different seeds pick different noises
        cand = list(range((s - 42) * 9, (s - 42) * 9 + 20))   # 20 candidates per seed, max index 191 < 200; j = max(cand, key=lambda i: float((LIB_L[i] * zL).sum() / (LIB_L[i].norm() * zL.norm()))); return LIB[j]
    groups = {'replace_g0.5': lambda s: replace(s, 0.5), 'replace_g1.0': lambda s: replace(s, 1.0), 'blend0.3_raw': lambda s: blend(s, False), 'blend0.3_norm': lambda s: blend(s, True),
              'patch_transplant': transplant, 'sdedit_raw': lambda s: sdedit(s, False), 'sdedit_norm': lambda s: sdedit(s, True), 'noise_select': select}
    for gname, fn in groups.items():
        paths = []
        for s in SEEDS:
            lat = fn(s); assert torch.isfinite(lat).all(); img = gen(pipe, emb, lat, s, T); q = f'{OUT}/gen/{k}_{gname}_s{s}.png'; img.save(q); check_image(img); paths.append(q)
        R[k][gname] = metrics(paths, remb, rdino, pad_recover=True); R[k][gname]['latent_std'] = float(np.mean([float(fn(s).float().std()) for s in SEEDS[:3]])); rep(k, gname, R[k][gname])
    # mask gate (oracle box from same-seed baseline) and two-pass
    paths, paths2 = [], []
    for s in SEEDS:
        b = bboxes[s]; mk = box_mask(b if b is not None else tb); img = gen_mask(emb, noise_for(s), s, mk); q = f'{OUT}/gen/{k}_mask_gate_oracle_s{s}.png'; img.save(q); paths.append(q)
        pipe.set_ip_adapter_scale(0.0); p1 = gen(pipe, emb, noise_for(s), s, T); pipe.set_ip_adapter_scale(0.6); b1 = m.face_bbox(cv2.cvtColor(np.asarray(p1), cv2.COLOR_RGB2BGR))
        img2 = gen_mask(emb, noise_for(s), s, box_mask(b1 if b1 is not None else tb)); q2 = f'{OUT}/gen/{k}_two_pass_s{s}.png'; img2.save(q2); paths2.append(q2)
    R[k]['mask_gate_oracle'] = metrics(paths, remb, rdino, pad_recover=True); rep(k, 'mask_gate_oracle', R[k]['mask_gate_oracle'])
    R[k]['two_pass'] = metrics(paths2, remb, rdino, pad_recover=True); rep(k, 'two_pass', R[k]['two_pass'])
    R[k]['step1'] = B1[k]['step1']; R[k]['rot45'] = B1[k]['A_ref']
    json.dump(R, open(f'{OUT}/results.json', 'w'), indent=1)
ok = [k for k in keys if 'two_pass' in R.get(k, {})]; S = dict(n=len(ok), rows={})
for g in ('rot45', 'replace_g0.5', 'replace_g1.0', 'blend0.3_raw', 'blend0.3_norm', 'patch_transplant', 'sdedit_raw', 'sdedit_norm', 'noise_select', 'mask_gate_oracle', 'two_pass'):
    p = paired(R, ok, g, 'step1'); p['sg'] = float(np.mean([R[k][g]['p_sunglasses'] for k in ok])); p['beach'] = float(np.mean([R[k][g]['p_beach'] for k in ok])); p['dino'] = float(np.mean([R[k][g]['dino'] for k in ok]))
    p['nan'] = int(sum(R[k][g]['n_nan'] for k in ok)); p['latent_std'] = float(np.mean([R[k][g].get('latent_std', 1.0) for k in ok])); S['rows'][g] = p
    print(f"{g:18s} arcface {p['mean_a']:.3f} Δ{p['delta']:+.3f} p={p['p_t']:.3g} {p['positive']}/{p['n']} | sg {p['sg']:.2f} beach {p['beach']:.2f} dino {p['dino']:.2f} nan {p['nan']} std {p['latent_std']:.2f}", flush=True)
S['step1'] = float(np.mean([R[k]['step1']['arcface'] for k in ok])); json.dump(S, open(f'{OUT}/summary.json', 'w'), indent=1); print("DONE", flush=True)
