# Scheme MB1: two-band rotation. Low band (|f|<=rho_lo) rotated toward the source by theta_lo as before; a MID band (rho_lo<|f|<=rho_hi) rotated
# toward the source's mid band by a small theta_mid, each band norm-preserving on its own sphere. Motivation: male identity cues (beard/jaw texture)
# live above rho=0.15 (I1-I3: male gain saturates at +0.04). DMD2 + FaceID, 30 ids x 20 seeds; results also split by sex.
#   groups: (lo 45@0.15, mid 10@0.15-0.35), (lo 45@0.15, mid 20@0.15-0.35), (lo 45@0.25, mid 10@0.25-0.45)
import sys; sys.path.insert(0, '/workspace/exp/cards')
from lib import *
import math
OUT = '/workspace/exp/MB1'; T = [399]
GROUPS = [(45, 0.15, 10, 0.35), (45, 0.15, 20, 0.35), (45, 0.25, 10, 0.45)]
def gname(tl, rl, tm, rh): return f"lo{tl}r{rl:.2f}_mid{tm}r{rh:.2f}"
ids = load_ids(); keys = list(ids); B1 = json.load(open('/workspace/exp/B1/results.json'))['results']; R1 = json.load(open('/workspace/exp/R1/results.json')); sex = {r['id']: r['sex'] for r in json.load(open('/workspace/exp/I1/rows.json'))}
def band(z, lo, hi): return split(z, hi)[0] - split(z, lo)[0]
def rotate_two_band(eps, src, tl, rl, tm, rh):
    e = eps.float(); out = rotate_lowfreq(eps, src, tl, rl).float()                      # low band rotated, rest untouched
    eM = band(e, rl, rh); sM = band(src.float(), rl, rh); r = eM.norm(); ehat = eM / r
    sp = sM - (sM * ehat).sum() * ehat; sp = sp / sp.norm().clamp_min(1e-8); th = math.radians(tm)
    out = out - eM + r * (math.cos(th) * ehat + math.sin(th) * sp)
    assert abs(float(out.norm()) - float(e.norm())) / float(e.norm()) < 2e-3; return out.to(eps.dtype)
pipe = build_faceid("dmd2_sdxl_1step_unet_fp16.bin", False, lora=False)
R = json.load(open(f'{OUT}/results.json')) if os.path.exists(f'{OUT}/results.json') else {}
for k, rimg in ids.items():
    if k in R and gname(*GROUPS[-1]) in R[k]: continue
    remb = m.arc_embed(cv2.cvtColor(np.asarray(rimg), cv2.COLOR_RGB2BGR)); rdino = m.dino_embed(rimg); emb = face_embeds(pipe, rimg)
    R[k] = {'sex': sex[k], 'step1': B1[k]['step1'], 'g_lo45r0.15': B1[k]['A_ref'], 'g_lo45r0.25': R1[k]['th45_rho0.25']}
    z = encode(pipe, Image.open(f'/workspace/exp/B1/gen/{k}_src_A_ref.png').convert('RGB'))
    for tl, rl, tm, rh in GROUPS:
        e0 = noise_for(42); r0 = rotate_two_band(e0, z, 0, rl, 0, rh); assert float((r0.float() - e0.float()).norm() / e0.float().norm()) < 2e-3
        paths = []
        for s in SEEDS:
            img = gen(pipe, emb, rotate_two_band(noise_for(s), z, tl, rl, tm, rh), s, T); q = f'{OUT}/gen/{k}_{gname(tl, rl, tm, rh)}_s{s}.png'; img.save(q); check_image(img); paths.append(q)
        R[k][gname(tl, rl, tm, rh)] = metrics(paths, remb, rdino, pad_recover=True); rep(k, gname(tl, rl, tm, rh), R[k][gname(tl, rl, tm, rh)]); json.dump(R, open(f'{OUT}/results.json', 'w'), indent=1)
ok = [k for k in keys if gname(*GROUPS[-1]) in R.get(k, {})]; S = dict(n=len(ok), rows={})
for g in ['g_lo45r0.15', 'g_lo45r0.25'] + [gname(*g_) for g_ in GROUPS]:
    p = paired(R, ok, g, 'step1'); p['sg'] = paired(R, ok, g, 'step1', 'p_sunglasses')['delta']; p['beach'] = paired(R, ok, g, 'step1', 'p_beach')['delta']; p['dino'] = float(np.mean([R[k][g]['dino'] for k in ok])); p['nan'] = int(sum(R[k][g]['n_nan'] for k in ok))
    for s_, nm in ((1, 'male'), (0, 'female')):
        kk = [k for k in ok if R[k]['sex'] == s_]; d = np.array([R[k][g]['arcface'] - R[k]['step1']['arcface'] for k in kk]); p[nm] = dict(delta=float(d.mean()), positive=int((d > 0).sum()), n=len(kk), p_t=float(stats.ttest_1samp(d, 0).pvalue))
    S['rows'][g] = p; print(f"{g:22s} arcface {p['mean_a']:.3f} Δ{p['delta']:+.4f} p={p['p_t']:.3g} {p['positive']}/{p['n']} | male Δ{p['male']['delta']:+.4f} (p={p['male']['p_t']:.2g}) female Δ{p['female']['delta']:+.4f} | sg Δ{p['sg']:+.3f} beach Δ{p['beach']:+.3f} dino {p['dino']:.3f} nan {p['nan']}", flush=True)
for g_ in GROUPS:
    g = gname(*g_); ref = 'g_lo45r0.15' if g_[1] == 0.15 else 'g_lo45r0.25'; q = paired(R, ok, g, ref); print(f"{g} vs {ref}: Δ{q['delta']:+.4f} p={q['p_t']:.3g} {q['positive']}/{q['n']}", flush=True)
json.dump(S, open(f'{OUT}/summary.json', 'w'), indent=1); print("DONE", flush=True)
