# Scheme FO1: face-only source. The current source pastes the WHOLE scaled reference (its background, flags, clothes) onto the beach; the grid
# outputs/sex_gap_examples.png shows the reference background leaking into the output (id11 flags). Here only an elliptical, feathered face region
# (box x1.15, feather 8% of box) is pasted; the rest is the beach bg. Groups (DMD2 + FaceID, 30x20): face-only at (45, 0.15), (45, 0.25), and low45@0.15+mid20@0.35.
import sys; sys.path.insert(0, '/workspace/exp/cards')
from lib import *
import math
OUT = '/workspace/exp/FO1'; T = [399]
ids = load_ids(); keys = list(ids); B1 = json.load(open('/workspace/exp/B1/results.json'))['results']; R1 = json.load(open('/workspace/exp/R1/results.json')); MB1 = json.load(open('/workspace/exp/MB1/results.json')); sex = {r['id']: r['sex'] for r in json.load(open('/workspace/exp/I1/rows.json'))}
def face_only_source(rimg, tb, expand=1.15, feather=0.08):
    full = paste_aligned(rimg, ref_box(rimg), tb)                                   # whole reference aligned (as before)
    x1, y1, x2, y2 = tb; cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * expand, (y2 - y1) * expand
    mask = Image.new('L', (1024, 1024), 0); ImageDraw.Draw(mask).ellipse([cx - w/2, cy - h/2, cx + w/2, cy + h/2], fill=255); mask = mask.filter(ImageFilter.GaussianBlur(feather * max(w, h)))
    return gray_region(Image.composite(full, beach_bg, mask), tb, 'eyes')
def band(z, lo, hi): return split(z, hi)[0] - split(z, lo)[0]
def rotate_two_band(eps, src, tl, rl, tm, rh):
    e = eps.float(); out = rotate_lowfreq(eps, src, tl, rl).float(); eM = band(e, rl, rh); sM = band(src.float(), rl, rh); r = eM.norm(); ehat = eM / r
    sp = sM - (sM * ehat).sum() * ehat; sp = sp / sp.norm().clamp_min(1e-8); th = math.radians(tm); out = out - eM + r * (math.cos(th) * ehat + math.sin(th) * sp)
    assert abs(float(out.norm()) - float(e.norm())) / float(e.norm()) < 2e-3; return out.to(eps.dtype)
GROUPS = {'fo_lo45r0.15': lambda e, z: rotate_lowfreq(e, z, 45, 0.15), 'fo_lo45r0.25': lambda e, z: rotate_lowfreq(e, z, 45, 0.25), 'fo_lo45r0.15_mid20r0.35': lambda e, z: rotate_two_band(e, z, 45, 0.15, 20, 0.35)}
pipe = build_faceid("dmd2_sdxl_1step_unet_fp16.bin", False, lora=False)
R = json.load(open(f'{OUT}/results.json')) if os.path.exists(f'{OUT}/results.json') else {}
for k, rimg in ids.items():
    if k in R and 'fo_lo45r0.15_mid20r0.35' in R[k]: continue
    remb = m.arc_embed(cv2.cvtColor(np.asarray(rimg), cv2.COLOR_RGB2BGR)); rdino = m.dino_embed(rimg); emb = face_embeds(pipe, rimg)
    R[k] = {'sex': sex[k], 'step1': B1[k]['step1'], 'full_lo45r0.15': B1[k]['A_ref'], 'full_lo45r0.25': R1[k]['th45_rho0.25'], 'full_lo45r0.15_mid20r0.35': MB1[k]['lo45r0.15_mid20r0.35']}
    boxes = [b for b in (m.face_bbox(cv2.imread(f'/workspace/exp/B1/gen/{k}_step1_s{s}.png')) for s in SEEDS) if b is not None]; tb = np.array(boxes).mean(0)
    src = face_only_source(rimg, tb); src.save(f'{OUT}/gen/{k}_src.png'); z = encode(pipe, src)
    for name, fn in GROUPS.items():
        paths = []
        for s in SEEDS:
            img = gen(pipe, emb, fn(noise_for(s), z), s, T); q = f'{OUT}/gen/{k}_{name}_s{s}.png'; img.save(q); check_image(img); paths.append(q)
        R[k][name] = metrics(paths, remb, rdino, pad_recover=True); rep(k, name, R[k][name]); json.dump(R, open(f'{OUT}/results.json', 'w'), indent=1)
ok = [k for k in keys if 'fo_lo45r0.15_mid20r0.35' in R.get(k, {})]; S = dict(n=len(ok), rows={})
for g in ['full_lo45r0.15', 'fo_lo45r0.15', 'full_lo45r0.25', 'fo_lo45r0.25', 'full_lo45r0.15_mid20r0.35', 'fo_lo45r0.15_mid20r0.35']:
    p = paired(R, ok, g, 'step1'); p['sg'] = paired(R, ok, g, 'step1', 'p_sunglasses')['delta']; p['beach'] = paired(R, ok, g, 'step1', 'p_beach')['delta']; p['dino'] = float(np.mean([R[k][g]['dino'] for k in ok])); p['nan'] = int(sum(R[k][g]['n_nan'] for k in ok))
    for s_, nm in ((1, 'male'), (0, 'female')):
        kk = [k for k in ok if R[k]['sex'] == s_]; d = np.array([R[k][g]['arcface'] - R[k]['step1']['arcface'] for k in kk]); p[nm] = dict(delta=float(d.mean()), n=len(kk), p_t=float(stats.ttest_1samp(d, 0).pvalue))
    S['rows'][g] = p; print(f"{g:26s} arcface {p['mean_a']:.3f} Δ{p['delta']:+.4f} p={p['p_t']:.3g} {p['positive']}/{p['n']} | male Δ{p['male']['delta']:+.4f} female Δ{p['female']['delta']:+.4f} | sg Δ{p['sg']:+.3f} beach Δ{p['beach']:+.3f} dino {p['dino']:.3f} nan {p['nan']}", flush=True)
for a, b in (('fo_lo45r0.15', 'full_lo45r0.15'), ('fo_lo45r0.25', 'full_lo45r0.25'), ('fo_lo45r0.15_mid20r0.35', 'full_lo45r0.15_mid20r0.35')):
    q = paired(R, ok, a, b); qb = paired(R, ok, a, b, 'p_beach'); print(f"{a} vs {b}: arcface Δ{q['delta']:+.4f} p={q['p_t']:.3g} {q['positive']}/{q['n']} | beach Δ{qb['delta']:+.3f} p={qb['p_t']:.3g}", flush=True)
json.dump(S, open(f'{OUT}/summary.json', 'w'), indent=1); print("DONE", flush=True)
