# Card R1 (follow-up to N1): rho sweep on DMD2 + FaceID with the reference-face source, full protocol (30 ids x 20 seeds).
#   rho in {0.20, 0.25, 0.30} at theta=45, plus rho=0.25 at theta=30. step1 and rho=0.15 (A_ref) reused from B1. Sources = B1's A_ref sources (identical construction).
import sys; sys.path.insert(0, '/workspace/exp/cards')
from lib import *
OUT = '/workspace/exp/R1'; os.makedirs(f'{OUT}/gen', exist_ok=True); T = [399]
GROUPS = [(45, 0.20), (45, 0.25), (45, 0.30), (30, 0.25)]
def gname(th, rho): return f"th{th}_rho{rho:.2f}"
ids = load_ids(); keys = list(ids); B1 = json.load(open('/workspace/exp/B1/results.json'))['results']
pipe = build_faceid("dmd2_sdxl_1step_unet_fp16.bin", False, lora=False)
R = json.load(open(f'{OUT}/results.json')) if os.path.exists(f'{OUT}/results.json') else {}
for k, rimg in ids.items():
    if k in R and gname(30, 0.25) in R[k]: continue
    remb = m.arc_embed(cv2.cvtColor(np.asarray(rimg), cv2.COLOR_RGB2BGR)); rdino = m.dino_embed(rimg); emb = face_embeds(pipe, rimg); R[k] = {'step1': B1[k]['step1'], 'th45_rho0.15': B1[k]['A_ref']}
    src = Image.open(f'/workspace/exp/B1/gen/{k}_src_A_ref.png').convert('RGB'); z = encode(pipe, src)
    for th, rho in GROUPS:
        l0 = rotate_lowfreq(noise_for(42), z, 0, rho); assert float((l0.float() - noise_for(42).float()).norm() / noise_for(42).float().norm()) < 2e-3
        paths = []
        for s in SEEDS:
            img = gen(pipe, emb, rotate_lowfreq(noise_for(s), z, th, rho), s, T); q = f'{OUT}/gen/{k}_{gname(th, rho)}_s{s}.png'; img.save(q); check_image(img); paths.append(q)
        R[k][gname(th, rho)] = metrics(paths, remb, rdino, pad_recover=True); rep(k, gname(th, rho), R[k][gname(th, rho)]); json.dump(R, open(f'{OUT}/results.json', 'w'), indent=1)
ok = [k for k in keys if gname(30, 0.25) in R.get(k, {})]; S = dict(n=len(ok), step1=float(np.mean([R[k]['step1']['arcface'] for k in ok])), groups={})
for g in ['th45_rho0.15'] + [gname(th, rho) for th, rho in GROUPS]:
    p = paired(R, ok, g, 'step1'); p['sg'] = paired(R, ok, g, 'step1', 'p_sunglasses'); p['beach'] = paired(R, ok, g, 'step1', 'p_beach'); p['dino'] = float(np.mean([R[k][g]['dino'] for k in ok])); p['dino_gt06'] = int(sum(R[k][g]['dino'] > 0.6 for k in ok)); p['nan'] = int(sum(R[k][g]['n_nan'] for k in ok)); S['groups'][g] = p
    print(f"{g:13s} arcface {p['mean_a']:.3f} Δ{p['delta']:+.4f} SE {p['delta_se']:.4f} p_t {p['p_t']:.3g} p_w {p['p_wilcoxon']:.3g} {p['positive']}/{p['n']} | sg Δ{p['sg']['delta']:+.3f} beach Δ{p['beach']['delta']:+.3f} dino {p['dino']:.3f} (>0.6: {p['dino_gt06']}) nan {p['nan']}", flush=True)
json.dump(S, open(f'{OUT}/summary.json', 'w'), indent=1); print("DONE", flush=True)
