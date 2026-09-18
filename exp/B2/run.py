# Card B2: theta sweep on Hyper-SD (1-step UNet, t=800) + FaceID, 30 FFHQ ids x 20 seeds.
#   theta in {15,25,35,45,60,75} at rho=0.15; plus rho in {0.10,0.20} at theta=45. Baseline regenerated and checked against AE-3 images.
# Verdict rule (card): exists theta* with paired p<0.05 on ArcFace and sunglasses-probe drop < 0.02.
import sys; sys.path.insert(0, '/workspace/exp/cards')
from lib import *
OUT = '/workspace/exp/B2'; os.makedirs(f'{OUT}/gen', exist_ok=True); T = [800]
GROUPS = [(th, 0.15) for th in (15, 25, 35, 45, 60, 75)] + [(45, 0.10), (45, 0.20)]
def gname(th, rho): return f"th{th}_rho{rho:.2f}"
ids = load_ids(); keys = list(ids)
pipe = build_hyper_1step(); R = {}; ctrl = {}
for k, rimg in ids.items():
    remb = m.arc_embed(cv2.cvtColor(np.asarray(rimg), cv2.COLOR_RGB2BGR)); rdino = m.dino_embed(rimg); emb = face_embeds(pipe, rimg); R[k] = {}
    paths = []; maxdiff = 0
    for s in SEEDS:
        img = gen(pipe, emb, noise_for(s), s, T); q = f'{OUT}/gen/{k}_step1_s{s}.png'; img.save(q); check_image(img); paths.append(q)
        old = f'/workspace/exp/AE3/gen/{k}_1step_s{s}.png'
        if os.path.exists(old): maxdiff = max(maxdiff, int(np.abs(np.asarray(img).astype(int) - np.asarray(Image.open(old)).astype(int)).max()))
    R[k]['step1'] = metrics(paths, remb, rdino); R[k]['step1']['maxdiff_vs_AE3'] = maxdiff; rep(k, 'step1', R[k]['step1'])
    boxes = [b for b in (m.face_bbox(cv2.imread(q)) for q in paths) if b is not None]
    if len(boxes) < 5: print(k, "skipped (few faces)", flush=True); continue
    src = make_source(rimg, boxes); src.save(f'{OUT}/gen/{k}_src.png'); z = encode(pipe, src)
    e0 = noise_for(42); base = np.asarray(Image.open(f'{OUT}/gen/{k}_step1_s42.png')).astype(int); c = {}
    for rho in (0.10, 0.15, 0.20):
        l0 = rotate_lowfreq(e0, z, 0, rho); rel = float((l0.float() - e0.float()).norm() / e0.float().norm()); assert rel < 2e-3, rel
        c[f'rho{rho:.2f}'] = dict(latent_rel_err=rel, max_pix_diff=int(np.abs(np.asarray(gen(pipe, emb, l0, 42, T)).astype(int) - base).max()))
    ctrl[k] = c; print(k, "theta0 ctrl", {n: (round(v['latent_rel_err'], 5), v['max_pix_diff']) for n, v in c.items()}, flush=True)
    for th, rho in GROUPS:
        paths = []
        for s in SEEDS:
            img = gen(pipe, emb, rotate_lowfreq(noise_for(s), z, th, rho), s, T); q = f'{OUT}/gen/{k}_{gname(th, rho)}_s{s}.png'; img.save(q); check_image(img); paths.append(q)
        R[k][gname(th, rho)] = metrics(paths, remb, rdino); rep(k, gname(th, rho), R[k][gname(th, rho)])
    json.dump(dict(results=R, theta0_ctrl=ctrl), open(f'{OUT}/results.json', 'w'), indent=1)
ok = [k for k in keys if gname(45, 0.20) in R.get(k, {})]
S = dict(n=len(ok), step1=float(np.mean([R[k]['step1']['arcface'] for k in ok])), groups={}, theta0_max_pix_diff=int(max(v['max_pix_diff'] for c in ctrl.values() for v in c.values())),
         step1_maxdiff_vs_AE3=int(max(R[k]['step1']['maxdiff_vs_AE3'] for k in ok)))
for th, rho in GROUPS:
    g = gname(th, rho); p = paired(R, ok, g, 'step1'); p['sg'] = paired(R, ok, g, 'step1', 'p_sunglasses'); p['beach'] = paired(R, ok, g, 'step1', 'p_beach'); p['dino'] = paired(R, ok, g, 'step1', 'dino')
    p['nan'] = int(sum(R[k][g]['n_nan'] for k in ok)); S['groups'][g] = p
    print(f"{g:14s} arcface {p['mean_a']:.3f} (Δ{p['delta']:+.3f}, p={p['p_t']:.3g}, {p['positive']}/{p['n']}) sg Δ{p['sg']['delta']:+.3f} beach Δ{p['beach']['delta']:+.3f} dino Δ{p['dino']['delta']:+.3f} nan {p['nan']}", flush=True)
json.dump(S, open(f'{OUT}/summary.json', 'w'), indent=1); print("DONE", flush=True)
