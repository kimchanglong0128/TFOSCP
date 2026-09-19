# Scheme SW1: spatially weighted low-frequency rotation. Rotate the low band globally as before, then blend rotated/original low bands with a soft
# window w (1 inside the per-identity mean 1-step face box x1.3, Gaussian falloff to w_bg outside), re-project the low band to its original norm.
# Goal: keep the identity gain of R1 (theta45, rho0.25: +0.062) while halving the background-probe cost (-0.137). DMD2 + FaceID, 30 ids x 20 seeds.
#   groups: (45, 0.25, w_bg=0.0), (45, 0.25, w_bg=0.3), (45, 0.30, w_bg=0.0). step1 / global rows reused from B1 / R1.
import sys; sys.path.insert(0, '/workspace/exp/cards')
from lib import *
import torch.nn.functional as F
OUT = '/workspace/exp/SW1'; T = [399]
GROUPS = [(45, 0.25, 0.0), (45, 0.25, 0.3), (45, 0.30, 0.0)]
def gname(th, rho, wbg): return f"sw_th{th}_rho{rho:.2f}_wbg{wbg:.1f}"
ids = load_ids(); keys = list(ids); B1 = json.load(open('/workspace/exp/B1/results.json'))['results']; R1 = json.load(open('/workspace/exp/R1/results.json'))
def window(tb, wbg, expand=1.3, sigma=6.0):
    x1, y1, x2, y2 = [v / 8 for v in tb]; cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * expand, (y2 - y1) * expand
    yy, xx = torch.meshgrid(torch.arange(128., device='cuda'), torch.arange(128., device='cuda'), indexing='ij')
    dx = (xx - cx).abs() - w / 2; dy = (yy - cy).abs() - h / 2; d = torch.sqrt(dx.clamp_min(0) ** 2 + dy.clamp_min(0) ** 2)     # distance outside the box (0 inside)
    return (wbg + (1 - wbg) * torch.exp(-(d / sigma) ** 2))[None, None]                                                        # (1,1,128,128)
def rotate_windowed(eps, src, th, rho, win):
    eL, eH = split(eps, rho); rot = rotate_lowfreq(eps, src, th, rho).float(); rL = rot - eH
    mix = win * rL + (1 - win) * eL; mix = mix * (eL.norm() / mix.norm()); out = eH + mix
    assert abs(float(out.norm()) - float(eps.float().norm())) / float(eps.float().norm()) < 2e-3; return out.to(eps.dtype)
pipe = build_faceid("dmd2_sdxl_1step_unet_fp16.bin", False, lora=False)
R = json.load(open(f'{OUT}/results.json')) if os.path.exists(f'{OUT}/results.json') else {}
for k, rimg in ids.items():
    if k in R and gname(*GROUPS[-1]) in R[k]: continue
    remb = m.arc_embed(cv2.cvtColor(np.asarray(rimg), cv2.COLOR_RGB2BGR)); rdino = m.dino_embed(rimg); emb = face_embeds(pipe, rimg)
    R[k] = {'step1': B1[k]['step1'], 'g_th45_rho0.25': R1[k]['th45_rho0.25'], 'g_th45_rho0.30': R1[k]['th45_rho0.30'], 'g_th45_rho0.15': B1[k]['A_ref']}
    boxes = [b for b in (m.face_bbox(cv2.imread(f'/workspace/exp/B1/gen/{k}_step1_s{s}.png')) for s in SEEDS) if b is not None]; tb = np.array(boxes).mean(0)
    z = encode(pipe, Image.open(f'/workspace/exp/B1/gen/{k}_src_A_ref.png').convert('RGB'))
    for th, rho, wbg in GROUPS:
        win = window(tb, wbg); e0 = noise_for(42); r0 = rotate_windowed(e0, z, 0, rho, win); assert float((r0.float() - e0.float()).norm() / e0.float().norm()) < 2e-3
        if k == 'id00': R[k][gname(th, rho, wbg) + '_win_mean'] = float(win.mean())
        paths = []
        for s in SEEDS:
            img = gen(pipe, emb, rotate_windowed(noise_for(s), z, th, rho, win), s, T); q = f'{OUT}/gen/{k}_{gname(th, rho, wbg)}_s{s}.png'; img.save(q); check_image(img); paths.append(q)
        R[k][gname(th, rho, wbg)] = metrics(paths, remb, rdino, pad_recover=True); rep(k, gname(th, rho, wbg), R[k][gname(th, rho, wbg)]); json.dump(R, open(f'{OUT}/results.json', 'w'), indent=1)
ok = [k for k in keys if gname(*GROUPS[-1]) in R.get(k, {})]; S = dict(n=len(ok), rows={})
for g in ['g_th45_rho0.15', 'g_th45_rho0.25', 'g_th45_rho0.30'] + [gname(*g_) for g_ in GROUPS]:
    p = paired(R, ok, g, 'step1'); p['sg'] = paired(R, ok, g, 'step1', 'p_sunglasses')['delta']; p['beach'] = paired(R, ok, g, 'step1', 'p_beach')['delta']; p['dino'] = float(np.mean([R[k][g]['dino'] for k in ok])); p['nan'] = int(sum(R[k][g]['n_nan'] for k in ok)); S['rows'][g] = p
    print(f"{g:26s} arcface {p['mean_a']:.3f} Δ{p['delta']:+.4f} p={p['p_t']:.3g} {p['positive']}/{p['n']} | sg Δ{p['sg']:+.3f} beach Δ{p['beach']:+.3f} dino {p['dino']:.3f} nan {p['nan']}", flush=True)
S['sw25_vs_g25'] = paired(R, ok, gname(45, 0.25, 0.0), 'g_th45_rho0.25'); S['sw25_vs_g25_beach'] = paired(R, ok, gname(45, 0.25, 0.0), 'g_th45_rho0.25', 'p_beach')
print("windowed vs global (rho0.25): arcface", S['sw25_vs_g25']['delta'], S['sw25_vs_g25']['p_t'], "| beach", S['sw25_vs_g25_beach']['delta'], S['sw25_vs_g25_beach']['p_t'])
json.dump(S, open(f'{OUT}/summary.json', 'w'), indent=1); print("DONE", flush=True)
