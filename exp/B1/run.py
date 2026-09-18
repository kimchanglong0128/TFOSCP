# Card B1: does the gain come from identity entering through the low frequencies, or from a layout prior?
# DMD2-SDXL 1-step (t=399) + IP-Adapter FaceID-PlusV2, 30 FFHQ ids x 20 seeds, rho=0.15, theta=45. Only the SOURCE image changes:
#   A ref        : reference face aligned to mean 1-step face box, eye band grayed, beach bg (current method)
#   B meanface   : pixel-mean face of the 30 ids (each warped to a canonical box) in place of the reference face; rest identical
#   C silhouette : gray ellipse inscribed in the face box, no features; beach bg
#   D other_id   : another identity (next id, cyclic) aligned to the same box; eye band grayed
# Controls: theta=0 must reproduce the baseline pixel-wise; regenerated baseline must match AE-2's stored 1-step images; z_L cosines B/C/D vs A.
import sys; sys.path.insert(0, '/workspace/exp/cards')
from lib import *
OUT = '/workspace/exp/B1'; os.makedirs(f'{OUT}/gen', exist_ok=True); THETA = 45; RHO = 0.15
ids = load_ids(); keys = list(ids)                                   # 30 FFHQ ids
CANON = np.array([320., 256., 704., 768.])                           # canonical face box for the mean face (w=h=384... box 384x512)
def warp_to_canon(img):
    return paste_aligned(img, ref_box(img), CANON, bg=Image.new('RGB', (1024, 1024), (128, 128, 128)))
mean_face = Image.fromarray(np.mean([np.asarray(warp_to_canon(v), dtype=np.float32) for v in ids.values()], 0).round().astype(np.uint8))
mean_face.save(f'{OUT}/mean_face.png')
def build_sources(k, rimg, boxes):
    tb = np.array(boxes).mean(0)
    A = make_source(rimg, boxes)
    B = make_source(mean_face, boxes, rb=CANON)
    C = beach_bg.copy(); ImageDraw.Draw(C).ellipse(list(tb), fill=(128, 128, 128))
    other = ids[keys[(keys.index(k) + 1) % len(keys)]]; D = make_source(other, boxes)
    return dict(A_ref=A, B_meanface=B, C_silhouette=C, D_other=D)
pipe = build_faceid("dmd2_sdxl_1step_unet_fp16.bin", False, lora=False)
R, ctrl = {}, {}
if os.path.exists(f'{OUT}/results.json'):
    _p = json.load(open(f'{OUT}/results.json')); R, ctrl = _p['results'], _p['theta0_ctrl']; R = {k: v for k, v in R.items() if 'D_other' in v}; ctrl = {k: v for k, v in ctrl.items() if k in R}; print('resume: done ids', sorted(R), flush=True)
for k, rimg in ids.items():
    if k in R: continue
    remb = m.arc_embed(cv2.cvtColor(np.asarray(rimg), cv2.COLOR_RGB2BGR)); rdino = m.dino_embed(rimg); emb = face_embeds(pipe, rimg); R[k] = {}
    # baseline (regenerated) + determinism check against AE-2
    paths = []; maxdiff = 0
    for s in SEEDS:
        img = gen(pipe, emb, noise_for(s), s, [399]); q = f'{OUT}/gen/{k}_step1_s{s}.png'; img.save(q); check_image(img); paths.append(q)
        old = f'/workspace/exp/AE2/gen/{k}_1step_s{s}.png'
        if os.path.exists(old): maxdiff = max(maxdiff, int(np.abs(np.asarray(img).astype(int) - np.asarray(Image.open(old)).astype(int)).max()))
    R[k]['step1'] = metrics(paths, remb, rdino); rep(k, 'step1', R[k]['step1']); R[k]['step1']['maxdiff_vs_AE2'] = maxdiff
    boxes = [b for b in (m.face_bbox(cv2.imread(q)) for q in paths) if b is not None]
    if len(boxes) < 5: print(k, "skipped (few faces)", flush=True); continue
    srcs = build_sources(k, rimg, boxes); Z = {}
    for name, im in srcs.items(): im.save(f'{OUT}/gen/{k}_src_{name}.png'); Z[name] = encode(pipe, im)
    R[k]['zL_cos_vs_A'] = {n: lowfreq_cos(Z[n], Z['A_ref'], RHO) for n in Z}
    # theta=0 control on seed 42 for every source: latent must equal eps, image must equal baseline
    e0 = noise_for(42); base = np.asarray(Image.open(f'{OUT}/gen/{k}_step1_s42.png')).astype(int); c = {}
    for name in Z:
        l0 = rotate_lowfreq(e0, Z[name], 0, RHO); rel = float((l0.float() - e0.float()).norm() / e0.float().norm())
        img0 = gen(pipe, emb, l0, 42, [399]); c[name] = dict(latent_rel_err=rel, max_pix_diff=int(np.abs(np.asarray(img0).astype(int) - base).max()))
        assert rel < 2e-3, f"theta=0 latent differs: {rel}"
    ctrl[k] = c; print(k, "theta0 ctrl", {n: (round(v['latent_rel_err'], 5), v['max_pix_diff']) for n, v in c.items()}, flush=True)
    for name in Z:
        paths = []
        for s in SEEDS:
            img = gen(pipe, emb, rotate_lowfreq(noise_for(s), Z[name], THETA, RHO), s, [399]); q = f'{OUT}/gen/{k}_{name}_s{s}.png'; img.save(q); check_image(img); paths.append(q)
        R[k][name] = metrics(paths, remb, rdino); rep(k, name, R[k][name])
    json.dump(dict(results=R, theta0_ctrl=ctrl), open(f'{OUT}/results.json', 'w'), indent=1)
ok = [k for k in keys if 'D_other' in R.get(k, {})]
S = dict(n=len(ok), means={c: float(np.mean([R[k][c]['arcface'] for k in ok])) for c in ('step1', 'A_ref', 'B_meanface', 'C_silhouette', 'D_other')},
         A_vs_B=paired(R, ok, 'A_ref', 'B_meanface'), A_vs_C=paired(R, ok, 'A_ref', 'C_silhouette'), A_vs_D=paired(R, ok, 'A_ref', 'D_other'),
         A_vs_step1=paired(R, ok, 'A_ref', 'step1'), B_vs_step1=paired(R, ok, 'B_meanface', 'step1'), C_vs_step1=paired(R, ok, 'C_silhouette', 'step1'), D_vs_step1=paired(R, ok, 'D_other', 'step1'),
         zL_cos_vs_A={n: float(np.mean([R[k]['zL_cos_vs_A'][n] for k in ok])) for n in ('B_meanface', 'C_silhouette', 'D_other')},
         theta0_max_pix_diff=int(max(v['max_pix_diff'] for c in ctrl.values() for v in c.values())), step1_maxdiff_vs_AE2=int(max(R[k]['step1']['maxdiff_vs_AE2'] for k in ok)),
         nan={c: int(sum(R[k][c]['n_nan'] for k in ok)) for c in ('step1', 'A_ref', 'B_meanface', 'C_silhouette', 'D_other')})
for key in ('p_sunglasses', 'p_beach', 'dino'):
    S[key] = {c: float(np.mean([R[k][c][key] for k in ok])) for c in ('step1', 'A_ref', 'B_meanface', 'C_silhouette', 'D_other')}
json.dump(S, open(f'{OUT}/summary.json', 'w'), indent=1); print(json.dumps(S, indent=1)); print("DONE", flush=True)
