# Card B1b (follow-up to B1): remove the background confound of source B and test a fully reference-free prior.
#   B_crop      : mean face cropped to its face box (x1.15 margin), pasted on the beach bg at the per-identity mean 1-step box, eye band grayed
#   B_fixedbox  : same, but at ONE global box (mean of all 30 identities' 1-step boxes) -> no per-identity calibration, 0 generation needed
#   B_loo       : as B_crop but the mean face excludes the current identity (leave-one-out)
# Same protocol as B1 (DMD2 + FaceID, 30 ids x 20 seeds, theta=45, rho=0.15); step1 and A_ref reused from B1.
import sys; sys.path.insert(0, '/workspace/exp/cards')
from lib import *
OUT = '/workspace/exp/B1b'; os.makedirs(f'{OUT}/gen', exist_ok=True); THETA, RHO = 45, 0.15
ids = load_ids(); keys = list(ids); B1 = json.load(open('/workspace/exp/B1/results.json'))['results']
CANON = np.array([320., 256., 704., 768.])
warped = {k: np.asarray(paste_aligned(v, ref_box(v), CANON, bg=Image.new('RGB', (1024, 1024), (128, 128, 128))), dtype=np.float32) for k, v in ids.items()}
def mean_face(exclude=None): return Image.fromarray(np.mean([w for k, w in warped.items() if k != exclude], 0).round().astype(np.uint8))
def crop_paste(mf, tb):
    x1, y1, x2, y2 = CANON; w, h = x2 - x1, y2 - y1; mg = 0.15; crop = mf.crop((int(x1 - mg*w), int(y1 - mg*h), int(x2 + mg*w), int(y2 + mg*h)))
    rb = np.array([mg*w, mg*h, mg*w + w, mg*h + h]); return gray_region(paste_aligned(crop, rb, tb), tb, 'eyes')
boxes_all = {k: [b for b in (m.face_bbox(cv2.imread(f'/workspace/exp/B1/gen/{k}_step1_s{s}.png')) for s in SEEDS) if b is not None] for k in keys}
GLOBAL_BOX = np.mean([np.array(v).mean(0) for v in boxes_all.values() if len(v) >= 5], 0); print("global box", GLOBAL_BOX.round(1), flush=True)
pipe = build_faceid("dmd2_sdxl_1step_unet_fp16.bin", False, lora=False)
R = json.load(open(f'{OUT}/results.json')) if os.path.exists(f'{OUT}/results.json') else {}
for k, rimg in ids.items():
    if k in R and 'B_loo' in R[k]: continue
    remb = m.arc_embed(cv2.cvtColor(np.asarray(rimg), cv2.COLOR_RGB2BGR)); rdino = m.dino_embed(rimg); emb = face_embeds(pipe, rimg); R[k] = {'step1': B1[k]['step1'], 'A_ref': B1[k]['A_ref'], 'B_meanface': B1[k]['B_meanface']}
    tb = np.array(boxes_all[k]).mean(0); mf = mean_face(); srcs = {'B_crop': crop_paste(mf, tb), 'B_fixedbox': crop_paste(mf, GLOBAL_BOX), 'B_loo': crop_paste(mean_face(exclude=k), tb)}
    for name, im in srcs.items():
        if k == 'id00': im.save(f'{OUT}/gen/{k}_src_{name}.png')
        z = encode(pipe, im); l0 = rotate_lowfreq(noise_for(42), z, 0, RHO); assert float((l0.float() - noise_for(42).float()).norm() / noise_for(42).float().norm()) < 2e-3
        paths = []
        for s in SEEDS:
            img = gen(pipe, emb, rotate_lowfreq(noise_for(s), z, THETA, RHO), s, [399]); q = f'{OUT}/gen/{k}_{name}_s{s}.png'; img.save(q); check_image(img); paths.append(q)
        R[k][name] = metrics(paths, remb, rdino, pad_recover=True); rep(k, name, R[k][name])
    json.dump(R, open(f'{OUT}/results.json', 'w'), indent=1)
ok = [k for k in keys if 'B_loo' in R.get(k, {})]; S = dict(n=len(ok), global_box=GLOBAL_BOX.tolist(), means={g: float(np.mean([R[k][g]['arcface'] for k in ok])) for g in ('step1', 'A_ref', 'B_meanface', 'B_crop', 'B_fixedbox', 'B_loo')})
for g in ('B_crop', 'B_fixedbox', 'B_loo'):
    S[f'{g}_vs_step1'] = paired(R, ok, g, 'step1'); S[f'{g}_vs_A'] = paired(R, ok, g, 'A_ref'); S[f'{g}_vs_B'] = paired(R, ok, g, 'B_meanface')
    for key in ('p_sunglasses', 'p_beach', 'dino'): S[f'{g}_{key}'] = float(np.mean([R[k][g][key] for k in ok]))
    print(f"{g:11s} arcface {S['means'][g]:.3f} vs step1 Δ{S[f'{g}_vs_step1']['delta']:+.4f} p={S[f'{g}_vs_step1']['p_t']:.3g} {S[f'{g}_vs_step1']['positive']}/{len(ok)} | vs A Δ{S[f'{g}_vs_A']['delta']:+.4f} p={S[f'{g}_vs_A']['p_t']:.3g} | vs B Δ{S[f'{g}_vs_B']['delta']:+.4f} p={S[f'{g}_vs_B']['p_t']:.3g} | sg {S[f'{g}_p_sunglasses']:.3f} beach {S[f'{g}_p_beach']:.3f} dino {S[f'{g}_dino']:.3f}", flush=True)
json.dump(S, open(f'{OUT}/summary.json', 'w'), indent=1); print("DONE", flush=True)
