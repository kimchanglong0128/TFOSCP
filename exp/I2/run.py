# Card I2: is the sex effect on the target side or the source side? Sex-matched mean-face sources on DMD2 + FaceID, 30 ids x 20 seeds, theta=45, rho=0.15.
#   mean_M / mean_F : mean face of the 17 male / 13 female references (insightface sex), cropped to box, pasted on beach at the per-identity mean 1-step box, eye band grayed.
#   Each identity gets BOTH sources -> matched vs mismatched. step1 / A_ref / B_crop (all-ids mean) reused from B1 / B1b.
import sys; sys.path.insert(0, '/workspace/exp/cards')
from lib import *
OUT = '/workspace/exp/I2'; T = [399]; RHO, THETA = 0.15, 45
ids = load_ids(); keys = list(ids); B1 = json.load(open('/workspace/exp/B1/results.json'))['results']; B1b = json.load(open('/workspace/exp/B1b/results.json'))
sex = {r['id']: r['sex'] for r in json.load(open('/workspace/exp/I1/rows.json'))}
CANON = np.array([320., 256., 704., 768.])
warped = {k: np.asarray(paste_aligned(v, ref_box(v), CANON, bg=Image.new('RGB', (1024, 1024), (128, 128, 128))), dtype=np.float32) for k, v in ids.items()}
MF = {s: Image.fromarray(np.mean([warped[k] for k in keys if sex[k] == s], 0).round().astype(np.uint8)) for s in (1, 0)}
for s in MF: MF[s].save(f'{OUT}/mean_face_{"M" if s == 1 else "F"}.png')
def crop_paste(mf, tb):
    x1, y1, x2, y2 = CANON; w, h = x2 - x1, y2 - y1; mg = 0.15; crop = mf.crop((int(x1 - mg*w), int(y1 - mg*h), int(x2 + mg*w), int(y2 + mg*h)))
    rb = np.array([mg*w, mg*h, mg*w + w, mg*h + h]); return gray_region(paste_aligned(crop, rb, tb), tb, 'eyes')
pipe = build_faceid("dmd2_sdxl_1step_unet_fp16.bin", False, lora=False)
R = json.load(open(f'{OUT}/results.json')) if os.path.exists(f'{OUT}/results.json') else {}
for k, rimg in ids.items():
    if k in R and 'mean_F' in R[k]: continue
    remb = m.arc_embed(cv2.cvtColor(np.asarray(rimg), cv2.COLOR_RGB2BGR)); rdino = m.dino_embed(rimg); emb = face_embeds(pipe, rimg)
    R[k] = {'sex': sex[k], 'step1': B1[k]['step1'], 'A_ref': B1[k]['A_ref'], 'B_crop': B1b[k]['B_crop']}
    boxes = [b for b in (m.face_bbox(cv2.imread(f'/workspace/exp/B1/gen/{k}_step1_s{s}.png')) for s in SEEDS) if b is not None]; tb = np.array(boxes).mean(0)
    for s_, name in ((1, 'mean_M'), (0, 'mean_F')):
        z = encode(pipe, crop_paste(MF[s_], tb)); paths = []
        for s in SEEDS:
            img = gen(pipe, emb, rotate_lowfreq(noise_for(s), z, THETA, RHO), s, T); q = f'{OUT}/gen/{k}_{name}_s{s}.png'; img.save(q); check_image(img); paths.append(q)
        R[k][name] = metrics(paths, remb, rdino, pad_recover=True); rep(k, name, R[k][name])
    json.dump(R, open(f'{OUT}/results.json', 'w'), indent=1)
ok = [k for k in keys if 'mean_F' in R.get(k, {})]
def grp(sel, a, b):
    kk = [k for k in ok if sel(R[k]['sex'])]; x = np.array([R[k][a]['arcface'] for k in kk]); y = np.array([R[k][b]['arcface'] for k in kk]); d = x - y
    return dict(n=len(kk), delta=float(d.mean()), positive=int((d > 0).sum()), p_t=float(stats.ttest_rel(x, y).pvalue))
S = {}
for grpname, sel in (('male', lambda s: s == 1), ('female', lambda s: s == 0), ('all', lambda s: True)):
    for src in ('A_ref', 'B_crop', 'mean_M', 'mean_F'):
        S[f'{grpname}_{src}'] = grp(sel, src, 'step1'); p = S[f'{grpname}_{src}']; print(f"{grpname:6s} {src:7s} vs step1 Δ{p['delta']:+.4f} {p['positive']}/{p['n']} p={p['p_t']:.3g}", flush=True)
    S[f'{grpname}_matched_vs_mismatched'] = None
mm = [(R[k]['mean_M' if R[k]['sex'] == 1 else 'mean_F']['arcface'], R[k]['mean_F' if R[k]['sex'] == 1 else 'mean_M']['arcface']) for k in ok]
x, y = np.array(mm).T; S['matched_vs_mismatched'] = dict(delta=float((x - y).mean()), positive=int(((x - y) > 0).sum()), n=len(ok), p_t=float(stats.ttest_rel(x, y).pvalue)); print("matched vs mismatched mean face:", S['matched_vs_mismatched'])
json.dump(S, open(f'{OUT}/summary.json', 'w'), indent=1); print("DONE", flush=True)
