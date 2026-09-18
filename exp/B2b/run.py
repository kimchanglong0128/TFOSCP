# Card B2b: Hyper-SD (1-step UNet t=800) + FaceID with the identity-neutral MEAN-FACE source (B1 finding), theta in {45,60,75}, rho=0.15, 30 ids x 20 seeds.
# Mean face cropped to its box, pasted on the beach bg at the per-identity mean 1-step box (from B2 baselines), eye band grayed. step1 reused from B2.
import sys; sys.path.insert(0, '/workspace/exp/cards')
from lib import *
OUT = '/workspace/exp/B2b'; os.makedirs(f'{OUT}/gen', exist_ok=True); T = [800]; RHO = 0.15; THETAS = (45, 60, 75)
ids = load_ids(); keys = list(ids); B2 = json.load(open('/workspace/exp/B2/results.json'))['results']
CANON = np.array([320., 256., 704., 768.])
mf = Image.fromarray(np.mean([np.asarray(paste_aligned(v, ref_box(v), CANON, bg=Image.new('RGB', (1024, 1024), (128, 128, 128))), dtype=np.float32) for v in ids.values()], 0).round().astype(np.uint8))
def crop_paste(tb):
    x1, y1, x2, y2 = CANON; w, h = x2 - x1, y2 - y1; mg = 0.15; crop = mf.crop((int(x1 - mg*w), int(y1 - mg*h), int(x2 + mg*w), int(y2 + mg*h)))
    rb = np.array([mg*w, mg*h, mg*w + w, mg*h + h]); return gray_region(paste_aligned(crop, rb, tb), tb, 'eyes')
pipe = build_hyper_1step()
R = json.load(open(f'{OUT}/results.json')) if os.path.exists(f'{OUT}/results.json') else {}
for k, rimg in ids.items():
    if k in R and 'mean_th75' in R[k]: continue
    remb = m.arc_embed(cv2.cvtColor(np.asarray(rimg), cv2.COLOR_RGB2BGR)); rdino = m.dino_embed(rimg); emb = face_embeds(pipe, rimg); R[k] = {'step1': B2[k]['step1'], 'ref_th45': B2[k]['th45_rho0.15'], 'ref_th75': B2[k]['th75_rho0.15']}
    boxes = [b for b in (m.face_bbox(cv2.imread(f'/workspace/exp/B2/gen/{k}_step1_s{s}.png')) for s in SEEDS) if b is not None]; tb = np.array(boxes).mean(0)
    src = crop_paste(tb); z = encode(pipe, src)
    if k == 'id00': src.save(f'{OUT}/gen/{k}_src_mean.png')
    l0 = rotate_lowfreq(noise_for(42), z, 0, RHO); assert float((l0.float() - noise_for(42).float()).norm() / noise_for(42).float().norm()) < 2e-3
    for th in THETAS:
        paths = []
        for s in SEEDS:
            img = gen(pipe, emb, rotate_lowfreq(noise_for(s), z, th, RHO), s, T); q = f'{OUT}/gen/{k}_mean_th{th}_s{s}.png'; img.save(q); check_image(img); paths.append(q)
        R[k][f'mean_th{th}'] = metrics(paths, remb, rdino, pad_recover=True); rep(k, f'mean_th{th}', R[k][f'mean_th{th}'])
    json.dump(R, open(f'{OUT}/results.json', 'w'), indent=1)
ok = [k for k in keys if 'mean_th75' in R.get(k, {})]; S = dict(n=len(ok), step1=float(np.mean([R[k]['step1']['arcface'] for k in ok])), groups={})
for g in ('ref_th45', 'ref_th75', 'mean_th45', 'mean_th60', 'mean_th75'):
    p = paired(R, ok, g, 'step1'); p['sg'] = paired(R, ok, g, 'step1', 'p_sunglasses'); p['beach'] = paired(R, ok, g, 'step1', 'p_beach'); p['dino'] = float(np.mean([R[k][g]['dino'] for k in ok])); p['nan'] = int(sum(R[k][g]['n_nan'] for k in ok)); S['groups'][g] = p
    print(f"{g:10s} arcface {p['mean_a']:.3f} Δ{p['delta']:+.4f} SE {p['delta_se']:.4f} p_t {p['p_t']:.3g} p_w {p['p_wilcoxon']:.3g} {p['positive']}/{p['n']} | sg Δ{p['sg']['delta']:+.3f} beach Δ{p['beach']['delta']:+.3f} dino {p['dino']:.3f} nan {p['nan']}", flush=True)
S['mean45_vs_ref45'] = paired(R, ok, 'mean_th45', 'ref_th45'); S['mean75_vs_ref75'] = paired(R, ok, 'mean_th75', 'ref_th75')
json.dump(S, open(f'{OUT}/summary.json', 'w'), indent=1); print("mean45 vs ref45", S['mean45_vs_ref45']['delta'], S['mean45_vs_ref45']['p_t'], "| mean75 vs ref75", S['mean75_vs_ref75']['delta'], S['mean75_vs_ref75']['p_t']); print("DONE", flush=True)
