# Scheme BG1: does the background cost come from the BLURRED beach in the source? Same as R1 (theta45, rho0.25) but the source background blur is
# varied: blur 0 (sharp text-generated beach), blur 6 (R1 row, reused), blur 14. Also blur 0 at rho0.15. DMD2 + FaceID, 30 ids x 20 seeds.
import sys; sys.path.insert(0, '/workspace/exp/cards')
from lib import *
import lib
OUT = '/workspace/exp/BG1'; T = [399]
GROUPS = [('blur0_rho0.25', 0, 0.25), ('blur14_rho0.25', 14, 0.25), ('blur0_rho0.15', 0, 0.15)]
ids = load_ids(); keys = list(ids); B1 = json.load(open('/workspace/exp/B1/results.json'))['results']; R1 = json.load(open('/workspace/exp/R1/results.json'))
raw_bg = Image.open('/workspace/exp/AD6/bg_beach.png').convert('RGB')
def bg(blur): return raw_bg if blur == 0 else raw_bg.filter(ImageFilter.GaussianBlur(blur))
pipe = build_faceid("dmd2_sdxl_1step_unet_fp16.bin", False, lora=False)
R = json.load(open(f'{OUT}/results.json')) if os.path.exists(f'{OUT}/results.json') else {}
for k, rimg in ids.items():
    if k in R and GROUPS[-1][0] in R[k]: continue
    remb = m.arc_embed(cv2.cvtColor(np.asarray(rimg), cv2.COLOR_RGB2BGR)); rdino = m.dino_embed(rimg); emb = face_embeds(pipe, rimg)
    R[k] = {'step1': B1[k]['step1'], 'blur6_rho0.15': B1[k]['A_ref'], 'blur6_rho0.25': R1[k]['th45_rho0.25']}
    boxes = [b for b in (m.face_bbox(cv2.imread(f'/workspace/exp/B1/gen/{k}_step1_s{s}.png')) for s in SEEDS) if b is not None]; tb = np.array(boxes).mean(0)
    for name, blur, rho in GROUPS:
        src = gray_region(paste_aligned(rimg, ref_box(rimg), tb, bg=bg(blur)), tb, 'eyes'); z = encode(pipe, src)
        if k == 'id00': src.save(f'{OUT}/gen/{k}_src_{name}.png')
        paths = []
        for s in SEEDS:
            img = gen(pipe, emb, rotate_lowfreq(noise_for(s), z, 45, rho), s, T); q = f'{OUT}/gen/{k}_{name}_s{s}.png'; img.save(q); check_image(img); paths.append(q)
        R[k][name] = metrics(paths, remb, rdino, pad_recover=True); rep(k, name, R[k][name]); json.dump(R, open(f'{OUT}/results.json', 'w'), indent=1)
ok = [k for k in keys if GROUPS[-1][0] in R.get(k, {})]; S = dict(n=len(ok), rows={})
for g in ['blur6_rho0.15', 'blur0_rho0.15', 'blur6_rho0.25', 'blur0_rho0.25', 'blur14_rho0.25']:
    p = paired(R, ok, g, 'step1'); p['sg'] = paired(R, ok, g, 'step1', 'p_sunglasses')['delta']; p['beach'] = paired(R, ok, g, 'step1', 'p_beach')['delta']; p['dino'] = float(np.mean([R[k][g]['dino'] for k in ok])); p['nan'] = int(sum(R[k][g]['n_nan'] for k in ok)); S['rows'][g] = p
    print(f"{g:15s} arcface {p['mean_a']:.3f} Δ{p['delta']:+.4f} p={p['p_t']:.3g} {p['positive']}/{p['n']} | sg Δ{p['sg']:+.3f} beach Δ{p['beach']:+.3f} dino {p['dino']:.3f} nan {p['nan']}", flush=True)
q = paired(R, ok, 'blur0_rho0.25', 'blur6_rho0.25', 'p_beach'); print("blur0 vs blur6 (rho0.25) beach:", q['delta'], q['p_t'], "| arcface:", paired(R, ok, 'blur0_rho0.25', 'blur6_rho0.25')['delta'])
json.dump(S, open(f'{OUT}/summary.json', 'w'), indent=1); print("DONE", flush=True)
