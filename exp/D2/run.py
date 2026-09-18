# Card D2: discriminate "query lacks structure" (ours) from "DMD loses the per-trajectory map" (OPAD).
# Part 1 (DMD2 1-step UNet t=399 + FaceID, 10 ids x 5 seeds, same eps as D1): inputs
#   step1        : eps
#   method       : rotate(eps -> z_ref, 45 deg)
#   teacher_xt401/601/801 : the teacher's (SDXL, DDIM-50, CFG 5) intermediate x_t at t=401/601/801, rescaled to unit global std, fed as the 1-step input
#   Prediction (ours): teacher_xt401 >= method ~ teacher final; OPAD: DMD2 stays well below the teacher even with full structure.
# Part 2 (seed consistency, post-processing): DINOv2 cosine and pixel MSE between student(eps) and teacher(eps) for LCM-LoRA (T1), Hyper-SD UNet (B2), DMD2 (this card).
import sys; sys.path.insert(0, '/workspace/exp/cards')
from lib import *
OUT = '/workspace/exp/D2'; os.makedirs(f'{OUT}/gen', exist_ok=True); IDS = [f'id{i:02d}' for i in range(10)]; SEEDS5 = SEEDS[:5]; T = [399]
ids = {k: v for k, v in load_ids().items() if k in IDS}
pipe = build_faceid("dmd2_sdxl_1step_unet_fp16.bin", False, lora=False); R = {}
for k, rimg in ids.items():
    remb = m.arc_embed(cv2.cvtColor(np.asarray(rimg), cv2.COLOR_RGB2BGR)); rdino = m.dino_embed(rimg); emb = face_embeds(pipe, rimg); R[k] = {}
    tb = f'/workspace/exp/D1/gen/{k}_s42_x0.png'; assert os.path.exists(tb), "run D1 first"
    groups = {'step1': lambda s: noise_for(s)}
    boxes = [b for b in (m.face_bbox(cv2.imread(f'/workspace/exp/B1/gen/{k}_step1_s{s}.png')) for s in SEEDS) if b is not None]
    z = encode(pipe, make_source(rimg, boxes)); groups['method'] = lambda s, z=z: rotate_lowfreq(noise_for(s), z, 45, 0.15)
    for t in (401, 601, 801):
        def f(s, t=t):
            x = torch.load(f'/workspace/exp/D1/cache/{k}_s{s}_xt{t}.pt').cuda(); return (x / x.float().std()).to(torch.float16)
        groups[f'teacher_xt{t}'] = f
    for gname, fn in groups.items():
        paths = []
        for s in SEEDS5:
            lat = fn(s); assert abs(float(lat.float().std()) - 1) < 0.05, float(lat.float().std())
            img = gen(pipe, emb, lat, s, T); q = f'{OUT}/gen/{k}_{gname}_s{s}.png'; img.save(q); check_image(img); paths.append(q)
        R[k][gname] = metrics(paths, remb, rdino, pad_recover=True); rep(k, gname, R[k][gname])
    R[k]['teacher_final'] = metrics([f'/workspace/exp/D1/gen/{k}_s{s}_x0.png' for s in SEEDS5], remb, rdino, pad_recover=True); rep(k, 'teacher_final', R[k]['teacher_final'])
    json.dump(R, open(f'{OUT}/results.json', 'w'), indent=1)
del pipe; torch.cuda.empty_cache()
# ---- part 2: seed consistency student(eps) vs teacher(eps) ----
def cons(student_path_fn):
    d, mse, n = [], [], 0
    for k in IDS:
        for s in SEEDS5:
            a, b = student_path_fn(k, s), f'/workspace/exp/D1/gen/{k}_s{s}_x0.png'
            if not os.path.exists(a): continue
            ia, ib = Image.open(a).convert('RGB'), Image.open(b).convert('RGB'); d.append(float(m.dino_embed(ia) @ m.dino_embed(ib)))
            mse.append(float(((np.asarray(ia).astype(np.float32) - np.asarray(ib).astype(np.float32)) ** 2).mean())); n += 1
    return dict(n=n, dino=float(np.mean(d)) if d else None, mse=float(np.mean(mse)) if mse else None)
C = {'dmd2_1s_399': cons(lambda k, s: f'{OUT}/gen/{k}_step1_s{s}.png'), 'hyper_unet_800': cons(lambda k, s: f'/workspace/exp/B2/gen/{k}_step1_s{s}.png'),
     'lcm_lora_999': cons(lambda k, s: f'/workspace/exp/T1/gen/{k}_lcm_lora_999_step1_s{s}.png'), 'hyper_lora_999': cons(lambda k, s: f'/workspace/exp/T1/gen/{k}_hyper_lora_999_step1_s{s}.png'),
     'teacher_vs_other_seed': cons(lambda k, s: f'/workspace/exp/D1/gen/{k}_s{42 + (s - 42 + 1) % 5}_x0.png')}
S = dict(groups={g: paired(R, IDS, g, 'step1') for g in ('method', 'teacher_xt401', 'teacher_xt601', 'teacher_xt801', 'teacher_final')}, seed_consistency=C,
         means={g: float(np.mean([R[k][g]['arcface'] for k in IDS])) for g in ('step1', 'method', 'teacher_xt401', 'teacher_xt601', 'teacher_xt801', 'teacher_final')},
         nan={g: int(sum(R[k][g]['n_nan'] for k in IDS)) for g in ('step1', 'method', 'teacher_xt401', 'teacher_xt601', 'teacher_xt801', 'teacher_final')},
         sg={g: float(np.mean([R[k][g]['p_sunglasses'] for k in IDS])) for g in ('step1', 'method', 'teacher_xt401', 'teacher_xt601', 'teacher_xt801', 'teacher_final')},
         xt401_vs_method=paired(R, IDS, 'teacher_xt401', 'method'), xt401_vs_teacher=paired(R, IDS, 'teacher_xt401', 'teacher_final'))
json.dump(S, open(f'{OUT}/summary.json', 'w'), indent=1); print(json.dumps(S, indent=1)); print("DONE", flush=True)
