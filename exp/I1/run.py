# Card I1 (analysis, no generation): what predicts the per-identity gain?
#   per identity: gain15 = B1 A_ref - step1, gain25 = R1 th45_rho0.25 - step1 (DMD2); hyper gains from B2 (th45, th75).
#   predictors: 1-step baseline ArcFace; face-box dispersion across the 20 baseline seeds (centre std, size std, normalised by 1024; detection rate);
#               reference attributes from insightface (sex, age, yaw, pitch, face-area fraction); low-freq cosine between source latent and mean baseline latent (B1 zL data not stored -> skip).
#   Also: box dispersion of DMD2-399 vs Hyper-UNet-800 vs Hyper-LoRA-999 / LCM-999 (T1) as a per-model "structure starvation" measurement.
import sys, os, json, glob; sys.path.insert(0, '/workspace/pilot'); sys.path.insert(0, '/workspace/exp/cards')
import numpy as np, cv2
from scipy import stats
import idmetrics as m
from PIL import Image
IDS = [f'id{i:02d}' for i in range(30)]; SEEDS = list(range(42, 62))
B1 = json.load(open('/workspace/exp/B1/results.json'))['results']; R1 = json.load(open('/workspace/exp/R1/results.json')); B2 = json.load(open('/workspace/exp/B2/results.json'))['results']
def boxes(pattern):
    out = []
    for s in SEEDS:
        p = pattern.format(s=s)
        if not os.path.exists(p): continue
        f = m._app.get(cv2.imread(p)); out.append(None if not f else max(f, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1])).bbox)
    return out
def dispersion(bx):
    b = np.array([x for x in bx if x is not None], dtype=float) / 1024.0
    if len(b) < 3: return dict(det=len(b) / max(1, len(bx)), c_std=np.nan, s_std=np.nan, size=np.nan)
    c = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1); sz = np.sqrt((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))
    return dict(det=len(b) / len(bx), c_std=float(np.sqrt((c.std(0) ** 2).sum())), s_std=float(sz.std()), size=float(sz.mean()))
rows = []
for k in IDS:
    ref = cv2.imread(f'/workspace/exp/AD9/ids/{k}.png'); ref = cv2.resize(ref, (1024, 1024)); f = max(m._app.get(ref), key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
    yaw, pitch = (float(f.pose[1]), float(f.pose[0])) if getattr(f, 'pose', None) is not None else (np.nan, np.nan)
    d_dmd = dispersion(boxes('/workspace/exp/B1/gen/' + k + '_step1_s{s}.png')); d_hyp = dispersion(boxes('/workspace/exp/B2/gen/' + k + '_step1_s{s}.png'))
    d_hl = dispersion(boxes('/workspace/exp/T1/gen/' + k + '_hyper_lora_999_step1_s{s}.png')); d_lcm = dispersion(boxes('/workspace/exp/T1/gen/' + k + '_lcm_lora_999_step1_s{s}.png'))
    rows.append(dict(id=k, base=B1[k]['step1']['arcface'], gain15=B1[k]['A_ref']['arcface'] - B1[k]['step1']['arcface'], gain25=R1[k]['th45_rho0.25']['arcface'] - R1[k]['step1']['arcface'],
                     gain_mean=B1[k]['B_meanface']['arcface'] - B1[k]['step1']['arcface'], hyp_base=B2[k]['step1']['arcface'], hyp_gain45=B2[k]['th45_rho0.15']['arcface'] - B2[k]['step1']['arcface'], hyp_gain75=B2[k]['th75_rho0.15']['arcface'] - B2[k]['step1']['arcface'],
                     sex=int(f.sex == 'M') if hasattr(f, 'sex') else (int(f.gender) if hasattr(f, 'gender') else -1), age=int(f.age) if hasattr(f, 'age') else -1, yaw=yaw, pitch=pitch, ref_area=float((f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]) / 1024**2),
                     dmd_cstd=d_dmd['c_std'], dmd_sstd=d_dmd['s_std'], dmd_size=d_dmd['size'], dmd_det=d_dmd['det'], hyp_cstd=d_hyp['c_std'], hyp_sstd=d_hyp['s_std'], hyp_size=d_hyp['size'], hyp_det=d_hyp['det'],
                     hl_cstd=d_hl['c_std'], hl_det=d_hl['det'], lcm_cstd=d_lcm['c_std'], lcm_det=d_lcm['det']))
    print(k, {a: (round(rows[-1][a], 3) if isinstance(rows[-1][a], float) else rows[-1][a]) for a in ('base', 'gain15', 'gain25', 'dmd_cstd', 'dmd_sstd', 'hyp_cstd', 'yaw', 'ref_area', 'sex', 'age')}, flush=True)
json.dump(rows, open('/workspace/exp/I1/rows.json', 'w'), indent=1)
def sp(a, b):
    x = np.array([r[a] for r in rows], float); y = np.array([r[b] for r in rows], float); ok = np.isfinite(x) & np.isfinite(y); r_, p_ = stats.spearmanr(x[ok], y[ok]); return f"{a:10s} vs {b:10s} rho {r_:+.2f} p {p_:.3g} (n={ok.sum()})"
print("\n== predictors of the DMD2 gain (Spearman) ==")
for tgt in ('gain15', 'gain25', 'gain_mean'):
    for pred in ('base', 'dmd_cstd', 'dmd_sstd', 'dmd_size', 'dmd_det', 'yaw', 'pitch', 'ref_area', 'age', 'sex'): print(sp(pred, tgt))
print("\n== Hyper-SD gain ==")
for tgt in ('hyp_gain45', 'hyp_gain75'):
    for pred in ('hyp_base', 'hyp_cstd', 'hyp_sstd', 'base', 'gain15'): print(sp(pred, tgt))
print("\n== per-model structure dispersion (mean over 30 ids) ==")
for mdl, cs, ss, dt in (('dmd2_399', 'dmd_cstd', 'dmd_sstd', 'dmd_det'), ('hyper_unet_800', 'hyp_cstd', 'hyp_sstd', 'hyp_det'), ('hyper_lora_999', 'hl_cstd', None, 'hl_det'), ('lcm_lora_999', 'lcm_cstd', None, 'lcm_det')):
    print(f"{mdl:15s} centre-std {np.nanmean([r[cs] for r in rows]):.4f}  size-std {np.nanmean([r[ss] for r in rows]) if ss else float('nan'):.4f}  det {np.mean([r[dt] for r in rows]):.2f}")
x = np.array([r['dmd_cstd'] for r in rows]); y = np.array([r['hyp_cstd'] for r in rows]); ok = np.isfinite(x) & np.isfinite(y); print("dmd2 vs hyper centre-std paired t p =", float(stats.ttest_rel(x[ok], y[ok]).pvalue), "wilcoxon p =", float(stats.wilcoxon(x[ok], y[ok]).pvalue))
print("DONE", flush=True)
