# why does the background probe fall with budget? face-box size fraction of the outputs: step1 vs global rho0.25 vs windowed rho0.25 (10 ids x 20 seeds)
import sys, json; sys.path.insert(0, '/workspace/pilot'); import numpy as np, cv2, idmetrics as m
IDS = [f'id{i:02d}' for i in range(10)]; out = {}
for name, pat in (('step1', '/workspace/exp/B1/gen/{k}_step1_s{s}.png'), ('global_rho0.25', '/workspace/exp/R1/gen/{k}_th45_rho0.25_s{s}.png'), ('windowed_rho0.25', '/workspace/exp/SW1/gen/{k}_sw_th45_rho0.25_wbg0.0_s{s}.png'), ('global_rho0.15', '/workspace/exp/B1/gen/{k}_A_ref_s{s}.png')):
    fr, nd = [], 0
    for k in IDS:
        for s in range(42, 62):
            b = m.face_bbox(cv2.imread(pat.format(k=k, s=s)))
            if b is None: nd += 1; continue
            fr.append((b[2] - b[0]) * (b[3] - b[1]) / 1024 ** 2)
    out[name] = dict(box_area_frac=float(np.mean(fr)), n=len(fr), undetected=nd); print(name, out[name], flush=True)
json.dump(out, open('/workspace/exp/SW1/boxsize.json', 'w'), indent=1); print("DONE", flush=True)
