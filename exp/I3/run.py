# Card I3 (analysis): is the sex gap a metric/prototype effect? ArcFace (buffalo_l) embeddings of the 30 references and of B1 step1 / A_ref images.
#   within-sex reference similarity (leave-self-out prototype cosine); prototype cosine of generated images before/after the method, by sex;
#   decomposition: gain toward the reference vs gain toward the sex prototype; and gain measured with prototype component projected out.
import sys, os, json; sys.path.insert(0, '/workspace/pilot')
import numpy as np, cv2
from scipy import stats
import idmetrics as m
IDS = [f'id{i:02d}' for i in range(30)]; SEEDS = range(42, 62); sex = {r['id']: r['sex'] for r in json.load(open('/workspace/exp/I1/rows.json'))}
ref = {k: m.arc_embed(cv2.resize(cv2.imread(f'/workspace/exp/AD9/ids/{k}.png'), (1024, 1024))) for k in IDS}
proto = {k: np.mean([ref[j] for j in IDS if j != k and sex[j] == sex[k]], 0) for k in IDS}; proto = {k: v / np.linalg.norm(v) for k, v in proto.items()}
print("within-sex reference similarity (leave-self-out prototype cosine): male %.3f  female %.3f" % (np.mean([ref[k] @ proto[k] for k in IDS if sex[k] == 1]), np.mean([ref[k] @ proto[k] for k in IDS if sex[k] == 0])))
cross = {k: np.mean([ref[j] for j in IDS if sex[j] != sex[k]], 0) for k in IDS}; cross = {k: v / np.linalg.norm(v) for k, v in cross.items()}
print("cross-sex prototype cosine of references: male %.3f  female %.3f" % (np.mean([ref[k] @ cross[k] for k in IDS if sex[k] == 1]), np.mean([ref[k] @ cross[k] for k in IDS if sex[k] == 0])))
rows = {}
for k in IDS:
    E = {}
    for g, pat in (('step1', '/workspace/exp/B1/gen/' + k + '_step1_s{s}.png'), ('A_ref', '/workspace/exp/B1/gen/' + k + '_A_ref_s{s}.png')):
        embs = [m.arc_embed(cv2.imread(pat.format(s=s))) for s in SEEDS]; E[g] = [e for e in embs if e is not None]
    def stat(embs):
        c_ref = np.mean([e @ ref[k] for e in embs]); c_pro = np.mean([e @ proto[k] for e in embs])
        # projected-out: remove the prototype direction from both the generated and the reference embedding, then cosine
        def po(v): w = v - (v @ proto[k]) * proto[k]; return w / (np.linalg.norm(w) + 1e-8)
        c_po = np.mean([po(e) @ po(ref[k]) for e in embs]); return c_ref, c_pro, c_po
    s1, a = stat(E['step1']), stat(E['A_ref']); rows[k] = dict(sex=sex[k], ref_step1=float(s1[0]), ref_A=float(a[0]), pro_step1=float(s1[1]), pro_A=float(a[1]), po_step1=float(s1[2]), po_A=float(a[2]))
    print(k, 'M' if sex[k] == 1 else 'F', f"to-ref {s1[0]:.3f}->{a[0]:.3f}  to-proto {s1[1]:.3f}->{a[1]:.3f}  proto-projected-out {s1[2]:.3f}->{a[2]:.3f}", flush=True)
json.dump(rows, open('/workspace/exp/I3/rows.json', 'w'), indent=1)
for s, name in ((1, 'male'), (0, 'female')):
    kk = [k for k in IDS if sex[k] == s]
    for a_, b_, lab in (('ref_A', 'ref_step1', 'gain to reference'), ('pro_A', 'pro_step1', 'gain to sex prototype'), ('po_A', 'po_step1', 'gain, prototype projected out')):
        d = np.array([rows[k][a_] - rows[k][b_] for k in kk]); print(f"{name:6s} {lab:32s} Δ{d.mean():+.4f} {int((d>0).sum())}/{len(kk)} p={stats.ttest_1samp(d, 0).pvalue:.3g}  (level {np.mean([rows[k][b_] for k in kk]):.3f}->{np.mean([rows[k][a_] for k in kk]):.3f})")
dm = [rows[k]['po_A'] - rows[k]['po_step1'] for k in IDS if sex[k] == 1]; df = [rows[k]['po_A'] - rows[k]['po_step1'] for k in IDS if sex[k] == 0]
print("projected-out gain male vs female Mann-Whitney p =", stats.mannwhitneyu(dm, df).pvalue); print("DONE", flush=True)
