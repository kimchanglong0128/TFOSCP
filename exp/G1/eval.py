# Card G1 evaluation: DINO (ViT-S/16), CLIP-I, CLIP-T (ViT-B/32) per image; per-subject means; paired tests across subjects, objects (21) and live subjects (9) separately.
# usage: python eval.py a      (tag of the source variant; baseline images are shared)
import sys; sys.path.insert(0, '/workspace/exp/cards')
from lib_subject import *
OUT = '/workspace/exp/G1'; TAG = sys.argv[1] if len(sys.argv) > 1 else 'a'; SEEDS = list(range(42, 52)); KEYS = ['dino', 'dino_heldout', 'dino_used', 'clip_i', 'clip_i_heldout', 'clip_t']
classes = load_classes(); M = Metrics(); R = {}
for k, cls in classes.items():
    if not os.path.exists(f'{OUT}/gen/{k}/p4_{TAG}_s{SEEDS[-1]}.png'): print('missing', k); continue
    refs = load_refs(k); rd, rc = M.dino_emb(refs), M.clip_img(refs); R[k] = dict(cls=cls, live=k in LIVE, base={q: [] for q in KEYS}, meth={q: [] for q in KEYS}, per_prompt=[])
    for pi, P in enumerate(PROMPTS):
        txt = M.clip_txt([P.format(cls)]); pp = {}
        for g, name in (('base', 'base'), ('meth', TAG)):
            sc = M.score([Image.open(f'{OUT}/gen/{k}/p{pi}_{name}_s{s}.png').convert('RGB') for s in SEEDS], rd, rc, txt)
            for q in KEYS: R[k][g][q] += sc[q]
            pp[g] = {q: float(np.mean(sc[q])) for q in KEYS}
        R[k]['per_prompt'].append(pp)
    for g in ('base', 'meth'): R[k][g] = {q: float(np.mean(v)) for q, v in R[k][g].items()}
    print(f"{k:20s} {'live' if R[k]['live'] else 'obj ':4s} DINO {R[k]['base']['dino']:.3f}->{R[k]['meth']['dino']:.3f} held-out {R[k]['base']['dino_heldout']:.3f}->{R[k]['meth']['dino_heldout']:.3f} CLIP-I {R[k]['base']['clip_i']:.3f}->{R[k]['meth']['clip_i']:.3f} CLIP-T {R[k]['base']['clip_t']:.3f}->{R[k]['meth']['clip_t']:.3f}", flush=True)
json.dump(R, open(f'{OUT}/results_{TAG}.json', 'w'), indent=1); S = {}
for grp, sel in (('all', lambda r: True), ('objects', lambda r: not r['live']), ('live', lambda r: r['live'])):
    kk = [k for k in R if sel(R[k])]; S[grp] = dict(n=len(kk))
    for q in KEYS:
        p = paired([R[k]['meth'][q] for k in kk], [R[k]['base'][q] for k in kk]); S[grp][q] = p
        print(f"{grp:8s} n={len(kk):2d} {q:15s} {p['mean_b']:.4f} -> {p['mean_a']:.4f}  Δ{p['delta']:+.4f} SE {p['delta_se']:.4f} p_t {p['p_t']:.3g} p_w {p['p_wilcoxon']:.3g} {p['positive']}/{p['n']}", flush=True)
S['per_prompt_clip_t'] = [dict(prompt=P, **paired([R[k]['per_prompt'][pi]['meth']['clip_t'] for k in R], [R[k]['per_prompt'][pi]['base']['clip_t'] for k in R])) for pi, P in enumerate(PROMPTS)]
for r in S['per_prompt_clip_t']: print(f"CLIP-T  {r['prompt']:32s} Δ{r['delta']:+.4f} p_t {r['p_t']:.3g}")
json.dump(S, open(f'{OUT}/summary_{TAG}.json', 'w'), indent=1); print("DONE", flush=True)
