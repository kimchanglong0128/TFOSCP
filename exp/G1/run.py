# Card G1, source (a): is the mechanism face-specific? DreamBench 30 subjects x 5 prompts x 10 seeds, DMD2 1-step (t=399) + IP-Adapter Plus SDXL ViT-H (scale 0.6).
#   baseline : eps
#   method   : two-band rotation (low 45 deg @ 0.15, mid 30 deg @ 0.15-0.35) toward the VAE latent of the reference itself (subject_mode='ref_direct': no alignment, no gray)
#   control  : theta=0 for both bands must reproduce the baseline pixel for pixel (asserted per subject on the first prompt/seed)
# Reference = first image (00.jpg) for both the adapter and the source; the other real images are held out for the copy-proof metrics. Generation only; metrics in eval.py.
import sys; sys.path.insert(0, '/workspace/exp/cards')
from lib_subject import *
OUT = '/workspace/exp/G1'; MODE = 'ref_direct'; SEEDS = list(range(42, 52)); TAG = 'a'
classes = load_classes(); pipe = build_dmd2_ipplus(); done = json.load(open(f'{OUT}/gen_done_{TAG}.json')) if os.path.exists(f'{OUT}/gen_done_{TAG}.json') else {}
for k, cls in classes.items():
    if k in done: continue
    os.makedirs(f'{OUT}/gen/{k}', exist_ok=True); refs = load_refs(k); emb = ip_embeds(pipe, refs[0]); src = build_source(refs[0], MODE); z = encode(pipe, src)
    if k in ('dog', 'teapot'): src.save(f'{OUT}/gen/{k}/src_{TAG}.png')
    for pi, P in enumerate(PROMPTS):
        prompt = P.format(cls)
        for s in SEEDS:
            e = noise_for(s); b = gen(pipe, emb, e, s, prompt); check_image(b); b.save(f'{OUT}/gen/{k}/p{pi}_base_s{s}.png')
            if pi == 0 and s == SEEDS[0]:
                e0 = rotate_two_band(e, z, 0, 0.15, 0, 0.35); c = gen(pipe, emb, e0, s, prompt); diff = int(np.abs(np.asarray(c).astype(int) - np.asarray(b).astype(int)).max())
                assert diff == 0, f"theta=0 control differs from baseline by {diff} for {k}"; rel = float((e0.float() - e.float()).norm() / e.float().norm())
            mimg = gen(pipe, emb, rotate_two_band(e, z), s, prompt); check_image(mimg); mimg.save(f'{OUT}/gen/{k}/p{pi}_{TAG}_s{s}.png')
    done[k] = dict(cls=cls, n_refs=len(refs), theta0_pix_diff=diff, theta0_latent_rel=rel); json.dump(done, open(f'{OUT}/gen_done_{TAG}.json', 'w'), indent=1)
    print(f"{k:20s} {cls:15s} refs {len(refs)} theta0 pix diff {diff} latent rel {rel:.1e}", flush=True)
print("DONE", flush=True)
