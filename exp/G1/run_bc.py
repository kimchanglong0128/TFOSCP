# Card G1, sources (b) and (c). Baseline images are shared with run.py (same seeds / eps), so only the method images are generated.
#   (b) class_box : reference subject cut out by U^2-Net saliency, scaled+shifted onto the mean subject box of that subject's 1-step baselines (25 images)
#   (c) ellipse   : class-agnostic gray centre ellipse, nothing from the reference
#   background of both sources, per (subject, prompt): the pixel mean of the 10 baseline images of that prompt — the model's own low-frequency rendering of the prompt scene,
#   the analogue of the text-generated beach background in the face protocol, and it contains nothing from the reference scene.
# theta=0 control is asserted per subject and source against the saved baseline image.
import sys; sys.path.insert(0, '/workspace/exp/cards')
from lib_subject import *
OUT = '/workspace/exp/G1'; SEEDS = list(range(42, 52)); TAGS = sys.argv[1:] or ['b', 'c']; MODE = dict(b='class_box', c='ellipse')
classes = load_classes(); pipe = build_dmd2_ipplus(); done = json.load(open(f'{OUT}/gen_done_bc.json')) if os.path.exists(f'{OUT}/gen_done_bc.json') else {}
for k, cls in classes.items():
    if all(t in done.get(k, {}) for t in TAGS): continue
    refs = load_refs(k); emb = ip_embeds(pipe, refs[0]); done.setdefault(k, {})
    bx = [mask_box(subject_mask(Image.open(f'{OUT}/gen/{k}/p{pi}_base_s{s}.png').convert('RGB'))) for pi in range(5) for s in SEEDS[:5]]; bx = [b for b in bx if b is not None]
    tb = np.array(bx).mean(0) if len(bx) >= 5 else np.array([256., 256., 768., 768.])
    for tag in TAGS:
        if tag in done[k]: continue
        for pi, P in enumerate(PROMPTS):
            prompt = P.format(cls); bg = Image.fromarray(np.mean([np.asarray(Image.open(f'{OUT}/gen/{k}/p{pi}_base_s{s}.png').convert('RGB'), dtype=np.float32) for s in SEEDS], 0).round().astype(np.uint8))
            src = build_source(refs[0], MODE[tag], tb=tb, bg=bg); z = encode(pipe, src)
            if pi in (0, 2): src.save(f'{OUT}/gen/{k}/src_{tag}_p{pi}.png')
            if pi == 0:
                e = noise_for(SEEDS[0]); c = gen(pipe, emb, rotate_two_band(e, z, 0, 0.15, 0, 0.35), SEEDS[0], prompt)
                diff = int(np.abs(np.asarray(c).astype(int) - np.asarray(Image.open(f'{OUT}/gen/{k}/p0_base_s{SEEDS[0]}.png').convert('RGB')).astype(int)).max()); assert diff == 0, f"theta=0 differs by {diff} for {k}/{tag}"
            for s in SEEDS:
                img = gen(pipe, emb, rotate_two_band(noise_for(s), z), s, prompt); check_image(img); img.save(f'{OUT}/gen/{k}/p{pi}_{tag}_s{s}.png')
        done[k][tag] = dict(theta0_pix_diff=diff, n_boxes=len(bx), tb=[float(v) for v in tb]); json.dump(done, open(f'{OUT}/gen_done_bc.json', 'w'), indent=1)
        print(f"{k:20s} {tag} boxes {len(bx)}/25 tb {tb.round().tolist()} theta0 pix diff {diff}", flush=True)
print("DONE", flush=True)
