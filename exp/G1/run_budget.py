# Card G1 budget sweep on source (b) (subject cut-out aligned to the mean baseline box, reference-free scene background).
# Question: is there an operating point with a significant DINO gain and NO CLIP-T drop? On faces the prompt/background cost tracked the LOW-band budget (MB1/MB2).
# Configs (low angle @|f|<=0.15, mid angle @0.15-0.35); reference row lo45_mid30 = tag 'b' from run_bc.py. Baselines, seeds, boxes (gen_done_bc.json) are shared.
import sys; sys.path.insert(0, '/workspace/exp/cards')
from lib_subject import *
OUT = '/workspace/exp/G1'; SEEDS = list(range(42, 52)); CFG = {'b_lo30_mid30': (30, 30), 'b_lo15_mid30': (15, 30), 'b_lo0_mid30': (0, 30), 'b_lo0_mid45': (0, 45), 'b_lo30_mid0': (30, 0)}
classes = load_classes(); pipe = build_dmd2_ipplus(); TB = json.load(open(f'{OUT}/gen_done_bc.json')); done = json.load(open(f'{OUT}/gen_done_budget.json')) if os.path.exists(f'{OUT}/gen_done_budget.json') else {}
for k, cls in classes.items():
    if k in done: continue
    refs = load_refs(k); emb = ip_embeds(pipe, refs[0]); tb = np.array(TB[k]['b']['tb'])
    for pi, P in enumerate(PROMPTS):
        prompt = P.format(cls); bg = Image.fromarray(np.mean([np.asarray(Image.open(f'{OUT}/gen/{k}/p{pi}_base_s{s}.png').convert('RGB'), dtype=np.float32) for s in SEEDS], 0).round().astype(np.uint8))
        z = encode(pipe, build_source(refs[0], 'class_box', tb=tb, bg=bg))
        for s in SEEDS:
            e = noise_for(s)
            for tag, (tl, tm) in CFG.items():
                img = gen(pipe, emb, rotate_two_band(e, z, tl, 0.15, tm, 0.35), s, prompt); check_image(img); img.save(f'{OUT}/gen/{k}/p{pi}_{tag}_s{s}.png')
    done[k] = True; json.dump(done, open(f'{OUT}/gen_done_budget.json', 'w')); print(f"{k:20s} done", flush=True)
print("DONE", flush=True)
