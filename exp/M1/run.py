# Card M1: multi-prompt. DMD2 1-step + FaceID, 30 FFHQ ids x 10 seeds, 5 prompts with prompt->conflict-region lookup and an offline text-generated background per prompt.
# Verdict: >=4/5 prompts with paired ArcFace p<0.05 and attribute probe not dropping.
import sys; sys.path.insert(0, '/workspace/exp/cards')
from lib import *
import lib
OUT = '/workspace/exp/M1'; os.makedirs(f'{OUT}/gen', exist_ok=True); T = [399]; SEEDS10 = SEEDS[:10]
PROMPTS = {
 'sunglasses_beach': dict(prompt=f"a close-up portrait photo of a {SUBJ} wearing sunglasses, on a beach", region='eyes', bg="a photo of a beach, no people", attr=[f"a photo of a {SUBJ} wearing dark sunglasses", f"a photo of a {SUBJ} without glasses"], scene=[f"a photo of a {SUBJ} on a beach", f"a photo of a {SUBJ} in front of a plain wall"]),
 'hat_street':       dict(prompt=f"a close-up portrait photo of a {SUBJ} wearing a red hat, on a city street", region='top', bg="a photo of a city street, no people", attr=[f"a photo of a {SUBJ} wearing a red hat", f"a photo of a {SUBJ} without a hat"], scene=[f"a photo of a {SUBJ} on a city street", f"a photo of a {SUBJ} in front of a plain wall"]),
 'beard_library':    dict(prompt=f"a close-up portrait photo of a {SUBJ} with a full beard, in a library", region='chin', bg="a photo of a library interior, no people", attr=[f"a photo of a {SUBJ} with a full beard", f"a photo of a clean-shaven {SUBJ}"], scene=[f"a photo of a {SUBJ} in a library", f"a photo of a {SUBJ} in front of a plain wall"]),
 'smile_snow':       dict(prompt=f"a close-up portrait photo of a {SUBJ} smiling broadly, in snowy mountains", region='mouth', bg="a photo of snowy mountains, no people", attr=[f"a photo of a {SUBJ} smiling broadly", f"a photo of a {SUBJ} with a neutral expression"], scene=[f"a photo of a {SUBJ} in snowy mountains", f"a photo of a {SUBJ} in front of a plain wall"]),
 'neutral_wall':     dict(prompt=f"a close-up portrait photo of a {SUBJ}, in front of a plain gray wall", region=None, bg="a photo of a plain gray wall", attr=None, scene=[f"a photo of a {SUBJ} in front of a plain wall", f"a photo of a {SUBJ} on a beach"]),
}
ids = load_ids(); keys = list(ids)
# offline backgrounds with the 4-step DMD2 UNet, text only (as AD6 did for the beach)
os.makedirs(f'{OUT}/bg', exist_ok=True)
if not all(os.path.exists(f'{OUT}/bg/{n}.png') for n in PROMPTS):
    p4 = build_faceid("dmd2_sdxl_4step_unet_fp16.safetensors", True, lora=False); p4.set_ip_adapter_scale(0.0)
    zero = [torch.zeros(1, 1, 512, device='cuda', dtype=torch.float16)]; layer = p4.unet.encoder_hid_proj.image_projection_layers[0]; layer.clip_embeds = torch.zeros(1, 257, 1280, device='cuda', dtype=torch.float16); layer.shortcut = True
    for n, P in PROMPTS.items():
        g = torch.Generator('cuda').manual_seed(7); p4(prompt=P['bg'], negative_prompt="person, face, people", timesteps=[999, 749, 499, 249], guidance_scale=0, ip_adapter_image_embeds=zero, generator=g).images[0].save(f'{OUT}/bg/{n}.png')
    del p4; torch.cuda.empty_cache()
pipe = build_faceid("dmd2_sdxl_1step_unet_fp16.bin", False, lora=False)
def probs2(paths, caps): return float(clip_probs(paths, caps)[0])
R = json.load(open(f'{OUT}/results.json')) if os.path.exists(f'{OUT}/results.json') else {}
for pname, P in PROMPTS.items():
    bg = Image.open(f'{OUT}/bg/{pname}.png').convert('RGB').filter(ImageFilter.GaussianBlur(6)); R.setdefault(pname, {})
    for k, rimg in ids.items():
        if k in R[pname]: continue
        remb = m.arc_embed(cv2.cvtColor(np.asarray(rimg), cv2.COLOR_RGB2BGR)); rdino = m.dino_embed(rimg); emb = face_embeds(pipe, rimg); row = {}
        paths = []
        for s in SEEDS10:
            img = gen(pipe, emb, noise_for(s), s, T, prompt=P['prompt']); q = f'{OUT}/gen/{pname}_{k}_step1_s{s}.png'; img.save(q); check_image(img); paths.append(q)
        def met(paths):
            a = np.array([arc_of(p, remb, True) for p in paths]); d = [float(m.dino_embed(Image.open(p).convert('RGB')) @ rdino) for p in paths]
            return dict(arcface=float(np.nanmean(a)), n_nan=int(np.isnan(a).sum()), dino=float(np.mean(d)), p_attr=probs2(paths, P['attr']) if P['attr'] else None, p_scene=probs2(paths, P['scene']))
        row['step1'] = met(paths); boxes = [b for b in (m.face_bbox(cv2.imread(q)) for q in paths) if b is not None]
        if len(boxes) < 3: row['method'] = None; R[pname][k] = row; print(pname, k, "few faces", flush=True); continue
        tb = np.array(boxes).mean(0); src = paste_aligned(rimg, ref_box(rimg), tb, bg=bg)
        if P['region']: src = gray_region(src, tb, P['region'])
        if k == 'id00': src.save(f'{OUT}/gen/{pname}_{k}_src.png')
        z = encode(pipe, src); paths = []
        for s in SEEDS10:
            img = gen(pipe, emb, rotate_lowfreq(noise_for(s), z, 45, 0.15), s, T, prompt=P['prompt']); q = f'{OUT}/gen/{pname}_{k}_method_s{s}.png'; img.save(q); check_image(img); paths.append(q)
        row['method'] = met(paths); R[pname][k] = row
        print(f"{pname} {k} step1 {row['step1']['arcface']:.3f} attr {row['step1']['p_attr']} scene {row['step1']['p_scene']:.2f} | method {row['method']['arcface']:.3f} attr {row['method']['p_attr']} scene {row['method']['p_scene']:.2f} dino {row['method']['dino']:.2f}", flush=True)
        json.dump(R, open(f'{OUT}/results.json', 'w'), indent=1)
S = {}
for pname in PROMPTS:
    ok = [k for k in keys if R[pname].get(k, {}).get('method')]; x = np.array([R[pname][k]['method']['arcface'] for k in ok]); y = np.array([R[pname][k]['step1']['arcface'] for k in ok])
    S[pname] = dict(n=len(ok), step1=float(y.mean()), method=float(x.mean()), delta=float((x - y).mean()), positive=int(((x - y) > 0).sum()), p_t=float(stats.ttest_rel(x, y).pvalue),
                    attr_step1=None if PROMPTS[pname]['attr'] is None else float(np.mean([R[pname][k]['step1']['p_attr'] for k in ok])), attr_method=None if PROMPTS[pname]['attr'] is None else float(np.mean([R[pname][k]['method']['p_attr'] for k in ok])),
                    scene_step1=float(np.mean([R[pname][k]['step1']['p_scene'] for k in ok])), scene_method=float(np.mean([R[pname][k]['method']['p_scene'] for k in ok])),
                    dino_step1=float(np.mean([R[pname][k]['step1']['dino'] for k in ok])), dino_method=float(np.mean([R[pname][k]['method']['dino'] for k in ok])), nan=int(sum(R[pname][k]['method']['n_nan'] + R[pname][k]['step1']['n_nan'] for k in ok)))
    print(f"{pname:18s} n={S[pname]['n']} step1 {S[pname]['step1']:.3f} method {S[pname]['method']:.3f} Δ{S[pname]['delta']:+.3f} p={S[pname]['p_t']:.3g} {S[pname]['positive']}/{S[pname]['n']} attr {S[pname]['attr_step1']}->{S[pname]['attr_method']} scene {S[pname]['scene_step1']:.2f}->{S[pname]['scene_method']:.2f} dino {S[pname]['dino_step1']:.2f}->{S[pname]['dino_method']:.2f}", flush=True)
json.dump(S, open(f'{OUT}/summary.json', 'w'), indent=1); print("DONE", flush=True)
