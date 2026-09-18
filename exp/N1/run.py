# Card N1: head-to-head with MoNO (same low-frequency sphere geometry, different objective and NFE). DMD2 1-step (t=399) + FaceID, 10 FFHQ ids.
# Differentiable path: manual 1-step UNet forward (verified against the pipeline image) -> TAESD-XL decode -> reward network.
#   G1 mono_div_K5   : MoNO's own objective (diversity: minimise pairwise DINOv2 cosine among 4 seeds of the same id), 5 geodesic steps of 6 deg on the low-freq sphere, seeds 42-45. NFE = 6 per sample
#   G2 id_opt_K{1,4,10}: MoNO-style iterative optimisation with an identity reward (facenet-VGGFace2 cosine on the mean face box; buffalo_l used only for evaluation), K steps of 6 deg. NFE = K+1
#   G3 rot45 (ours, closed form, 1 NFE) at rho in {0.05,0.10,0.15,0.25}; rho=0.15 seeds 42-51 reused from B1.
# Verdict rule: if rot45 reaches >= 80% of id_opt_K4's gain at 1/5 of the NFE, the closed-form reference direction is a one-shot shortcut of the iterative optimum.
import sys; sys.path.insert(0, '/workspace/exp/cards')
from lib import *
from common import abar
from diffusers import AutoencoderTiny
from facenet_pytorch import InceptionResnetV1
import torch.nn.functional as F
OUT = '/workspace/exp/N1'; os.makedirs(f'{OUT}/gen', exist_ok=True); IDS = [f'id{i:02d}' for i in range(10)]; SEEDS10 = SEEDS[:10]; T = 399; RHO = 0.15; STEP_DEG = 6.0
ids = {k: v for k, v in load_ids().items() if k in IDS}; B1 = json.load(open('/workspace/exp/B1/results.json'))['results']
pipe = build_faceid("dmd2_sdxl_1step_unet_fp16.bin", False, lora=False); pipe.unet.requires_grad_(False)
taesd = AutoencoderTiny.from_pretrained('madebyollin/taesdxl', torch_dtype=torch.float16).to('cuda').eval(); taesd.requires_grad_(False)
fnet = InceptionResnetV1(pretrained='vggface2').cuda().eval(); fnet.requires_grad_(False)
A = abar(pipe, T); c_skip, c_out = pipe.scheduler.get_scalings_for_boundary_condition_discrete(T)
pe, _, ppe, _ = pipe.encode_prompt(PROMPT, device='cuda', num_images_per_prompt=1, do_classifier_free_guidance=False)
time_ids = pipe._get_add_time_ids((1024, 1024), (0, 0), (1024, 1024), torch.float16, text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim).to('cuda')
def forward_x0(lat, emb):
    """manual DMD2 1-step: returns denoised latent (differentiable wrt lat)."""
    ie = pipe.prepare_ip_adapter_image_embeds(None, emb, 'cuda', 1, False)
    eps = pipe.unet(lat, torch.tensor([T], device='cuda'), encoder_hidden_states=pe, added_cond_kwargs=dict(text_embeds=ppe, time_ids=time_ids, image_embeds=ie)).sample
    x0 = (lat - (1 - A) ** 0.5 * eps) / A ** 0.5; return c_out * x0 + c_skip * lat
def decode_tiny(x0): return taesd.decode(x0.to(torch.float16)).sample.clamp(-1, 1)          # (1,3,1024,1024) in [-1,1]
def facenet_emb(img_m11, box):
    x1, y1, x2, y2 = [int(v) for v in box]; crop = img_m11[:, :, max(0, y1):y2, max(0, x1):x2]; crop = F.interpolate(crop.float(), size=(160, 160), mode='bilinear', align_corners=False)
    return F.normalize(fnet(crop * 0.5 * 255 / 128), dim=-1)   # facenet expects (x-127.5)/128 with x in [0,255]: img_m11*127.5/128
def dino_feat(img_m11):
    x = F.interpolate(img_m11.float(), size=(224, 224), mode='bilinear', align_corners=False); x = (x * 0.5 + 0.5 - torch.tensor([0.485, 0.456, 0.406], device='cuda')[None, :, None, None]) / torch.tensor([0.229, 0.224, 0.225], device='cuda')[None, :, None, None]
    return F.normalize(m._dino(pixel_values=x).last_hidden_state[:, 0], dim=-1)
def geo_step(lat, grad, deg, rho=RHO):
    """rotate the low band of lat toward grad's low band by `deg` on the fixed-norm sphere (tangent projection); high band untouched."""
    eL, eH = split(lat, rho); gL, _ = split(grad, rho); r = eL.norm(); ehat = eL / r
    gt = gL - (gL * ehat).sum() * ehat; gt = gt / gt.norm().clamp_min(1e-12); th = math.radians(deg)
    out = eH + r * (math.cos(th) * ehat + math.sin(th) * gt); assert abs(float(out.norm()) - float(lat.float().norm())) / float(lat.float().norm()) < 1e-3; return out.to(lat.dtype)
def to_pil(x0):
    pipe.vae.to(torch.float32)
    with torch.no_grad(): im = pipe.vae.decode(x0.float() / pipe.vae.config.scaling_factor).sample
    pipe.vae.to(torch.float16); return pipe.image_processor.postprocess(im, output_type='pil')[0]
import math
# ---- verification: manual forward == pipeline ----
emb0 = face_embeds(pipe, ids['id00'])
with torch.no_grad(): x0 = forward_x0(noise_for(42), emb0)
ref_img = gen(pipe, emb0, noise_for(42), 42, [T]); diff = int(np.abs(np.asarray(to_pil(x0)).astype(int) - np.asarray(ref_img).astype(int)).max()); print("manual-vs-pipeline max pixel diff", diff, flush=True); assert diff <= 3, diff
R = {}
for k, rimg in ids.items():
    remb = m.arc_embed(cv2.cvtColor(np.asarray(rimg), cv2.COLOR_RGB2BGR)); rdino = m.dino_embed(rimg); emb = face_embeds(pipe, rimg); R[k] = {}
    boxes = [b for b in (m.face_bbox(cv2.imread(f'/workspace/exp/B1/gen/{k}_step1_s{s}.png')) for s in SEEDS) if b is not None]; tb = np.array(boxes).mean(0)
    rb = ref_box(rimg); rt = torch.from_numpy(np.asarray(rimg, dtype=np.float32) / 127.5 - 1).permute(2, 0, 1)[None].cuda()
    with torch.no_grad(): ref_f = facenet_emb(rt, rb)
    src = make_source(rimg, boxes); z = encode(pipe, src)
    # G3: closed-form rotation at several rho (rho=0.15 reused from B1)
    for rho in (0.05, 0.10, 0.25):
        paths = []
        for s in SEEDS10:
            img = gen(pipe, emb, rotate_lowfreq(noise_for(s), z, 45, rho), s, [T]); q = f'{OUT}/gen/{k}_rot45_rho{rho:.2f}_s{s}.png'; img.save(q); paths.append(q)
        R[k][f'rot45_rho{rho:.2f}'] = metrics(paths, remb, rdino, pad_recover=True); rep(k, f'rot45_rho{rho:.2f}', R[k][f'rot45_rho{rho:.2f}'])
    R[k]['rot45_rho0.15'] = dict(arcface=float(np.nanmean(B1[k]['A_ref']['per_seed'][:10])), per_seed=B1[k]['A_ref']['per_seed'][:10]); R[k]['step1'] = dict(arcface=float(np.nanmean(B1[k]['step1']['per_seed'][:10])), per_seed=B1[k]['step1']['per_seed'][:10])
    # G2: identity-reward optimisation, K in {1,4,10}
    for K in (1, 4, 10):
        paths, rew = [], []
        for s in SEEDS10:
            lat = noise_for(s)
            for it in range(K):
                lat = lat.detach().requires_grad_(True); x0 = forward_x0(lat, emb); r_ = (facenet_emb(decode_tiny(x0), tb) * ref_f).sum(); g_ = torch.autograd.grad(r_, lat)[0]
                lat = geo_step(lat.detach(), g_.float(), STEP_DEG); rew.append(float(r_))
            img = gen(pipe, emb, lat, s, [T]); q = f'{OUT}/gen/{k}_id_opt_K{K}_s{s}.png'; img.save(q); paths.append(q)
        R[k][f'id_opt_K{K}'] = metrics(paths, remb, rdino, pad_recover=True); R[k][f'id_opt_K{K}']['facenet_reward_first'] = float(np.mean(rew[::K])); rep(k, f'id_opt_K{K}', R[k][f'id_opt_K{K}']); torch.cuda.empty_cache()
    # G1: MoNO diversity objective on 4 seeds (alternating updates, 5 steps)
    lats = {s: noise_for(s) for s in SEEDS[:4]}; feats = {}
    with torch.no_grad():
        for s in SEEDS[:4]: feats[s] = dino_feat(decode_tiny(forward_x0(lats[s], emb))).detach()
    for it in range(5):
        for s in SEEDS[:4]:
            lat = lats[s].detach().requires_grad_(True); f_ = dino_feat(decode_tiny(forward_x0(lat, emb)))
            loss = sum((f_ * feats[o]).sum() for o in SEEDS[:4] if o != s); g_ = torch.autograd.grad(loss, lat)[0]
            lats[s] = geo_step(lat.detach(), -g_.float(), STEP_DEG); feats[s] = f_.detach()
    paths = []
    for s in SEEDS[:4]:
        img = gen(pipe, emb, lats[s], s, [T]); q = f'{OUT}/gen/{k}_mono_div_s{s}.png'; img.save(q); paths.append(q)
    R[k]['mono_div_K5'] = metrics(paths, remb, rdino, pad_recover=True); R[k]['mono_div_K5']['step1_same4'] = float(np.nanmean(B1[k]['step1']['per_seed'][:4])); rep(k, 'mono_div_K5', R[k]['mono_div_K5'])
    json.dump(R, open(f'{OUT}/results.json', 'w'), indent=1); torch.cuda.empty_cache()
NFE = {'step1': 1, 'rot45_rho0.05': 1, 'rot45_rho0.10': 1, 'rot45_rho0.15': 1, 'rot45_rho0.25': 1, 'id_opt_K1': 2, 'id_opt_K4': 5, 'id_opt_K10': 11, 'mono_div_K5': 6}
S = dict(n=len(IDS), rows={})
for g in NFE:
    x = np.array([R[k][g]['arcface'] for k in IDS]); y = np.array([R[k]['step1']['arcface'] for k in IDS]) if g != 'mono_div_K5' else np.array([R[k][g]['step1_same4'] for k in IDS])
    S['rows'][g] = dict(nfe=NFE[g], arcface=float(x.mean()), delta=float((x - y).mean()), positive=int(((x - y) > 0).sum()), p_t=float(stats.ttest_rel(x, y).pvalue) if g != 'step1' else None)
    print(f"{g:14s} NFE {NFE[g]:2d} arcface {x.mean():.3f} Δ{(x - y).mean():+.3f} {int(((x - y) > 0).sum())}/{len(IDS)} p={S['rows'][g]['p_t']}", flush=True)
S['shortcut_ratio'] = S['rows']['rot45_rho0.15']['delta'] / S['rows']['id_opt_K4']['delta'] if S['rows']['id_opt_K4']['delta'] > 0 else None
json.dump(S, open(f'{OUT}/summary.json', 'w'), indent=1); print("shortcut ratio (rot45 gain / id_opt_K4 gain):", S['shortcut_ratio']); print("DONE", flush=True)
