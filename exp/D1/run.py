# Card D1: teacher trajectory. SDXL base + FaceID-PlusV2, DDIM 50 steps, CFG 5, 10 FFHQ ids x 5 seeds.
# Per step t: x0_hat(t) = (x_t - sqrt(1-abar_t) eps_cfg)/sqrt(abar_t) decoded -> ArcFace vs ref (pad-recovered), face-box IoU vs final x0;
# identity-token attention maps A_id(t): for every IP cross-attn layer, softmax over the 4 FaceID tokens (decoupled branch), column-normalised
# over pixels, averaged over heads/tokens -> mass inside the final face box + spatial entropy, grouped by resolution (64x64 / 32x32).
# Also caches x_t at t in {801, 601, 401} (+ the initial noise) for card D2.
import sys; sys.path.insert(0, '/workspace/exp/cards')
from lib import *
import math
OUT = '/workspace/exp/D1'; os.makedirs(f'{OUT}/gen', exist_ok=True); os.makedirs(f'{OUT}/cache', exist_ok=True)
IDS = [f'id{i:02d}' for i in range(10)]; SEEDS5 = SEEDS[:5]; G = 5.0; NSTEP = 50; CACHE_T = (801, 601, 401)
ids = {k: v for k, v in load_ids().items() if k in IDS}
enc = CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter", subfolder="models/image_encoder", torch_dtype=torch.float16)
pipe = DiffusionPipeline.from_pretrained(BASE, image_encoder=enc, dtype=torch.float16, variant="fp16", use_safetensors=True).to('cuda')
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config); attach_faceid(pipe)
AC = pipe.scheduler.alphas_cumprod.clone()
# ---- hooks ----
TR = dict(xt=[], eps=[], t=[]); Q = {}; K = {}; MAPS = {}
def unet_pre(mod, args, kwargs):
    x = args[0] if args else kwargs['sample']; t = args[1] if len(args) > 1 else kwargs['timestep']
    TR['xt'].append(x[1:2].detach().clone()); TR['t'].append(int(t))
def unet_post(mod, args, kwargs, out):
    e = out.sample if hasattr(out, 'sample') else out[0]; u, c = e[0:1], e[1:2]; TR['eps'].append((u + G * (c - u)).detach().clone())
    # attention maps of identity tokens for this step
    step = {}
    for name in Q:
        q, k = Q[name], K[name]; B, HW, C = q.shape; h = HEADS[name]; d = C // h
        qh = q.view(B, HW, h, d).transpose(1, 2).float(); kh = k.view(B, -1, h, d).transpose(1, 2).float()
        P = torch.softmax(qh @ kh.transpose(-1, -2) / math.sqrt(d), dim=-1)          # (B,h,HW,4): decoupled branch softmax over the 4 id tokens
        col = P[1] / P[1].sum(dim=2, keepdim=True)                                   # cond half; normalise each token column over pixels
        step[name] = col.mean(dim=(0, 2)).cpu()                                        # (HW,) averaged over heads and tokens
    MAPS.setdefault('steps', []).append(step); Q.clear(); K.clear()
HEADS = {}
for name, mod in pipe.unet.named_modules():
    if name.endswith('attn2') and hasattr(mod.processor, 'to_k_ip'):
        HEADS[name] = mod.heads
        mod.to_q.register_forward_hook(lambda m_, a, o, n=name: Q.__setitem__(n, o.detach()))
        mod.processor.to_k_ip[0].register_forward_hook(lambda m_, a, o, n=name: K.__setitem__(n, o.detach()))
print("IP attn2 layers hooked:", len(HEADS), flush=True); assert len(HEADS) == 70
pipe.unet.register_forward_pre_hook(unet_pre, with_kwargs=True); pipe.unet.register_forward_hook(unet_post, with_kwargs=True)
def decode(z):
    pipe.vae.to(torch.float32)
    with torch.no_grad(): x = pipe.vae.decode(z.float() / pipe.vae.config.scaling_factor).sample
    pipe.vae.to(torch.float16); return pipe.image_processor.postprocess(x, output_type='pil')[0]
def box_of(img):
    b = m.face_bbox(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)); return None if b is None else b.astype(float)
def iou(a, b):
    if a is None or b is None: return float('nan')
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return float(i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i))
def arc_img(img, remb):
    e = m.arc_embed(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
    if e is None:
        pad = ImageOps.expand(img, border=int(0.3*1024), fill=(128,128,128)); e = m.arc_embed(cv2.cvtColor(np.asarray(pad), cv2.COLOR_RGB2BGR))
    return float('nan') if e is None else float(e @ remb)
def map_stats(mp, box):
    n = int(round(math.sqrt(mp.numel()))); M = mp.view(n, n); x1, y1, x2, y2 = [v / 1024 * n for v in box]
    mass = float(M[int(y1):int(math.ceil(y2)), int(x1):int(math.ceil(x2))].sum()); p = M.flatten().clamp_min(1e-12); ent = float(-(p * p.log()).sum() / math.log(p.numel()))
    return mass, ent, (x2-x1)*(y2-y1)/(n*n)
R = {}
for k, rimg in ids.items():
    remb = m.arc_embed(cv2.cvtColor(np.asarray(rimg), cv2.COLOR_RGB2BGR)); emb = face_embeds(pipe, rimg, do_cfg=True); R[k] = {}
    for s in SEEDS5:
        TR['xt'].clear(); TR['eps'].clear(); TR['t'].clear(); MAPS.clear()
        g = torch.Generator('cuda').manual_seed(s)
        img = pipe(prompt=PROMPT, num_inference_steps=NSTEP, guidance_scale=G, ip_adapter_image_embeds=emb, latents=noise_for(s), generator=g).images[0]
        img.save(f'{OUT}/gen/{k}_s{s}_x0.png'); assert len(TR['t']) == NSTEP, len(TR['t']); fb = box_of(img); a_final = arc_img(img, remb)
        rows = []
        for i, t in enumerate(TR['t']):
            a = float(AC[t]); x0h = (TR['xt'][i] - (1 - a) ** 0.5 * TR['eps'][i]) / a ** 0.5; im = decode(x0h)
            if t in CACHE_T: torch.save(TR['xt'][i].cpu(), f'{OUT}/cache/{k}_s{s}_xt{t}.pt')
            if i % 10 == 0 or t in CACHE_T: im.save(f'{OUT}/gen/{k}_s{s}_x0hat_t{t}.png')
            b = box_of(im); row = dict(t=t, arcface=arc_img(im, remb), iou=iou(b, fb) if fb is not None else float('nan'))
            if fb is not None:
                st = {'64': [], '32': []}
                for name, mp in MAPS['steps'][i].items(): st['64' if mp.numel() == 4096 else '32'].append(map_stats(mp, fb))
                for r_, v in st.items():
                    if v: row[f'mass{r_}'] = float(np.mean([x[0] for x in v])); row[f'ent{r_}'] = float(np.mean([x[1] for x in v])); row['box_frac'] = v[0][2]
            rows.append(row)
        torch.save(noise_for(s).cpu(), f'{OUT}/cache/{k}_s{s}_eps.pt')
        R[k][str(s)] = dict(final_arcface=a_final, final_box=None if fb is None else fb.tolist(), rows=rows)
        print(k, s, f"final arcface {a_final:+.3f}", "arc@t:", {r['t']: round(r['arcface'], 2) for r in rows[::10]}, "iou@t:", {r['t']: round(r['iou'], 2) for r in rows[::10]}, flush=True)
        json.dump(R, open(f'{OUT}/results.json', 'w'), indent=1)
# ---- aggregate per t ----
ts = [r['t'] for r in next(iter(next(iter(R.values())).values()))['rows']]; agg = []
for i, t in enumerate(ts):
    rr = [R[k][s]['rows'][i] for k in R for s in R[k]]
    agg.append(dict(t=t, arcface=float(np.nanmean([r['arcface'] for r in rr])), iou=float(np.nanmean([r['iou'] for r in rr])), iou_gt05=float(np.nanmean([r['iou'] > 0.5 for r in rr if not np.isnan(r['iou'])])),
                    mass64=float(np.nanmean([r.get('mass64', np.nan) for r in rr])), mass32=float(np.nanmean([r.get('mass32', np.nan) for r in rr])), ent64=float(np.nanmean([r.get('ent64', np.nan) for r in rr])), ent32=float(np.nanmean([r.get('ent32', np.nan) for r in rr])),
                    box_frac=float(np.nanmean([r.get('box_frac', np.nan) for r in rr]))))
final = float(np.nanmean([R[k][s]['final_arcface'] for k in R for s in R[k]]))
json.dump(dict(final_arcface=final, per_t=agg), open(f'{OUT}/summary.json', 'w'), indent=1)
for a in agg[::5]: print(f"t={a['t']:4d} arcface {a['arcface']:.3f} iou {a['iou']:.2f} iou>0.5 {a['iou_gt05']:.2f} mass64 {a['mass64']:.2f} mass32 {a['mass32']:.2f} (box frac {a['box_frac']:.2f}) ent64 {a['ent64']:.2f}", flush=True)
print("final teacher arcface", round(final, 3)); print("DONE", flush=True)
