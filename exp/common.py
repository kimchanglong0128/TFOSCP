# Shared harness for the scheme loop (G, H, J, ...). Keeps each run.py to the scheme itself.
import sys, os, json, time; sys.path.insert(0, '/workspace/pilot')
import torch, numpy as np, cv2
from PIL import Image
from diffusers import DiffusionPipeline, LCMScheduler
from diffusers.utils.torch_utils import randn_tensor
import idmetrics as m

REF_PATH = '/workspace/pilot/refs/person.jpeg'
REF = Image.open(REF_PATH).convert('RGB')
PROMPT = "a photo of a woman wearing sunglasses, on a beach"
SEEDS = [42, 43, 44]
REQ = dict(arcface_min=0.069, dino_max=0.60)        # 2-step level re-estimated on 10 seeds the archive way (was 0.111 on 3 seeds)
LATENT_SHAPE = (1, 4, 128, 128)

ADAPTERS = {
    'base': dict(weight_name='ip-adapter_sdxl.bin', image_encoder_folder='sdxl_models/image_encoder'),
    'face': dict(weight_name='ip-adapter-plus-face_sdxl_vit-h.safetensors', image_encoder_folder='models/image_encoder'),
}
def load_pipe(scale=0.6, adapter='base'):
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                             dtype=torch.float16, variant="fp16", use_safetensors=True).to('cuda')
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
    a = ADAPTERS[adapter]
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder='sdxl_models', weight_name=a['weight_name'], image_encoder_folder=a['image_encoder_folder'])
    pipe.set_ip_adapter_scale(scale)
    pipe.set_progress_bar_config(disable=True)
    return pipe

_AC = None
def abar(pipe, t):
    """alphas_cumprod[t] as a python float, read from a CPU copy so scheduler state is never touched."""
    global _AC
    if _AC is None: _AC = pipe.scheduler.alphas_cumprod.detach().cpu().clone()
    return float(_AC[t])

def anchor(pipe, z0, eps, t):
    """x_t = sqrt(abar_t) z0 + sqrt(1-abar_t) eps, without scheduler.add_noise (which moves alphas_cumprod to cuda)."""
    a = abar(pipe, t)
    return (a ** 0.5) * z0.to(eps.dtype) + ((1 - a) ** 0.5) * eps

def noise_for(seed):
    g = torch.Generator('cuda').manual_seed(seed)
    return randn_tensor(LATENT_SHAPE, generator=g, device='cuda', dtype=torch.float16)

def generate(pipe, latents, seed, timesteps=None, prompt=None, **kw):
    g = torch.Generator('cuda').manual_seed(seed)
    tk = dict(timesteps=timesteps) if timesteps else dict(num_inference_steps=1)
    return pipe(prompt=prompt or PROMPT, guidance_scale=1, ip_adapter_image=REF, latents=latents, generator=g, **tk, **kw).images[0]

def check_image(img):
    a = np.asarray(img)
    assert img.size == (1024, 1024), f"bad size {img.size}"
    assert a.dtype == np.uint8 and np.isfinite(a).all(), "non-finite pixels"
    assert a.std() > 5, "flat image (std<5)"
    return a

def score(path):
    d, a = m.score(path, 'woman')
    assert -1.0 <= d <= 1.0, f"dino out of range {d}"
    assert np.isnan(a) or -1.0 <= a <= 1.0, f"arcface out of range {a}"
    return d, a

def vram_check():
    used = torch.cuda.max_memory_allocated() / 2**30
    assert used < 23.0, f"VRAM {used:.1f} GB too close to 24 GB"
    return round(used, 2)

def run_scheme(name, pipe, make_latents, outdir, timesteps=None, extra_kw=None, note="", prompt=None, archive_check=True, base_kw=None, setup_base=None, setup_scheme=None):
    """make_latents(seed) -> latents. Runs scheme + same-seed baseline, writes results.json, returns dict."""
    os.makedirs(outdir, exist_ok=True); extra_kw = extra_kw or {}
    assert pipe.scheduler.alphas_cumprod.device.type == 'cpu', 'scheduler state mutated (alphas_cumprod on GPU): do not call scheduler.add_noise'
    rows = {'baseline': [], name: []}
    for seed in SEEDS:
        if setup_base: setup_base()
        base_img = generate(pipe, noise_for(seed), seed, prompt=prompt, **(base_kw or {}))
        bp = f'{outdir}/baseline_seed{seed}.png'; base_img.save(bp); check_image(base_img)
        # reproducibility guard: baseline must match the archived 1-step image bit-for-bit
        if archive_check:
            ref = np.asarray(Image.open(f'/workspace/outputs/ipa_seeds/ref_woman/woman_step1_seed{seed}.png'))
            assert np.array_equal(np.asarray(base_img), ref), f"baseline drift at seed {seed}"
        if setup_scheme: setup_scheme()
        lat = make_latents(seed)
        assert torch.isfinite(lat).all(), "NaN/Inf in latents"
        img = generate(pipe, lat, seed, timesteps=timesteps, prompt=prompt, **extra_kw)
        p = f'{outdir}/{name}_seed{seed}.png'; img.save(p); check_image(img)
        rows['baseline'].append(dict(seed=seed, path=bp, **dict(zip(('dino', 'arcface'), score(bp)))))
        rows[name].append(dict(seed=seed, path=p, **dict(zip(('dino', 'arcface'), score(p)))))
    def agg(rs, k):
        v = np.array([r[k] for r in rs], dtype=float); return dict(mean=float(np.nanmean(v)), std=float(np.nanstd(v, ddof=1)), n_nan=int(np.isnan(v).sum()))
    summary = {k: {'dino': agg(v, 'dino'), 'arcface': agg(v, 'arcface')} for k, v in rows.items()}
    delta = {k: summary[name][k]['mean'] - summary['baseline'][k]['mean'] for k in ('dino', 'arcface')}
    s = summary[name]
    passed = (s['arcface']['mean'] >= REQ['arcface_min'] and s['dino']['mean'] <= REQ['dino_max'] and s['arcface']['n_nan'] == 0)
    res = dict(scheme=name, note=note, requirement=REQ, passed=bool(passed), vram_gb=vram_check(),
               per_seed=rows, summary=summary, delta_vs_baseline=delta, timestamp=time.strftime('%Y-%m-%d %H:%M:%S'))
    json.dump(res, open(f'{outdir}/results.json', 'w'), indent=2)
    print(json.dumps(dict(scheme=name, passed=passed, summary=summary, delta_vs_baseline=delta, vram_gb=res['vram_gb']), indent=2))
    return res
