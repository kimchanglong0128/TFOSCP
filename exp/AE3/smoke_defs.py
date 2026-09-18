# AE-3: second 1-step backbone — Hyper-SD (ByteDance). 1-step: Hyper-SDXL-1step UNet, LCMScheduler, t=[800];
# 4-step ceiling: base SDXL + Hyper-SDXL-4steps LoRA (fused), DDIM trailing, eta=1, 4 steps. FaceID-PlusV2 adapter (no LoRA).
# Phase 0 sanity (id1, 5 seeds) -> phase 1 id1 20 seeds -> phase 2 30 FFHQ ids.
import sys, os, json, glob; sys.path.insert(0, '/workspace/exp'); sys.path.insert(0, '/workspace/exp/AE2')
import torch, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFilter
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, UNet2DConditionModel, LCMScheduler, DDIMScheduler
from transformers import CLIPVisionModelWithProjection, CLIPModel, CLIPProcessor
import importlib.util as _ilu
_s = _ilu.spec_from_file_location('adrun', '/workspace/exp/AD/run.py'); adrun = _ilu.module_from_spec(_s); _s.loader.exec_module(adrun); rotate_lowfreq, _enc = adrun.rotate_lowfreq, adrun.encode
from common import REF, noise_for, check_image
from faceid import face_embeds
import idmetrics as m
OUT = '/workspace/exp/AE3'; SEEDS = list(range(42, 62)); SUBJ = 'person'; BASE = "stabilityai/stable-diffusion-xl-base-1.0"
PROMPT = f"a close-up portrait photo of a {SUBJ} wearing sunglasses, on a beach"
def attach_faceid(pipe):
    pipe.load_ip_adapter("h94/IP-Adapter-FaceID", subfolder=None, weight_name="ip-adapter-faceid-plusv2_sdxl.bin", image_encoder_folder=None); pipe.set_ip_adapter_scale(0.6); pipe.set_progress_bar_config(disable=True); return pipe
def build_1step():
    unet = UNet2DConditionModel.from_config(UNet2DConditionModel.load_config(BASE, subfolder="unet")).to('cuda', torch.float16)
    from safetensors.torch import load_file
    sd = load_file(hf_hub_download("ByteDance/Hyper-SD", "Hyper-SDXL-1step-Unet.safetensors")); missing, _ = unet.load_state_dict(sd, strict=False); assert len(missing) == 0, missing[:3]
    enc = CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter", subfolder="models/image_encoder", torch_dtype=torch.float16)
    pipe = DiffusionPipeline.from_pretrained(BASE, unet=unet, image_encoder=enc, dtype=torch.float16, variant="fp16", use_safetensors=True).to('cuda')
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config); return attach_faceid(pipe)
def build_4step():
    enc = CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter", subfolder="models/image_encoder", torch_dtype=torch.float16)
    pipe = DiffusionPipeline.from_pretrained(BASE, image_encoder=enc, dtype=torch.float16, variant="fp16", use_safetensors=True).to('cuda')
    pipe.load_lora_weights(hf_hub_download("ByteDance/Hyper-SD", "Hyper-SDXL-4steps-lora.safetensors")); pipe.fuse_lora()
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing"); return attach_faceid(pipe)
def gen1(pipe, emb, latents, seed):
    g = torch.Generator('cuda').manual_seed(seed); return pipe(prompt=PROMPT, timesteps=[800], guidance_scale=0, ip_adapter_image_embeds=emb, latents=latents, generator=g).images[0]
def gen4(pipe, emb, seed):
    g = torch.Generator('cuda').manual_seed(seed); return pipe(prompt=PROMPT, num_inference_steps=4, guidance_scale=0, eta=1.0, ip_adapter_image_embeds=emb, generator=g).images[0]
