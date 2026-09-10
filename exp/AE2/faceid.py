# Shared: DMD2 pipeline with IP-Adapter FaceID-PlusV2 SDXL (ArcFace id embedding + CLIP face crop, shortcut=True).
import sys, os; sys.path.insert(0, '/workspace/exp'); sys.path.insert(0, '/workspace/pilot')
import torch, numpy as np, cv2
from PIL import Image
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, UNet2DConditionModel, LCMScheduler
from transformers import CLIPVisionModelWithProjection
import idmetrics as m
BASE = "stabilityai/stable-diffusion-xl-base-1.0"
def build_faceid(unet_file, safetensors, scale=0.6, lora=True):
    unet = UNet2DConditionModel.from_config(UNet2DConditionModel.load_config(BASE, subfolder="unet")).to('cuda', torch.float16)
    path = hf_hub_download("tianweiy/DMD2", unet_file)
    sd = __import__('safetensors.torch', fromlist=['load_file']).load_file(path) if safetensors else torch.load(path, map_location='cpu')
    missing, _ = unet.load_state_dict(sd, strict=False); assert len(missing) == 0
    enc = CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter", subfolder="models/image_encoder", torch_dtype=torch.float16)
    pipe = DiffusionPipeline.from_pretrained(BASE, unet=unet, image_encoder=enc, dtype=torch.float16, variant="fp16", use_safetensors=True).to('cuda')
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_ip_adapter("h94/IP-Adapter-FaceID", subfolder=None, weight_name="ip-adapter-faceid-plusv2_sdxl.bin", image_encoder_folder=None)
    pipe.set_ip_adapter_scale(scale)
    if lora:
        pipe.load_lora_weights("h94/IP-Adapter-FaceID", weight_name="ip-adapter-faceid-plusv2_sdxl_lora.safetensors", adapter_name="faceid")
        pipe.set_adapters(["faceid"], adapter_weights=[0.6])
    pipe.set_progress_bar_config(disable=True); return pipe
def face_embeds(pipe, ref_pil, do_cfg=False):
    """returns ip_adapter_image_embeds list for FaceID-PlusV2: id embedding (ArcFace) + sets CLIP face-crop embeds on the projection layer."""
    bgr = cv2.cvtColor(np.asarray(ref_pil.convert('RGB')), cv2.COLOR_RGB2BGR); faces = m._app.get(bgr); assert len(faces) >= 1, "no face in reference"
    f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
    idv = torch.from_numpy(f.normed_embedding).unsqueeze(0).unsqueeze(0).to('cuda', torch.float16)  # (1,1,512) = (batch, n_images, 512)
    x1, y1, x2, y2 = f.bbox.astype(int); pad = int(0.3 * (x2 - x1)); crop = ref_pil.crop((max(0, x1-pad), max(0, y1-pad), x2+pad, y2+pad)).resize((224, 224))
    clip = pipe.prepare_ip_adapter_image_embeds([crop], None, 'cuda', 1, do_cfg)[0]            # hidden states of face crop
    layer = pipe.unet.encoder_hid_proj.image_projection_layers[0]; layer.clip_embeds = clip.to(torch.float16); layer.shortcut = True
    if do_cfg: idv = torch.cat([torch.zeros_like(idv), idv])
    return [idv]
