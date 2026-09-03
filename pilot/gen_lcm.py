import torch
from diffusers import DiffusionPipeline 
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", dtype=torch.float16, variant="fp16", use_safetensors=True)
pipe.to('cuda')
print(pipe.unet.dtype)
print("loaded:", pipe.__class__.__name__)