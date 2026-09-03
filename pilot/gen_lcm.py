import torch
import os 
from diffusers import DiffusionPipeline, LCMScheduler

os.makedirs('/workspace/outputs/w', exist_ok = True)
os.makedirs('/workspace/outputs/steps', exist_ok = True)

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", dtype=torch.float16, variant="fp16", use_safetensors=True)
pipe.to('cuda')
print(pipe.unet.dtype)
print("loaded:", pipe.__class__.__name__)

pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
print(pipe.scheduler.__class__.__name__)


for i in [1, 1.5, 2, 4 ,8]:
    generator = torch.Generator('cuda').manual_seed(42)
    image = pipe(
        prompt = 'a photo of a corgi wearing sunglasses on a beach',
        num_inference_steps = 4,
        guidance_scale = i,
        generator = generator
    ).images[0]
    image.save(f'/workspace/outputs/w/lcm_4step_w{i}.png')

for n in [8, 4, 2, 1]:
    generator = torch.Generator('cuda').manual_seed(42)
    image = pipe(
        prompt = 'a photo of a corgi wearing sunglasses on a beach',
        num_inference_steps = n,
        guidance_scale = 1,
        # num_images_per_prompt = 10,
        generator = generator
    ).images[0]
    image.save(f'/workspace/outputs/steps/lcm_step_{n}.png')