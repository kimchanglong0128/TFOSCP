# SDXL + LCM_LORA + Scheduler + IP-Adapter
from diffusers import DiffusionPipeline, LCMScheduler
import torch 
import os 
from PIL import Image

os.makedirs('/workspace/outputs/ipa', exist_ok=True)
os.makedirs('/workspace/outputs/ipa/ref_dog', exist_ok=True)
os.makedirs('/workspace/outputs/ipa/ref_woman', exist_ok=True)

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", dtype=torch.float16, variant="fp16", use_safetensors=True)
pipe.to('cuda')

pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")

pipe.load_ip_adapter("h94/IP-Adapter", subfolder='sdxl_models', weight_name='ip-adapter_sdxl.bin')
pipe.set_ip_adapter_scale(0.6)
print(type(list(pipe.unet.attn_processors.values())[0]).__name__)


refs = {'woman':'person.jpeg', 'dog':'dog.jpg'}
for i in [8, 6, 4, 3, 2, 1]:
    for j, k in refs.items():
        generator = torch.Generator('cuda').manual_seed(42)
        image = pipe(
            prompt = f"a photo of a {j} wearing sunglasses, on a beach",
            num_inference_steps=i,
            guidance_scale=1,
            ip_adapter_image = Image.open(f'/workspace/pilot/refs/{k}').convert('RGB'),
            generator = generator
        ).images[0]
        image.save(f'/workspace/outputs/ipa/ref_{j}/{j}_step{i}.png') 
