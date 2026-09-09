from transformers import AutoImageProcessor, AutoModel
import torch.nn.functional as F 
import torch
from PIL import Image

pil_image = Image.open('/workspace/pilot/refs/dog.jpg').convert('RGB')

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base").to('cuda').eval()

inputs = processor(images=pil_image, return_tensors='pt').to('cuda')

with torch.no_grad():
    out = model(**inputs)

feat = out.last_hidden_state[:, 0]
feat = F.normalize(feat, dim=-1)
print(feat.shape, feat.norm().item())