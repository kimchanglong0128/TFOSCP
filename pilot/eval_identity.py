from insightface.app import FaceAnalysis
import cv2 
from transformers import AutoImageProcessor, AutoModel
import torch.nn.functional as F 
import pandas as pd
import numpy as np 
import torch
from PIL import Image

cate = {"woman":"person.jpeg", "dog": "dog.jpg"}
app = FaceAnalysis(name='buffalo_l', root='/workspace/insightface', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base").to('cuda').eval()

def dino_embed(pil):
    inputs = processor(images=pil, return_tensors='pt').to('cuda')
    with torch.no_grad():
        out = model(**inputs)

    feat = out.last_hidden_state[:, 0]
    feat = F.normalize(feat, dim=-1)

    return feat[0].cpu().numpy()

def arc_embed(bgr):
    faces = app.get(bgr)
    return None if len(faces) == 0 else faces[0].normed_embedding
      
ref_dino = {}
for subject, fname in cate.items():
    pil = Image.open(f"/workspace/pilot/refs/{fname}").convert('RGB')
    ref_dino[subject] = dino_embed(pil)

ref_arc = {'woman': arc_embed(cv2.imread('/workspace/pilot/refs/person.jpeg'))}

IN_DIR = '/workspace/outputs/ipa_seeds'
rows = []
for subject in cate:
    for step in [8, 6, 4, 3, 2, 1]:
        for seed in [42, 43, 44]:
            path = f'{IN_DIR}/ref_{subject}/{subject}_step{step}_seed{seed}.png'
            d = float(dino_embed(Image.open(path).convert('RGB')) @ ref_dino[subject])
            if subject in ref_arc:
                e = arc_embed(cv2.imread(path))
                a = float('nan') if e is None else float(e @ ref_arc[subject])
            else:
                a = float('nan')
            rows.append(dict(subject=subject, step=step, dino=d, arcface=a, seed=seed))

df = pd.DataFrame(rows)
print(df.groupby(['subject', 'step'])[['dino', 'arcface']].agg(['mean', 'std']).round(3))
df.to_csv('/workspace/outputs/eval_ipa_seeds.csv', index=False)
