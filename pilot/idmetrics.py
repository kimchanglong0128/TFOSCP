# Shared identity metrics: DINOv2 CLS cosine and ArcFace cosine against reference images.
import cv2, numpy as np, torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from insightface.app import FaceAnalysis

_app = FaceAnalysis(name='buffalo_l', root='/workspace/insightface', providers=['CPUExecutionProvider'])
_app.prepare(ctx_id=0, det_size=(320, 320))
_proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
_dino = AutoModel.from_pretrained("facebook/dinov2-base").to('cuda').eval()

def dino_embed(pil):
    inputs = _proc(images=pil, return_tensors='pt').to('cuda')
    with torch.no_grad():
        out = _dino(**inputs)
    return F.normalize(out.last_hidden_state[:, 0], dim=-1)[0].cpu().numpy()

def arc_embed(bgr):
    faces = _app.get(bgr)
    return None if len(faces) == 0 else faces[0].normed_embedding

def face_bbox(bgr):
    faces = _app.get(bgr)
    return None if len(faces) == 0 else faces[0].bbox.astype(int)   # x1,y1,x2,y2

REFS = {'woman': '/workspace/pilot/refs/person.jpeg', 'dog': '/workspace/pilot/refs/dog.jpg'}
REF_DINO = {k: dino_embed(Image.open(v).convert('RGB')) for k, v in REFS.items()}
REF_ARC  = {'woman': arc_embed(cv2.imread(REFS['woman']))}

def score(path, subject):
    """Return (dino_cos, arcface_cos or nan) of a generated image vs its reference."""
    d = float(dino_embed(Image.open(path).convert('RGB')) @ REF_DINO[subject])
    a = float('nan')
    if subject in REF_ARC:
        e = arc_embed(cv2.imread(path))
        if e is not None:
            a = float(e @ REF_ARC[subject])
    return d, a
