from insightface.app import FaceAnalysis
import cv2 

img = cv2.imread('/workspace/pilot/refs/person.jpeg')
app = FaceAnalysis(name='buffalo_l', root='/workspace/insightface', providers=['CPUExecutionProvider'])

app.prepare(ctx_id=0, det_size=(640, 640))
faces=app.get(img)

print(len(faces))
print(faces[0].normed_embedding.shape)