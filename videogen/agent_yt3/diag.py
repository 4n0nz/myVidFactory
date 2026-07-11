
import cv2, sys
import mediapipe as mp
from mediapipe.tasks import python as P
from mediapipe.tasks.python import vision as V
MODEL='/home/anon/videogen/efficientdet.tflite'
det=V.ObjectDetector.create_from_options(V.ObjectDetectorOptions(base_options=P.BaseOptions(model_asset_path=MODEL),score_threshold=0.30,category_allowlist=['person'],max_results=5))
cap=cv2.VideoCapture('/home/anon/videogen/agent_yt3/source_h264.mp4')
fps=cap.get(cv2.CAP_PROP_FPS); W=int(cap.get(3)); H=int(cap.get(4))
t=0.0
while t<15:
    cap.set(cv2.CAP_PROP_POS_FRAMES,int(t*fps)); ok,fr=cap.read()
    if not ok: t+=0.5; continue
    rgb=cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)
    res=det.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb))
    if res.detections:
        b=max(res.detections,key=lambda d:d.bounding_box.width*d.bounding_box.height).bounding_box
        fw=b.width/W; fh=b.height/H; cx=(b.origin_x+b.width/2)/W; cy=(b.origin_y+b.height/2)/H
        n=len(res.detections)
        print("t=%.1f area=%.3f cx=%.2f cy=%.2f n=%d"%(t,fw*fh,cx,cy,n))
    else:
        print("t=%.1f NONE"%t)
    t+=0.5
