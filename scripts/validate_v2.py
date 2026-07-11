#!/usr/bin/env python3
# Valide le modele v2 (3 sources) sur de vraies frames des 3 videos, incl. Ofr (jamais fixe avant).
from ultralytics import YOLO
import cv2, os
W = os.path.expanduser("~/yolo/runs/facecam2/weights/best.pt")
m = YOLO(W)
tests = [
    ("/home/boss/videogen/wk_masortie/source.mp4",  [202, 1065]),
    ("/home/boss/videogen/wk_lastvideo/source.mp4", [220, 1100]),
    ("/home/boss/videogen/wk_ofr/source.mp4",       [100, 142, 230, 300, 430, 585]),
]
out = os.path.expanduser("~/yolo/val_v2"); os.makedirs(out, exist_ok=True)
for src, ts in tests:
    if not os.path.exists(src): continue
    cap = cv2.VideoCapture(src); fps = cap.get(cv2.CAP_PROP_FPS) or 30
    name = os.path.basename(os.path.dirname(src))
    for t in ts:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t*fps)); ok, f = cap.read()
        if not ok: continue
        r = m.predict(f, imgsz=960, conf=0.25, verbose=False)[0]
        confs = [round(float(b.conf[0]),2) for b in r.boxes]
        print("%s t=%d -> %d box %s" % (name, t, len(r.boxes), confs))
        for b in r.boxes:
            x1,y1,x2,y2=[int(v) for v in b.xyxy[0]]; cf=float(b.conf[0])
            cv2.rectangle(f,(x1,y1),(x2,y2),(0,0,255),4)
            cv2.putText(f,"%.2f"%cf,(x1,max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),3)
        cv2.imwrite(os.path.join(out,"%s_%d.jpg"%(name,t)), cv2.resize(f,(960,540)), [cv2.IMWRITE_JPEG_QUALITY,88])
    cap.release()
print("VAL2_DONE")
