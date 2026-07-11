#!/usr/bin/env python3
# Diagnostic : YOLO facecam sur Ofr a TRES bas seuil -> detecte-t-il quelque chose ?
from ultralytics import YOLO
import cv2, os
m = YOLO(os.path.expanduser("~/yolo/runs/facecam/weights/best.pt"))
src = "/home/boss/videogen/wk_ofr/source.mp4"
out = os.path.expanduser("~/yolo/ofr_diag"); os.makedirs(out, exist_ok=True)
cap = cv2.VideoCapture(src); fps = cap.get(cv2.CAP_PROP_FPS) or 30
for t in [44, 100, 142, 230, 300, 430, 585]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t*fps)); ok, f = cap.read()
    if not ok: continue
    r = m.predict(f, imgsz=960, conf=0.05, verbose=False)[0]
    print("t=%d -> %d box(es):" % (t, len(r.boxes)), [round(float(b.conf[0]),2) for b in r.boxes])
    for b in r.boxes:
        x1,y1,x2,y2=[int(v) for v in b.xyxy[0]]; cf=float(b.conf[0])
        cv2.rectangle(f,(x1,y1),(x2,y2),(0,0,255),4)
        cv2.putText(f,"%.2f"%cf,(x1,max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),3)
    cv2.imwrite(os.path.join(out,"ofr_%d.jpg"%t), cv2.resize(f,(960,540)), [cv2.IMWRITE_JPEG_QUALITY,88])
cap.release()
print("DIAG_DONE")
