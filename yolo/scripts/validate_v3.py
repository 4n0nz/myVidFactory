#!/usr/bin/env python3
# Valide le modele v3 (8 personnes) sur des frames des 5 nouveaux createurs + 2 originaux.
from ultralytics import YOLO
import cv2, os, glob
m = YOLO(os.path.expanduser("~/yolo/runs/facecam3/weights/best.pt"))
out = os.path.expanduser("~/yolo/val_v3"); os.makedirs(out, exist_ok=True)
srcs = sorted(glob.glob(os.path.expanduser("~/videogen/wk_hv_*/source.mp4"))) + [
    "/home/boss/videogen/wk_masortie/source.mp4", "/home/boss/videogen/wk_ofr/source.mp4"]
for src in srcs:
    if not os.path.exists(src): continue
    name = os.path.basename(os.path.dirname(src))
    cap = cv2.VideoCapture(src); fps = cap.get(5) or 30; dur = cap.get(7)/fps
    for frac in (0.3, 0.5, 0.7):
        t = dur*frac; cap.set(1, int(t*fps)); ok, f = cap.read()
        if not ok: continue
        r = m.predict(f, imgsz=960, conf=0.25, verbose=False)[0]
        print("%s %.0f%% -> %d %s" % (name, frac*100, len(r.boxes), [round(float(b.conf[0]),2) for b in r.boxes]))
        for b in r.boxes:
            x1,y1,x2,y2=[int(v) for v in b.xyxy[0]]; cf=float(b.conf[0])
            cv2.rectangle(f,(x1,y1),(x2,y2),(0,0,255),4)
            cv2.putText(f,"%.2f"%cf,(x1,max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),3)
        cv2.imwrite(os.path.join(out,"%s_%d.jpg"%(name,int(frac*100))), cv2.resize(f,(960,540)), [cv2.IMWRITE_JPEG_QUALITY,88])
    cap.release()
print("VAL3_DONE")
