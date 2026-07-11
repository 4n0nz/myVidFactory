#!/usr/bin/env python3
# Valide un modele YOLO facecam sur de VRAIES frames : 4 nouveaux createurs + 2 originaux.
import sys, os, glob, cv2
from ultralytics import YOLO
weights = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/yolo/runs/facecam4/weights/best.pt")
m = YOLO(weights)
out = os.path.expanduser("~/yolo/val_v4"); os.makedirs(out, exist_ok=True)
srcs = [
  ("6gtf","/home/boss/videogen/wk_test_6GtF/source.mp4"),
  ("1x32","/home/boss/videogen/wk_test_1x32/source.mp4"),
  ("ethx","/home/boss/videogen/wk_test_Ethx/source.mp4"),
  ("vkmx","/home/boss/videogen/wk_test_vKMx/source.mp4"),
  ("masortie","/home/boss/videogen/wk_masortie/source.mp4"),
  ("ofr","/home/boss/videogen/wk_ofr/source.mp4"),
]
for name, src in srcs:
    if not os.path.exists(src): print("skip",name); continue
    cap=cv2.VideoCapture(src); fps=cap.get(5) or 30; dur=cap.get(7)/fps
    res=[]
    for frac in (0.2,0.4,0.6,0.8):
        t=dur*frac; cap.set(1,int(t*fps)); ok,f=cap.read()
        if not ok: continue
        r=m.predict(f,imgsz=960,conf=0.25,verbose=False)[0]
        confs=[round(float(b.conf[0]),2) for b in r.boxes]
        res.append("%.0f%%:%d%s"%(frac*100,len(r.boxes),confs))
        for b in r.boxes:
            x1,y1,x2,y2=[int(v) for v in b.xyxy[0]]; cf=float(b.conf[0])
            cv2.rectangle(f,(x1,y1),(x2,y2),(0,0,255),4)
            cv2.putText(f,"%.2f"%cf,(x1,max(24,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),3)
        cv2.imwrite(os.path.join(out,"%s_%d.jpg"%(name,int(frac*100))),cv2.resize(f,(960,540)),[cv2.IMWRITE_JPEG_QUALITY,88])
    cap.release()
    print("%-9s | %s"%(name, "  ".join(res)))
print("VAL4_DONE")
