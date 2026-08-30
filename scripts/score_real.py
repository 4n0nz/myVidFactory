#!/usr/bin/env python3
# Score un modele YOLO facecam sur VRAIES frames. Metric bulk : par video, on echantillonne
# 4 frames ; +1 par frame avec exactement 1 detection conf>0.5 (webcam trouvee, pas de decoy),
# -0.5 par frame avec >1 detection (decoy/faux positif). Sortie = total + detail par createur.
import sys, os, cv2
from ultralytics import YOLO
w = sys.argv[1]
m = YOLO(w)
srcs = [("6gtf","wk_test_6GtF"),("1x32","wk_test_1x32"),("ethx","wk_test_Ethx"),
        ("vkmx","wk_test_vKMx"),("masortie","wk_masortie"),("ofr","wk_ofr")]
base="/home/boss/videogen"
total=0.0; det=0
line=[]
for name,wd in srcs:
    src=f"{base}/{wd}/source.mp4"
    if not os.path.exists(src): line.append(f"{name}:NA"); continue
    cap=cv2.VideoCapture(src); fps=cap.get(5) or 30; dur=cap.get(7)/fps
    sc=0.0; d=0; parts=[]
    for frac in (0.2,0.4,0.6,0.8):
        cap.set(1,int(dur*frac*fps)); ok,f=cap.read()
        if not ok: parts.append("-"); continue
        r=m.predict(f,imgsz=960,conf=0.25,verbose=False)[0]
        strong=[b for b in r.boxes if float(b.conf[0])>0.5]
        n=len(strong)
        if n==1: sc+=1; d+=1; parts.append("1")
        elif n==0: parts.append("0")
        else: sc-=0.5; d+=1; parts.append(f"{n}!")
    cap.release()
    total+=sc; det+=d
    line.append(f"{name}:{d}/4[{''.join(parts)}]")
print(f"{os.path.basename(os.path.dirname(os.path.dirname(w)))}/{os.path.basename(w)}  SCORE={total:.1f} det={det}/24 | "+" ".join(line))
