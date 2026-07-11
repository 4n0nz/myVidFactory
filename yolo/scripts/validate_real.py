#!/usr/bin/env python3
# Teste le YOLO facecam entraine sur de VRAIES frames source (pas synthetiques).
# Le vrai test de generalisation : trouve-t-il le facecam a la bonne position/taille ?
from ultralytics import YOLO
import cv2, os
W = os.path.expanduser("~/yolo/runs/facecam/weights/best.pt")
m = YOLO(W)
tests = [
    ("/home/boss/videogen/wk_masortie/source.mp4",  [202, 367, 1065]),
    ("/home/boss/videogen/wk_lastvideo/source.mp4", [220, 700, 1100]),
]
out = os.path.expanduser("~/yolo/val_real"); os.makedirs(out, exist_ok=True)
for src, ts in tests:
    if not os.path.exists(src): print("absent:", src); continue
    cap = cv2.VideoCapture(src); fps = cap.get(cv2.CAP_PROP_FPS) or 30
    name = os.path.basename(os.path.dirname(src))
    for t in ts:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps)); ok, f = cap.read()
        if not ok: continue
        r = m.predict(f, imgsz=960, conf=0.25, verbose=False)[0]
        n = len(r.boxes)
        for b in r.boxes:
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0]]; cf = float(b.conf[0])
            cv2.rectangle(f, (x1, y1), (x2, y2), (0, 0, 255), 4)
            cv2.putText(f, "facecam %.2f" % cf, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
        cv2.imwrite(os.path.join(out, "%s_%d.jpg" % (name, t)), cv2.resize(f, (960, 540)),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        print("%s t=%d -> %d facecam(s)" % (name, t, n))
    cap.release()
print("VAL_DONE ->", out)
