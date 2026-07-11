#!/usr/bin/env python3
# Harvest de crops facecam depuis une video, via YuNet + live-motion (l'heuristique de analyze_host2).
# Sort des crops de facecam (insets = visage vivant, petit/coin) + des fonds (contenu sans facecam),
# pour alimenter le dataset synthetique YOLO. PAS de render, juste ~80 frames echantillonnees.
# Usage : harvest_crops.py <source.mp4> <name>
import cv2, sys, os
import numpy as np

src = sys.argv[1]; name = sys.argv[2]
YUNET = '/home/boss/videogen/face_detection_yunet_2023mar.onnx'
CROPS = os.path.expanduser("~/yolo/data/crops/%s" % name)
BGS   = os.path.expanduser("~/yolo/data/bgs/%s" % name)
os.makedirs(CROPS, exist_ok=True); os.makedirs(BGS, exist_ok=True)

cap = cv2.VideoCapture(src)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
W = int(cap.get(3)); H = int(cap.get(4)); total = int(cap.get(7))
dur = total / fps
fd = cv2.FaceDetectorYN.create(YUNET, "", (W, H), score_threshold=0.5, nms_threshold=0.3, top_k=5000)
fd.setInputSize((W, H))
K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
MOT_DT = 0.15; MOT_TH = 12; LIVE = 0.12

def frame_at(t):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps)); ok, f = cap.read(); return f if ok else None

n_face = n_bg = 0
N = 90
for i in range(N):
    t = dur * (i + 0.5) / N
    f0 = frame_at(t)
    if f0 is None: continue
    f1 = frame_at(min(dur - 0.05, t + MOT_DT)); f1 = f1 if f1 is not None else f0
    g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY); g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    mot = (cv2.absdiff(g0, g1) > MOT_TH).astype(np.uint8) * 255
    motc = cv2.dilate(cv2.morphologyEx(mot, cv2.MORPH_CLOSE, K, iterations=2), K, iterations=1)
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(motc)
    _, faces = fd.detect(f0)
    faces = faces if faces is not None else []
    best = None
    for fc in faces:
        fx, fy, fw, fh = int(fc[0]), int(fc[1]), int(fc[2]), int(fc[3])
        if fw < 8 or fh < 8: continue
        # inset : visage pas trop gros (sinon = narrateur plein ecran, pas un facecam pip)
        if fw > 0.42 * W: continue
        cx = min(W-1, max(0, fx+fw//2)); cy = min(H-1, max(0, fy+fh//2))
        rx0, ry0 = max(0, fx-fw//2), max(0, fy-fh//2); rx1, ry1 = min(W, fx+fw+fw//2), min(H, fy+fh+2*fh)
        if motc[ry0:ry1, rx0:rx1].mean()/255.0 < LIVE: continue   # pas vivant -> thumbnail
        lid = int(lab[cy, cx])
        if lid > 0:
            mx, my, mw, mh = int(stats[lid,0]), int(stats[lid,1]), int(stats[lid,2]), int(stats[lid,3])
        else:
            mx, my, mw, mh = fx, fy, fw, fh
        # borner autour du visage (anti-bridge vers le contenu)
        cx0, cy0 = max(0, fx-fw), max(0, fy-fh); cx1, cy1 = min(W, fx+2*fw), min(H, fy+3*fh)
        ax, ay = max(mx, cx0), max(my, cy0); ax2, ay2 = min(mx+mw, cx1), min(my+mh, cy1)
        aw, ah = ax2-ax, ay2-ay
        if aw < 40 or ah < 40: continue
        area = aw*ah
        if best is None or area > best[4]:
            best = (ax, ay, aw, ah, area)
    if best:
        ax, ay, aw, ah, _ = best
        crop = f0[ay:ay+ah, ax:ax+aw]
        if crop.size:
            cv2.imwrite(os.path.join(CROPS, "f%03d.jpg" % i), crop, [cv2.IMWRITE_JPEG_QUALITY, 90]); n_face += 1
        # fond = region large a l'oppose du facecam (moitie sans le crop)
        if ax > W/2:   bg = f0[:, :max(1, ax-10)]
        else:          bg = f0[:, min(W-1, ax+aw+10):]
        if bg.size and bg.shape[1] > 200:
            cv2.imwrite(os.path.join(BGS, "b%03d.jpg" % i), cv2.resize(bg, (1280, 720)), [cv2.IMWRITE_JPEG_QUALITY, 85]); n_bg += 1
cap.release()
print("harvest %s -> %d crops, %d fonds" % (name, n_face, n_bg))
