#!/usr/bin/env python3
# webcam_mask.py — masque PIXEL-EXACT de la fenetre webcam (grabCut union multi-frames).
# Regle les fuites/formes (cam ronde, coin arrondi) que le rectangle rate. Le masque = la
# FENETRE (pas la personne) : on segmente sur plusieurs frames et on UNION (la personne bouge,
# la fenetre est statique -> l'union couvre toute la fenetre), puis remplissage trous + convexe.
# API : seg_mask(cap, W, H, t0, t1, seed_box, yfd) -> (mask uint8 HxW 0/1, [x,y,w,h] frac) ou (None,None)
import cv2, numpy as np

def _grab_one(fr, seed):
    H, W = fr.shape[:2]
    x, y, w, h = seed
    x = max(0, min(W-2, x)); y = max(0, min(H-2, y))
    w = max(4, min(W-x, w)); h = max(4, min(H-y, h))
    m = np.zeros((H, W), np.uint8)
    try:
        cv2.grabCut(fr, m, (x, y, w, h), np.zeros((1,65),np.float64), np.zeros((1,65),np.float64),
                    4, cv2.GC_INIT_WITH_RECT)
    except Exception:
        return None
    return np.where((m==cv2.GC_FGD)|(m==cv2.GC_PR_FGD), 1, 0).astype(np.uint8)

def seg_mask(cap, W, H, t0, t1, seed_box, yfd):
    """seed_box = [x,y,w,h] fractions (indicatif). La graine grabcut est en fait construite
    CENTREE SUR LE VISAGE et genereuse (une fenetre webcam ~ 3x le visage) pour ne pas rater
    les bords (bras a gauche, etc.). Retourne masque fenetre + bbox."""
    dur = t1 - t0
    bx, by, bw, bh = seed_box
    bcx, bcy = bx + bw/2, by + bh/2   # centre indicatif (secours si aucun visage)
    acc = np.zeros((H, W), np.uint16)
    n = 0; fcxs = []; fcys = []
    for f in (0.2, 0.4, 0.6, 0.8):
        cap.set(cv2.CAP_PROP_POS_MSEC, (t0 + dur*f)*1000.0)
        ok, fr = cap.read()
        if not ok: continue
        # graine CENTREE VISAGE, genereuse (3.4x large / 3.8x haut) — laisse grabcut trouver
        # toute la fenetre. Fallback : centre du seed_box.
        _, faces = yfd.detect(fr)
        if faces is not None and len(faces) > 0:
            fc = max(faces, key=lambda z: z[2]*z[3])
            cx = (fc[0]+fc[2]/2)/W; cy = (fc[1]+fc[3]/2)/H
            gw = min(0.5, (fc[2]/W)*3.4); gh = min(0.9, (fc[3]/H)*3.8)
        else:
            cx, cy, gw, gh = bcx, bcy, min(0.5, bw*1.6), min(0.9, bh*1.4)
        fcxs.append(cx); fcys.append(cy)
        sx = int(max(0, (cx-gw/2))*W); sy = int(max(0, (cy-gh*0.42))*H)
        sw2 = int(min(W-sx, gw*W)); sh2 = int(min(H-sy, gh*H))
        m = _grab_one(fr, (sx, sy, sw2, sh2))
        if m is None: continue
        acc += m; n += 1
    if n < 2:
        return None, None
    fcx = int(np.median(fcxs)*W) if fcxs else int(bcx*W)
    fcy = int(np.median(fcys)*H) if fcys else int(bcy*H)
    # UNION : pixel retenu s'il est foreground sur >=1 frame (couvre la fenetre malgre le mouvement)
    uni = (acc >= 1).astype(np.uint8)
    # garde le blob contenant le visage
    nb, lab, stats, _ = cv2.connectedComponentsWithStats(uni, 8)
    fid = int(lab[min(H-1,fcy), min(W-1,fcx)])
    if fid == 0:
        return None, None
    win = (lab == fid).astype(np.uint8)
    # remplissage : ferme les trous internes -> fenetre pleine (close + flood des trous)
    win = cv2.morphologyEx(win, cv2.MORPH_CLOSE, np.ones((21,21),np.uint8))
    ff = win.copy()
    hh, ww = win.shape
    fmask = np.zeros((hh+2, ww+2), np.uint8)
    cv2.floodFill(ff, fmask, (0,0), 1)
    win = win | (1 - ff)   # trous internes -> 1
    ys, xs = np.where(win > 0)
    if len(xs) == 0:
        return None, None
    bx0, bx1, by0, by1 = xs.min(), xs.max(), ys.min(), ys.max()
    bbox = [round(bx0/W,4), round(by0/H,4), round((bx1-bx0)/W,4), round((by1-by0)/H,4)]
    return win, bbox

if __name__ == "__main__":
    import sys
    YUNET = "/home/boss/videogen/face_detection_yunet_2023mar.onnx"
    vid, t0, t1, sb = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), [float(x) for x in sys.argv[4].split(",")]
    cap = cv2.VideoCapture(f"/home/boss/videogen/wk_b_{vid}/source.mp4")
    W=int(cap.get(3)); H=int(cap.get(4))
    fd = cv2.FaceDetectorYN.create(YUNET,"",(W,H),score_threshold=0.5)
    m, bb = seg_mask(cap, W, H, t0, t1, sb, fd)
    print("bbox", bb, "aire", (m.sum()/float(W*H)) if m is not None else None)
    if m is not None:
        cap.set(cv2.CAP_PROP_POS_MSEC, ((t0+t1)/2)*1000.0); _, fr = cap.read()
        g = np.zeros_like(fr); g[:] = (0,255,0)
        out = np.where(m[...,None]==1, (0.4*fr+0.6*g).astype(np.uint8), fr)
        cv2.imwrite(f"/tmp/mask_{vid}_{int(t0)}.jpg", out)
