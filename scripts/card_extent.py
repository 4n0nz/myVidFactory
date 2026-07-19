#!/usr/bin/env python3
# card_extent.py — detecteur de CARTE webcam (rectangle arrondi) robuste.
# Remplace le sizing-visage : on trouve la carte complete qui contient le visage YuNet.
# Multi-frames (mediane sur 3), extension haut-de-tete, fallback None si rien.
# API : card_box(cap, W, H, t_start, t_end, yfd) -> [x,y,w,h] fractions ou None.
import cv2, numpy as np, statistics

def _card_one(fr, fx, fy, fw, fh):
    """Une frame : plus grand contour rectangulaire contenant le visage. Pixels -> (x,y,w,h)."""
    H, W = fr.shape[:2]
    fcx, fcy = fx + fw/2, fy + fh/2
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 90)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=2)
    cnts, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < 8 or h < 8: continue
        if not (x <= fcx <= x+w and y <= fcy <= y+h): continue
        area = (w*h)/float(W*H)
        if area < 0.03 or area > 0.45: continue
        ar = w/float(h)
        if ar < 0.35 or ar > 2.8: continue
        if w < 1.3*fw or h < 1.3*fh: continue
        if best is None or w*h > best[2]*best[3]:
            best = (x, y, w, h)
    if best is None:
        return None
    x, y, w, h = best
    # GARDE cadrage webcam : le visage doit etre dans la MOITIE HAUTE de la carte (une webcam
    # cadre la tete en haut). Si le visage est bas dans la box = ce n'est pas une carte webcam
    # (contour de page attrape par erreur, cf KKni fond blanc) -> rejet, fallback.
    if fcy > y + 0.62*h:
        return None
    # GARDE centrage : le visage doit etre a peu pres au centre horizontal de la carte
    # (pas colle a un bord — sinon c'est un bloc de contenu, pas la carte).
    if fcx < x + 0.15*w or fcx > x + 0.85*w:
        return None
    # extension HAUT-DE-TETE : le contour rate souvent le sommet du crane -> on remonte de
    # 0.6x hauteur visage pour couvrir les cheveux, borne a l'ecran.
    want_top = max(0, int(fy - 0.6*fh))
    if y > want_top:
        h += (y - want_top); y = want_top
    # extension BAS : le contour rate souvent le bas de la carte (torse/mains qui fuient sous
    # l'avatar, cf 0sqC). On descend le bord bas tant que la colonne verticale sous le visage
    # RESSEMBLE a la carte (continuite de bord/contenu), jusqu'a un bord horizontal net ou le
    # bas de l'ecran. On scanne le gradient horizontal median dans la colonne du visage.
    y1 = y + h
    if y1 < H - 2:
        col = gray[:, max(0, int(fcx - 0.1*w)):min(W, int(fcx + 0.1*w))]
        gy = np.abs(cv2.Sobel(col, cv2.CV_32F, 0, 1, ksize=3)).mean(axis=1)
        lim = min(H, y1 + int(0.35 * h))          # au plus +35% de la hauteur carte
        seg = gy[y1:lim]
        if len(seg) > 3:
            j = int(np.argmax(seg))
            if float(seg[j]) > 22.0:               # bord bas net trouve
                y1 = y1 + j + 2
            else:
                y1 = lim                            # pas de bord -> etend au max (carte continue)
        h = y1 - y
    return (x, y, w, h)

def card_box(cap, W, H, t_start, t_end, yfd):
    dur = t_end - t_start
    cards = []
    for f in (0.3, 0.5, 0.7):
        cap.set(cv2.CAP_PROP_POS_MSEC, (t_start + dur*f)*1000.0)
        ok, fr = cap.read()
        if not ok: continue
        _, faces = yfd.detect(fr)
        if faces is None or len(faces) == 0: continue
        fc = max(faces, key=lambda z: z[2]*z[3])
        c = _card_one(fr, int(fc[0]), int(fc[1]), int(fc[2]), int(fc[3]))
        if c: cards.append(c)
    if len(cards) < 2:          # besoin d'au moins 2 frames concordantes
        return None
    x = statistics.median([c[0] for c in cards]); y = statistics.median([c[1] for c in cards])
    w = statistics.median([c[2] for c in cards]); h = statistics.median([c[3] for c in cards])
    return [round(x/W,4), round(y/H,4), round(w/W,4), round(h/H,4)]

if __name__ == "__main__":
    YUNET = "/home/boss/videogen/face_detection_yunet_2023mar.onnx"
    tests = [
        ("0sqCK2u70q4", 30, 60, (0.732,0.668,0.268,0.294)),
        ("KKniWb9RKq4", 80, 120, None),
        ("_KzObeom88Y", 380, 402, (0.7708,0.0,0.2292,1.0)),
        ("_KzObeom88Y", 600, 628, (0.7406,0.3565,0.2594,0.6389)),
        ("QQEgIo4Juxg", 100, 130, (0.681,0.0,0.319,1.0)),
        ("QQEgIo4Juxg", 356, 402, (0.813,0.0,0.187,1.0)),
    ]
    for vid, a, b, old in tests:
        path = f"/home/boss/videogen/wk_b_{vid}/source.mp4"
        cap = cv2.VideoCapture(path)
        W = int(cap.get(3)); H = int(cap.get(4))
        fd = cv2.FaceDetectorYN.create(YUNET, "", (W, H), score_threshold=0.5)
        nb = card_box(cap, W, H, a, b, fd)
        # dessine au milieu
        cap.set(cv2.CAP_PROP_POS_MSEC, ((a+b)/2)*1000.0); ok, fr = cap.read()
        if old:
            ox,oy,ow,oh = int(old[0]*W),int(old[1]*H),int(old[2]*W),int(old[3]*H)
            cv2.rectangle(fr,(ox,oy),(ox+ow,oy+oh),(0,0,255),3)
        if nb:
            x,y,w,h = int(nb[0]*W),int(nb[1]*H),int(nb[2]*W),int(nb[3]*H)
            cv2.rectangle(fr,(x,y),(x+w,y+h),(0,255,0),4)
        cv2.imwrite(f"/tmp/cv2_{vid}_{a}.jpg", fr)
        print(f"{vid} {a}-{b}: {'carte='+str(nb) if nb else 'AUCUNE (fallback)'}")
        cap.release()
