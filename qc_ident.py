#!/usr/bin/env python3
# qc_ident.py <workdir> <rendered.mp4> — QC final : scan du RENDU a la recherche du narrateur.
# Le narrateur (identite SFace de narrator_feat.npy) ne doit plus apparaitre en LIVE nulle part.
# Toute occurrence visage-narrateur + mouvement = FUITE -> timestamp rapporte, exit 1.
# (Ses photos/vignettes statiques dans le contenu sont legitimes -> gate mouvement.)
import sys, os, json, cv2, numpy as np

wd, rend = sys.argv[1], sys.argv[2]
VG = "/home/boss/videogen"
narr = np.load(os.path.join(wd, "narrator_feat.npy"))
hmap = json.load(open(os.path.join(wd, "host_map.json")))

def covered_by_avatar(t, fb):
    """True si le visage fb=[x,y,w,h] est ENTIEREMENT dans une zone avatar active a t
    (le masque Anonymous est un visage qui bouge -> le QC se flaggait LUI-MEME, SdMp
    LEAK_6 en boucle). Un popout qui depasse n'est PAS entierement dedans -> reste flagge."""
    for s in hmap:
        if s["start"]-0.6 <= t <= s["end"]+0.6:
            if s["host"] == "hero": return True
            if s["host"] == "pip" and s.get("bbox"):
                b = s["bbox"]; m = 0.01
                if (fb[0] >= b[0]-m and fb[1] >= b[1]-m
                        and fb[0]+fb[2] <= b[0]+b[2]+m and fb[1]+fb[3] <= b[1]+b[3]+m):
                    return True
    return False
cap = cv2.VideoCapture(rend)
W = int(cap.get(3)); H = int(cap.get(4)); DUR = cap.get(7)/(cap.get(5) or 30)
yfd = cv2.FaceDetectorYN.create(VG+"/face_detection_yunet_2023mar.onnx", "", (W, H), score_threshold=0.6)
rec = cv2.FaceRecognizerSF.create(VG+"/face_recognition_sface_2021dec.onnx", "")

def _frame(t):
    cap.set(cv2.CAP_PROP_POS_MSEC, t*1000.0); ok, fr = cap.read()
    return fr if ok else None

def _cos(a, b):
    return float(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))

leaks = []
t = 0.5
while t < DUR:
    fr = _frame(t); fr2 = _frame(min(t+0.5, DUR-0.05))
    if fr is None: t += 1.0; continue
    _, faces = yfd.detect(fr)
    if faces is not None:
        for f in faces:
            if f[3]/H < 0.045: continue
            try:
                feat = rec.feature(rec.alignCrop(fr, f)).flatten().astype(np.float32)
            except Exception:
                continue
            big = f[3]/H > 0.22   # plein ecran = fuite peu importe l'identite
            if not big and _cos(feat, narr) < 0.363: continue
            x=max(0,int(f[0])); y=max(0,int(f[1])); w=int(f[2]); h=int(f[3])
            if covered_by_avatar(t, [x/W, y/H, w/W, h/H]): continue
            if fr2 is not None:
                a = cv2.cvtColor(fr[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                b = cv2.cvtColor(fr2[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                if b.shape == a.shape and float(cv2.absdiff(a, b).mean()) < 0.3:
                    continue
            leaks.append({"t": round(t,2), "box": [round(x/W,4), round(y/H,4),
                          round(w/W,4), round(h/H,4)]})
    t += 1.0
cap.release()
json.dump(leaks, open(os.path.join(wd, "qc_leaks.json"), "w"))
if leaks:
    print("QC FUITES : %d occurrences narrateur dans le rendu" % len(leaks))
    for L in leaks[:40]:
        print("  t=%.1fs pos=(%.3f,%.3f) fh=%.3f" % (L["t"], L["box"][0], L["box"][1], L["box"][3]))
    sys.exit(1)
print("QC PROPRE : narrateur introuvable dans le rendu (0 fuite)")
