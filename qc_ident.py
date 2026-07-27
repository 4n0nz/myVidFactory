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

def covered_by_avatar(t, fb, fr=None, fr2=None):
    """True si le visage fb=[x,y,w,h] est ENTIEREMENT dans une zone avatar active a t
    (le masque Anonymous est un visage qui bouge -> le QC se flaggait LUI-MEME, SdMp
    LEAK_6 en boucle). Un popout qui depasse n'est PAS entierement dedans -> reste flagge.
    TROU bouche : pip qui couvre juste la TETE d'un hero rate -> le CORPS plein ecran
    restait invisible au QC (pas de visage). Si le pip couvrant est PETIT (<25% ecran)
    et que la bande SOUS lui bouge fort (corps), c'est un hero rate -> fuite."""
    for s in hmap:
        if s["start"]-0.6 <= t <= s["end"]+0.6:
            if s["host"] == "hero": return True
            if s["host"] == "pip" and s.get("bbox"):
                b = s["bbox"]; m = 0.01
                if (fb[0] >= b[0]-m and fb[1] >= b[1]-m
                        and fb[0]+fb[2] <= b[0]+b[2]+m and fb[1]+fb[3] <= b[1]+b[3]+m):
                    if (fr is not None and fr2 is not None and b[2]*b[3] < 0.25
                            and b[1]+b[3] < 0.85):
                        y0 = int(min(H-2, (b[1]+b[3])*H)); y1 = int(min(H, (b[1]+b[3]+0.15)*H))
                        x0 = int(b[0]*W); x1 = int(min(W, (b[0]+b[2])*W))
                        if y1 > y0+4 and x1 > x0+4:
                            def _hot(a, c):
                                d = cv2.absdiff(cv2.cvtColor(a[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY),
                                                cv2.cvtColor(c[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY))
                                return float((d > 18).mean()) > 0.25
                            # corps = mouvement SOUTENU : 2 fenetres espacees d'1s toutes
                            # les deux chaudes. Une seule paire declenchait sur un scroll/
                            # curseur sous le pip -> qc_fix etendait la box pleine hauteur
                            # en patch intouchable (T-chq h=1.0, QQE 0.99x0.97, avatar
                            # geant, verdicts Boss). Un scroll est transitoire ; un corps
                            # qui parle bouge aux deux fenetres.
                            if _hot(fr, fr2):
                                f3 = _frame(t+1.0); f4 = _frame(t+1.5)
                                if f3 is not None and f4 is not None and _hot(f3, f4):
                                    return False   # corps qui bouge sous le pip = hero rate
                    return True
    return False
def _off_layout(t, fb):
    """Le visage tombe-t-il LOIN de la carte que l'avatar couvre a t ?

    Une FUITE est la carte du narrateur mal recouverte. Un visage qui matche son
    identite AILLEURS dans le cadre est du CONTENU : eglV 1365-1369, la page filmee
    montre sa PROPRE video (cx=0.580, cos 0.805-0.833, fh 0.127-0.131) pendant que sa
    vraie carte est en bas a gauche (cx=0.135, cos 0.906-0.936). La garde de persistance
    ne separe pas les deux (la video de la page garde le visage en place : cx 0.580 /
    0.580 / 0.581 a 1366 / 1367 / 1368) et le cos non plus. qc_fix unissait donc les
    deux fuites en UNE box [0.0089,0.1157,0.7724,0.8833] : 68% de l'ecran peint
    par-dessus le site (mesure greenscan passe 15, bord BAS touche), alors que la source
    a un pip rond avec ses marges. Meme doctrine que partout ailleurs dans la chaine :
    un layout = une position ET une taille.

    Tolerance 0.10 autour de la box : un popout deplace le centre de la carte de ~0.1 au
    plus. Aucune scene a t (carte ratee de bout en bout par pinpoint3) -> on ne filtre
    pas, la fuite reste. Les GROS visages (plein cadre) ne passent pas ici : un hero
    rate se juge sur la taille, pas sur la position.
    """
    cx, cy = fb[0]+fb[2]/2.0, fb[1]+fb[3]/2.0
    seen = False
    for s in hmap:
        if s["start"]-0.6 <= t <= s["end"]+0.6:
            if s["host"] != "pip" or not s.get("bbox"): return False
            b = s["bbox"]; seen = True
            if (b[0]-0.10 <= cx <= b[0]+b[2]+0.10
                    and b[1]-0.10 <= cy <= b[1]+b[3]+0.10): return False
    return seen

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
            # plein ecran = fuite peu importe l'identite. Seuil 0.38 : un vrai plan
            # serre a un visage >=0.40 (TzJC 0.40+, eglV 0.43-0.52) ; les visages de
            # CONTENU montent a 0.31 (4D7 carte template verticale : 0.22 les flaggait
            # sans test d'identite -> patch geant -> promotion hero -> ecran vert,
            # verdict Boss ; l'anneau d'entourage seul ne suffit pas, l'inertie de fin
            # de scroll [1.5-6.9] chevauche un vrai hero calme [1.67]).
            big = f[3]/H > 0.38
            # Le narrateur est aussi PRESENT PHYSIQUEMENT dans le b-roll (plan d'atelier,
            # photo de groupe) : son visage y cos-matche au-dessus du seuil de fuite, qc_fix
            # en fait un patch, pin_render le promeut hero -> ecran vert sur du contenu qui
            # doit rester INTACT (verdict Boss 27/07 07h50 : la presence physique du
            # narrateur dans un plan d'evenement ne declenche RIEN). Une FUITE, elle, est
            # le narrateur LIVE (sa carte webcam ou un plan face-camera) : identite franche.
            # Mesures eglV source : b-roll 29-38 cos 0.41-0.71 (h 0.06-0.19) contre live
            # 39-45 cos 0.89-0.93 — les deux nuages ne se touchent pas, seuil a 0.75
            # (= COS_PIP de pinpoint3, meme doctrine : identite exigee pour couvrir).
            _c = _cos(feat, narr)
            if 0.363 <= _c < 0.75: continue
            if not big and _c < 0.363: continue
            x=max(0,int(f[0])); y=max(0,int(f[1])); w=int(f[2]); h=int(f[3])
            if not big and _off_layout(t, [x/W, y/H, w/W, h/H]): continue
            if big and _cos(feat, narr) < 0.363:
                # gros visage d'identite INCONNUE : le contenu peut en montrer (4D7 :
                # carte template verticale avec preview video, faceH=0.29 -> flaggee
                # sans test d'identite -> patch geant qc_fix -> promotion hero
                # pin_render = ecran vert plein, verdict Boss). Un hero rate VIT :
                # son ENTOURAGE bouge de maniere soutenue (2 fenetres espacees d'1s,
                # patron corps-sous-pip) ET son visage RESTE EN PLACE a t+1 (patron
                # persistance pinpoint3 2eb3563). Mesures : preview carte = anneau
                # 0.0-1.5 avec au moins une fenetre morte ; scroll = position instable
                # (cy 0.50->0.33 ou visage disparu) ; vrai hero eglV = 27.9/16.3 +
                # stable. Narrateur reconnu (cos>=0.363) ne passe jamais ici.
                mx, my = int(w*0.6), int(h*0.6)
                X0, Y0 = max(0, x-mx), max(0, y-my)
                X1, Y1 = min(W, x+w+mx), min(H, y+h+my)
                def _ring(a, c):
                    if a is None or c is None: return 1e9   # non mesurable -> chaud
                    d = cv2.absdiff(cv2.cvtColor(a[Y0:Y1, X0:X1], cv2.COLOR_BGR2GRAY),
                                    cv2.cvtColor(c[Y0:Y1, X0:X1], cv2.COLOR_BGR2GRAY)
                                    ).astype(np.float32)
                    d[max(0, y-Y0):y-Y0+h, max(0, x-X0):x-X0+w] = 0.0
                    return float(d.sum()/max((Y1-Y0)*(X1-X0) - h*w, 1))
                f3 = _frame(min(t+1.0, DUR-0.6)); f4 = _frame(min(t+1.5, DUR-0.1))
                hot = _ring(fr, fr2) > 0.5 and _ring(f3, f4) > 0.5
                stable = False
                if f3 is not None:
                    _, nf = yfd.detect(f3)
                    if nf is not None:
                        stable = any(abs((g[0]+g[2]/2)-(x+w/2))/W < 0.08
                                     and abs((g[1]+g[3]/2)-(y+h/2))/H < 0.08
                                     for g in nf if g[3]/H > 0.22)
                if not (hot and stable): continue
            if covered_by_avatar(t, [x/W, y/H, w/W, h/H], fr, fr2): continue
            if fr2 is not None:
                a = cv2.cvtColor(fr[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                b = cv2.cvtColor(fr2[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                if b.shape == a.shape and float(cv2.absdiff(a, b).mean()) < 0.3:
                    continue
            if not big:
                # persistance : un visage cos-matche dans le CONTENU (thumbnail de
                # player, avatars de sidebar) se DEPLACE ou disparait pendant un
                # scroll (eglV 1366 : non re-detecte a t+0.5) -> le check statique a
                # position fixe le croit vivant, qc_fix l'unionne avec la box pip et
                # cree un patch geant (eglV 1365-85, 0.68x0.77). Le narrateur LIVE
                # persiste a ~la meme position (patron pinpoint3 2eb3563, deja
                # applique aux big ci-dessus ; backstage camera epaule derive
                # ~0.06/s < 0.08, mesure 27/07). La taille du visage NE separe PAS
                # (thumbnail 0.131 > b-roll legitime 0.099-0.108).
                fpers = _frame(min(t + 1.0, DUR - 0.1))
                persist = False
                if fpers is not None:
                    _, pf = yfd.detect(fpers)
                    if pf is not None:
                        persist = any(abs((g[0]+g[2]/2)-(x+w/2))/W < 0.08
                                      and abs((g[1]+g[3]/2)-(y+h/2))/H < 0.08
                                      and 0.7 < g[3]/max(float(f[3]), 1e-6) < 1.43
                                      for g in pf)
                if not persist: continue
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
