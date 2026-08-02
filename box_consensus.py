#!/usr/bin/env python3
# box_consensus.py <workdir> [draw_prefix] — mesure CONSENSUS TEMPOREL de la box pip.
#
# Principe (remplace la tour card_extent/motion_extend/grabcut/cap) : le pip est une
# VIDEO live -> bruit temporel PARTOUT et EN CONTINU (capteur, compression, respiration).
# La page derriere est STATIQUE -> pixels identiques entre frames sauf scroll (transitoire,
# minoritaire sur N paires). Carte d'activite = part des paires ou le pixel change.
# Le blob d'activite autour du visage narrateur = la carte, au pixel. Ses coins vides
# d'activite = la forme (rond / rect arrondi / rect90). UNE box par cluster de position,
# mesuree sur TOUTES ses scenes — aucune boucle corrective necessaire.
import sys, os, json, cv2, numpy as np

VG = "/home/boss/videogen"
wd = sys.argv[1].rstrip("/")
draw_prefix = sys.argv[2] if len(sys.argv) > 2 else None

src = os.path.join(wd, "source.mp4")
narr = np.load(os.path.join(wd, "narrator_feat.npy"))
cap = cv2.VideoCapture(src)
W = int(cap.get(3)); H = int(cap.get(4)); FPS = cap.get(5) or 30; DUR = cap.get(7)/FPS
yfd = cv2.FaceDetectorYN.create(VG+"/face_detection_yunet_2023mar.onnx", "", (W, H), score_threshold=0.6)
rec = cv2.FaceRecognizerSF.create(VG+"/face_recognition_sface_2021dec.onnx", "")

COS_SAME = 0.363
COS_PIP = 0.75      # identite MEDIANE exigee d un cluster pour etre une vraie carte
                    # (meme constante et meme regle que pinpoint3.py:45 / flag ghost)
ACT_DIFF = 5        # seuil de changement pixel (attrape le bruit video, pas le rendu statique)
ACT_MIN = 0.55      # pixel "carte" = change dans >55% des paires
NPAIRS = 60
BG_LIVE_MAX = 0.40  # part de l ecran HORS box qui bouge : au-dela, page animee = anneau aveugle

def _frame(t):
    cap.set(cv2.CAP_PROP_POS_MSEC, t*1000.0); ok, fr = cap.read()
    return fr if ok else None

def _cos(a, b):
    return float(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))

# ---- 1. echantillonnage narrateur (1/s) ----
samples = []
t = 0.0
while t < DUR:
    fr = _frame(t)
    if fr is not None:
        _, faces = yfd.detect(fr)
        if faces is not None:
            for f in faces:
                if f[3]/H < 0.045: continue
                try:
                    feat = rec.feature(rec.alignCrop(fr, f)).flatten().astype(np.float32)
                except Exception:
                    continue
                _c = _cos(feat, narr)
                if _c >= COS_SAME:
                    samples.append({"t": t, "f": [float(v) for v in f[:4]], "c": _c})
    t += 1.0
if not samples:
    print("aucun sample narrateur"); sys.exit(1)

# ---- 2. clusters de position (memes criteres que pinpoint3), heros ecartes ----
clusters = []
for s in samples:
    fh = s["f"][3]/H
    if fh > 0.40: continue                       # hero plein cadre : pas une box a mesurer
    cx = (s["f"][0]+s["f"][2]/2)/W; cy = (s["f"][1]+s["f"][3]/2)/H
    hit = None
    for c in clusters:
        if abs(cx-c["cx"]) < 0.10 and abs(cy-c["cy"]) < 0.10: hit = c; break
    if hit is None:
        clusters.append({"cx": cx, "cy": cy, "members": [s]})
    else:
        n = len(hit["members"])
        hit["cx"] = (hit["cx"]*n+cx)/(n+1); hit["cy"] = (hit["cy"]*n+cy)/(n+1)
        hit["members"].append(s)
clusters = [c for c in clusters if len(c["members"]) >= 8]
# Un cluster doit etre porte par l IDENTITE, pas seulement par la position.
# box_consensus recrute ses membres a COS_SAME (0.363) et n appliquait aucun test
# median, alors que pinpoint3 exige deja COS_PIP (0.75) sur la mediane pour appeler
# un cluster "vraie carte" (pinpoint3.py:45 et 248, flag ghost). Sans ce test une
# vignette de grille devient un pip de consensus : mesure vKMxgHn6P_Q 02/08, cluster
# [0.387,0.5398,0.2333,0.4324] = 7 visages sur 235 frames, cos median 0.59, contre
# 230/235 a cos 0.97 pour le vrai pip bas-gauche. Les scenes dont la mesure locale a
# echoue (box geante) s y accrochaient et peignaient 228 000 px de contenu la ou le
# narrateur n etait pas. Aucun juge ne le voit (doctrine monotone).
_idm = lambda c: float(np.median([m["c"] for m in c["members"]]))
for c in clusters:
    if _idm(c) < COS_PIP:
        print("cluster cx=%.2f cy=%.2f n=%d -> REJETE identite mediane %.3f < %.2f"
              % (c["cx"], c["cy"], len(c["members"]), _idm(c), COS_PIP))
clusters = [c for c in clusters if _idm(c) >= COS_PIP]

# ---- 3. par cluster : carte d'activite temporelle -> box + forme ----
def consensus(c):
    ms = c["members"]
    fw = float(np.median([m["f"][2] for m in ms])); fh = float(np.median([m["f"][3] for m in ms]))
    x0 = max(0, int(min(m["f"][0] for m in ms) - 2.2*fw))
    y0 = max(0, int(min(m["f"][1] for m in ms) - 1.2*fh))
    x1 = min(W, int(max(m["f"][0]+m["f"][2] for m in ms) + 2.2*fw))
    y1 = min(H, int(max(m["f"][1]+m["f"][3] for m in ms) + 3.2*fh))
    ts = [m["t"] for m in ms]
    idx = np.linspace(0, len(ts)-1, min(NPAIRS, len(ts))).astype(int)
    cntF = np.zeros((H, W), np.float32)     # activite PLEIN CADRE (l'anneau peut sortir
    gcntF = np.zeros((H, W), np.float32)          # de la fenetre visage)
    npairs = 0
    for i in idx:
        a = _frame(ts[i]); b = _frame(min(ts[i]+0.4, DUR-0.05))
        if a is None or b is None: continue
        gaF = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        gbF = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        cntF += (cv2.absdiff(gaF, gbF) > ACT_DIFF).astype(np.float32)
        gx = cv2.Sobel(gaF, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(gaF, cv2.CV_32F, 0, 1, ksize=3)
        gcntF += (cv2.magnitude(gx, gy) > 60).astype(np.float32)   # PLEIN CADRE : les
        # bornes de recherche des bords sont proportionnelles au visage, donc trop
        # petites pour une carte large — la fenetre elle-meme cachait la reponse.
        npairs += 1
    if npairs < 10: return None
    actF = cntF / npairs
    act = actF[y0:y1, x0:x1]
    gperF = gcntF / npairs
    gper = gperF[y0:y1, x0:x1]   # persistance de gradient : bord de CARTE = ligne droite
                           # presente sur ~toutes les frames (le contenu bouge, pas elle)
    blob = (act >= ACT_MIN).astype(np.uint8)
    blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    nl, lab, stats, _ = cv2.connectedComponentsWithStats(blob)
    fcx = int(np.median([m["f"][0]+m["f"][2]/2 for m in ms])) - x0
    fcy = int(np.median([m["f"][1]+m["f"][3]/2 for m in ms])) - y0
    best = -1; ba = 0
    for i in range(1, nl):
        lx, ly, lw, lh, area = stats[i]
        if lx <= fcx <= lx+lw and ly <= fcy <= ly+lh and area > ba:
            best = i; ba = area
    if best < 0: return None
    lx, ly, lw, lh, _ = stats[best]
    # EXTENSION AUX BORDS DE CARTE : les zones sombres statiques du pip sont des
    # skip-blocks H.264 (zero diff) -> le blob d'activite = la partie VIVE seulement
    # (eglV : largeur 0.10 vs carte 0.22). Le bord reel = ligne de gradient PERSISTANTE
    # (presente sur ~toutes les frames) la plus forte a moins d'une taille de visage.
    # portee ASYMETRIQUE verticale : le visage vit dans le tiers HAUT de la carte ->
    # le bord bas peut etre a 2.5 hauteurs de visage sous le blob (T-chq h mesuree 0.19
    # vs 0.42 reelle, moitie basse = bureau sombre statique hors de portee de recherche)
    rx = int(1.5*fw); ry = int(1.2*fh); ryb = int(2.5*fh)
    hgt, wdt = gper.shape
    colstr = np.mean(gper[ly:ly+lh, :], axis=0)   # force de ligne verticale par colonne
    rowstr = np.mean(gper[:, lx:lx+lw], axis=1)   # force de ligne horizontale par rangee
    def _best(vals, rng, thr=0.35):
        cand, strength = None, thr
        for p in rng:
            if 0 <= p < len(vals) and float(vals[p]) > strength:
                cand, strength = p, float(vals[p])
        return cand
    eL = _best(colstr, range(max(0, lx-rx), lx))                   # gauche
    if eL is not None: lw += lx-eL; lx = eL
    eR = _best(colstr, range(lx+lw, min(wdt, lx+lw+rx)))           # droite
    if eR is not None: lw = eR+1-lx
    eT = _best(rowstr, range(max(0, ly-ry), ly))                   # haut
    if eT is not None: lh += ly-eT; ly = eT
    eB = _best(rowstr, range(ly+lh, min(hgt, ly+lh+ryb)))          # bas
    if eB is not None: lh = eB+1-ly
    # SYMETRIE : la personne est ~centree horizontalement dans sa cam. Un bord lateral
    # trouve + l'autre invisible (bord de carte sombre-sur-sombre en PERMANENCE, aucun
    # gradient a moyenner — eglV bord droit) -> bord manquant = miroir du bord trouve
    # autour du centre visage, affine par la ligne la plus forte a ±0.35fw (seuil 0.2),
    # sinon miroir sec. Extension seulement, jamais sous le blob.
    fL = eL is not None; fR = eR is not None; fT = eT is not None; fB = eB is not None
    if fL and not fR:
        mr = int(2*fcx - lx)
        if mr > lx+lw:
            e = _best(colstr, range(max(0, mr-int(0.35*fw)), min(wdt, mr+int(0.35*fw))), 0.2)
            lw = min(wdt, (e if e is not None else mr)+1) - lx
            fR = True
    elif fR and not fL:
        ml = int(2*fcx - (lx+lw))
        if ml < lx:
            e = _best(colstr, range(max(0, ml-int(0.35*fw)), min(wdt, ml+int(0.35*fw))), 0.2)
            nl = max(0, e if e is not None else ml)
            lw += lx-nl; lx = nl
            fL = True
    # PLANCHER ancre-visage, PAR COTE SANS BORD DETECTE seulement (un bord mesure fait
    # toujours foi), et SEULEMENT si le blob d'activite est quasi vide (fill < 0.10 :
    # T-chq 0.04). Un blob nourri (og_i 0.23) donne deja une bonne box — le plancher
    # proportionnel au visage EXPLOSAIT les gros visages (og_i 0.34H, marges geantes).
    fillpre = float(np.mean(act[ly:ly+lh, lx:lx+lw] >= ACT_MIN))
    def _pcv(vals, q):
        v = sorted(vals); return v[min(len(v)-1, max(0, int(q*(len(v)-1))))]
    ffx0 = int(_pcv([m["f"][0] for m in ms], 0.10) - 0.6*fw) - x0
    ffx1 = int(_pcv([m["f"][0]+m["f"][2] for m in ms], 0.90) + 0.6*fw) - x0
    ffy0 = int(_pcv([m["f"][1] for m in ms], 0.10) - 0.7*fh) - y0
    ffy1 = int(_pcv([m["f"][1]+m["f"][3] for m in ms], 0.90) + 1.6*fh) - y0
    r0 = lx+lw; b0 = ly+lh
    if fillpre < 0.10:
        if not fL: lx = max(0, min(lx, ffx0))
        if not fR: r0 = min(wdt, max(r0, ffx1))
        if not fT: ly = max(0, min(ly, ffy0))
        if not fB: b0 = min(hgt, max(b0, ffy1))
    lw = r0-lx; lh = b0-ly
    box = [(x0+lx)/W, (y0+ly)/H, lw/W, lh/H]
    # ANNEAU-JUGE (l'oeil de Boss, automatise) : bande adjacente a chaque cote de la box —
    # activite video soutenue dans la bande = pip a decouvert (cheveux au-dessus du vert
    # QPZ, torse sous la box og_i) -> grandir jusqu'au silence. Mesure sur la SOURCE,
    # aucun render requis. Monotone, borne ecran + cap 2.5x par dimension (une demo video
    # qui joue sur la page adjacente ne doit pas avaler l'ecran).
    bx0 = int(box[0]*W); by0 = int(box[1]*H)
    bx1 = bx0+int(box[2]*W); by1 = by0+int(box[3]*H)
    w0 = max(1, bx1-bx0); h0 = max(1, by1-by0)
    band = max(6, int(0.025*min(W, H)))
    def _hotband(xa, xb, ya, yb):
        z = actF[max(0,ya):min(H,yb), max(0,xa):min(W,xb)]
        return z.size > 100 and float(np.mean(z >= 0.30)) > 0.10
    # L anneau suppose la page STATIQUE derriere le pip (voir en-tete). Mesure du 28/07
    # 06h30 : sur un fond ANIME cette hypothese s inverse — XzEg 315-356 (gameplay) a
    # 75.7% de l ecran hors carte qui bouge, PLUS que la carte elle-meme (31.7%). Chaque
    # bande est donc chaude, la box grandit a tous les tours et ne s arrete que sur le cap
    # 2.5x : 314.79-346.46 sortait a 0.165x0.678 alors que la carte mesure 0.062x0.199,
    # soit x3.4 en hauteur, vert sur le gameplay et sur le HUD (frames run_05h00).
    # Fond anime = l anneau n a AUCUN signal, il ne mesure que le bruit du fond. On
    # ABSTIENT : la box reste la carte mesuree — jamais plus grande que la carte.
    # Neutre sur les etalons PAR MESURE, pas par chance : bg_live vaut 0.000 sur 4D7 et
    # 0.000 sur le cluster principal d eglV (21 des 23 min), et sur ces deux-la la boucle
    # ne bougeait deja rien (pre == post). Le seul cluster d etalon au-dessus de 0.25 est
    # eglV n2 (0.285, b-roll atelier) : c est le defaut (B) deja ouvert, une scene SANS
    # carte — probleme de classement, pas de geometrie. Seuil place a 0.40 pour le laisser
    # hors de portee et garder les deux etalons a l identique ; il pourra descendre quand
    # eglV sera reprise, la mesure est deja faite.
    # CORRECTION mesuree A LA FRAME le 28/07 07h00 (le chiffre seul m avait trompe) :
    # sur XzEg 315-356 la carte REELLE vaut x 0.052-0.238 y 0.587-0.863, soit 0.186x0.276.
    # L anneau n est donc pas uniformement faux sur fond anime : il porte la LARGEUR de
    # 0.060 a 0.158 (32% -> 85% de la carte, du bon travail) et la HAUTEUR de 0.264 a
    # 0.665 (96% -> 241%, 58% du vert peint hors carte). L abstention garde la hauteur
    # juste au pixel et laisse la largeur SOUS-COUVERTE a 32%. Erreur totale mesuree en
    # aire d ecran : 0.069 avant, 0.035 apres — moitie moins, et plus jamais de vert hors
    # carte. Ce n est pas la fin : la vraie reponse est que l anneau doit avoir le droit
    # de grandir mais PAS de traverser un bord de carte, c est-a-dire une ligne de gradient
    # persistante (gper est deja calcule juste au-dessus, et cardness sait la prouver).
    # Defaut restant a ouvrir : sous-couverture horizontale sur fond anime.
    bgm = np.ones((H, W), bool); bgm[by0:by1, bx0:bx1] = False
    bg_live = float(np.mean(actF[bgm] >= 0.30))
    for _ in range(0 if bg_live > BG_LIVE_MAX else 10):
        grew = False
        if by0 > 0 and (by1-by0) < 2.5*h0 and _hotband(bx0, bx1, by0-band, by0):
            by0 = max(0, by0-band); grew = True
        if by1 < H and (by1-by0) < 2.5*h0 and _hotband(bx0, bx1, by1, by1+band):
            by1 = min(H, by1+band); grew = True
        if bx0 > 0 and (bx1-bx0) < 2.5*w0 and _hotband(bx0-band, bx0, by0, by1):
            bx0 = max(0, bx0-band); grew = True
        if bx1 < W and (bx1-bx0) < 2.5*w0 and _hotband(bx1, bx1+band, by0, by1):
            bx1 = min(W, bx1+band); grew = True
        if not grew: break
    # EXTENSION LATERALE vers une ligne de carte FORTE. Les deux bornes de la mesure de
    # bord sont proportionnelles au VISAGE (fenetre x0/x1 a 2.2*fw, portee rx a 1.5*fw).
    # Une carte large par rapport au visage a donc ses vrais bords HORS PORTEE, et les
    # bords retenus sont des lignes INTERNES. Mesure itWI5CDVVfQ (2026-08-02) : carte
    # reelle 0..413, fw=67 donc rx=100, fenetre 11..393 (le bord droit n est meme pas
    # dedans), portees 48..147 et 248..347, box sortie 85..335 -> 38.7% de la carte a nu.
    # Un bord de carte est une ligne droite qui traverse TOUTE la hauteur de la carte et
    # survit a ~toutes les frames : on moyenne gper sur la hauteur de BOX (pas sur les
    # rangees du blob) et on exige un seuil ABSOLU. Sur la hauteur de carte itWI5 donne
    # 0.939 en x=413 contre 0.244 au meilleur rival ; sur le tiers central — ce que
    # mesure colstr — le vrai bord tombe a 0.968 contre 0.959, indiscernable.
    # Le RATIO est inutilisable : teste le 2026-08-02, il declenche sur les DEUX etalons
    # (4D7 gauche 0.222 sur un bord a 0.074, eglV droite 0.150 sur 0.084) parce que leurs
    # bords ne sont pas des lignes de gradient. Le seuil absolu les laisse immobiles au
    # chiffre pres (candidats 0.204 / 0.141 / 0.024 / 0.174 / 0.132 / 0.151, tous < 0.60)
    # avec un facteur 3.8 de marge sur le vrai bord. Monotone : n agrandit que vers une
    # ligne MESUREE, borne a une hauteur de carte, jamais au-dela de l ecran.
    # RESTE OUVERT : le cote qui touche le bord d ecran n a AUCUNE ligne a trouver
    # (itWI5 gauche, carte a x=0, meilleur candidat 0.464 rejete a raison) — la bande
    # 0..84 reste a nu. Absence de bord ne prouve pas un bord d ecran, mesure a faire.
    EXT_MIN = 0.60
    if by1 > by0:
        csF = np.mean(gperF[by0:by1, :], axis=0)
        ext = by1-by0
        cL = max(range(max(0, bx0-ext), max(0, bx0-4)), key=lambda i: csF[i], default=None)
        if cL is not None and csF[cL] >= EXT_MIN: bx0 = cL
        cR = max(range(min(W, bx1+4), min(W, bx1+ext)), key=lambda i: csF[i], default=None)
        if cR is not None and csF[cR] >= EXT_MIN: bx1 = cR+1
    lx = bx0-x0; ly = by0-y0; lw = bx1-bx0; lh = by1-by0
    box = [bx0/W, by0/H, lw/W, lh/H]
    # snap bords ecran (<2%)
    if box[0] < 0.02: box[2] += box[0]; box[0] = 0.0
    if box[1] < 0.02: box[3] += box[1]; box[1] = 0.0
    if box[0]+box[2] > 0.98: box[2] = 1.0-box[0]
    if box[1]+box[3] > 0.98: box[3] = 1.0-box[1]
    # forme : sondes de coin MULTI-FRAMES (in/out aux profondeurs 2% et 8%), un coin ne
    # vote que si contraste exterieur/carte a cet instant (dark-on-dark = abstention).
    # Vote majoritaire sur ~20 frames : les moments contrastes decident, le reste se tait.
    bx = x0+lx; by = y0+ly
    d1 = max(2, int(min(lw, lh)*0.02)); d2 = max(6, int(min(lw, lh)*0.08))
    def _pm(img, px, py):
        if px < 4 or py < 4 or px >= W-4 or py >= H-4: return None
        return img[py-2:py+3, px-2:px+3].reshape(-1, 3).mean(axis=0)
    v1 = v2 = valid = 0
    for i in idx[::3]:
        fr = _frame(ts[i])
        if fr is None: continue
        img = fr.astype("float32")
        pe = _pm(img, bx+lw//2, by+lh//2)
        for cxp, cyp, sx, sy in ((bx, by, 1, 1), (bx+lw-1, by, -1, 1),
                                 (bx, by+lh-1, 1, -1), (bx+lw-1, by+lh-1, -1, -1)):
            po = _pm(img, cxp-sx*8, cyp-sy*8)
            a1 = _pm(img, cxp+sx*d1, cyp+sy*d1)
            a2 = _pm(img, cxp+sx*d2, cyp+sy*d2)
            if po is None or pe is None or a1 is None or a2 is None: continue
            n = np.linalg.norm
            if n(po-pe) < 40: continue          # sonde aveugle : pas de vote
            valid += 1
            if n(a1-po)+15 < n(a1-pe): v1 += 1
            if n(a2-po)+15 < n(a2-pe): v2 += 1
    if valid >= 8:
        r1, r2 = v1/valid, v2/valid
        # plus JAMAIS d'ellipse (regle Boss 2026-07-21 : trop de faux positifs) —
        # un pip rond est couvert par un rect arrondi (le rect contient le cercle)
        shape = "rect" if r1 > 0.4 else "rect90"
    else:
        shape = "rect"                          # prior Boss : rond rare, ambigu = rect
    # CLASSIFICATION pip vs NARRATEUR PLEIN ECRAN : un vrai pip a des bords de carte
    # (lignes persistantes trouvees OU box collee aux bords ecran). Un narrateur libre
    # flottant sur son decor n'en a AUCUN + visage central -> kind=hero (ecran vert
    # complet), jamais une box (ADJj 0:08 : oeuf vert sur la tete, corps visible,
    # verdict Boss). Le juge qc_size a le test miroir (presence deborde).
    edges_found = sum(1 for e in (eL, eR, eT, eB) if e is not None)
    if box[0] < 0.02: edges_found += 1
    if box[1] < 0.02: edges_found += 1
    if box[0]+box[2] > 0.98: edges_found += 1
    if box[1]+box[3] > 0.98: edges_found += 1
    ccx = box[0]+box[2]/2
    kind = "hero" if (edges_found <= 1 and 0.30 < ccx < 0.70) else "pip"
    sub = actF[by0:by1, bx0:bx1]      # coords absolues : l'anneau peut sortir de la fenetre
    fill = float(np.mean(sub >= ACT_MIN))
    return {"box": [round(v, 4) for v in box], "shape": shape, "n": len(ms),
            "kind": kind, "npairs": npairs, "fill": round(fill, 2),
            "votes": [valid, round(v1/max(1,valid), 2), round(v2/max(1,valid), 2)]}

out = []
for c in clusters:
    r = consensus(c)
    if r: out.append(r)
    print("cluster cx=%.2f cy=%.2f n=%d ->" % (c["cx"], c["cy"], len(c["members"])),
          r if r else "ECHEC mesure")

# ---- 4. dessin de controle ----
if draw_prefix and out:
    tbest, vbest = None, -1
    for m in clusters[0]["members"][:: max(1, len(clusters[0]["members"])//20)]:
        fr = _frame(m["t"])
        if fr is None: continue
        v = float(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).mean())
        if v > vbest: vbest, tbest = v, m["t"]
    fr = _frame(tbest if tbest is not None else clusters[0]["members"][0]["t"])
    for r in out:
        b = r["box"]
        p1 = (int(b[0]*W), int(b[1]*H)); p2 = (int((b[0]+b[2])*W), int((b[1]+b[3])*H))
        cv2.rectangle(fr, p1, p2, (0, 255, 0), 3)
        cv2.putText(fr, "%s n=%d fill=%.2f" % (r["shape"], r["n"], r["fill"]),
                    (p1[0]+4, max(20, p1[1]-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite(draw_prefix + ".jpg", fr)
    print("image:", draw_prefix + ".jpg")

json.dump(out, open(os.path.join(wd, "box_consensus.json"), "w"), indent=2)
