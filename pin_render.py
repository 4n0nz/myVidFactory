#!/usr/bin/env python3
# pin_render.py <workdir> — construit host_map.json a partir de host_map_pin.json (box stables
# pinpoint) : chaque scene -> segment pip avec masque FORME PROPRE (ellipse/rect-arrondi) dessine
# au box pinpoint + marge de sur-couverture. Forme decidee par grabcut fill_ratio (1 passe).
# Trous entre scenes -> 'off'. Puis build_seg rend.
import sys, os, json, cv2, numpy as np
sys.path.insert(0, "/home/boss/videogen/agent_yt")
sys.path.insert(0, "/home/boss/yolo/scripts")
import webcam_mask, vlm_probe
YUNET = "/home/boss/videogen/face_detection_yunet_2023mar.onnx"
MG = 0.08

wd = sys.argv[1]
pin = json.load(open(os.path.join(wd, "host_map_pin.json")))
src = os.path.join(wd, "source.mp4")
cap = cv2.VideoCapture(src); W=int(cap.get(3)); H=int(cap.get(4)); DUR=cap.get(7)/(cap.get(5) or 30)
yfd = cv2.FaceDetectorYN.create(YUNET, "", (W, H), score_threshold=0.6)
mdir = os.path.join(wd, "masks"); os.makedirs(mdir, exist_ok=True)

def shape_of(t0, t1, box, allow_shrink=True):
    """(shape, box) : ellipse si le blob grabcut remplit ~78% du box (rond), sinon rect.
    fill < 0.45 = la box detectee SUR-couvre largement le blob reel (card_extent qui deborde
    sur page blanche : cercle rendu en rect geant, 0sqC 492-531s) -> RETRECIR la box au
    bounding-box du blob (+marge) et re-decider la forme. allow_shrink=False pour les scenes
    patchees par qc_fix (le shrink annulait leurs elargissements -> boucle QC sterile 6-6-6)."""
    m, _ = webcam_mask.seg_mask(cap, W, H, t0, t1, box, yfd)
    if m is None: return "rect", box
    def _fill(b):
        x=int(b[0]*W);y=int(b[1]*H);w=max(4,int(b[2]*W));h=max(4,int(b[3]*H))
        x=max(0,min(W-w,x));y=max(0,min(H-h,y))
        return float(m[y:y+h, x:x+w].sum())/max(1,w*h)
    # ORDRE : bords REELS de la carte d'abord (_true_rect sur la box canonique). Le shrink
    # au blob grabcut passe APRES et SEULEMENT si aucun bord trouve : sinon il retrecissait
    # la box au buste AU MILIEU d'une vraie carte rect -> sondes de forme dans un referentiel
    # casse (l'exterieur = encore la carte) -> petite ellipse grotesque (w_Px).
    fr = _fill(box)
    # scenes patchees (allow_shrink=False) : box INTOUCHABLE — _true_rect re-retrecissait
    # aux bords de carte, excluant la tete popout que le patch venait couvrir (ADJj LEAK_2)
    tr = _true_rect(box, t0, t1) if allow_shrink else None
    if tr is not None:
        box = tr
    elif fr < 0.45 and allow_shrink:
        ys, xs = np.nonzero(m)
        if len(xs) > 2000:
            mg = 0.05
            bx0,by0 = xs.min()/W, ys.min()/H; bx1,by1 = xs.max()/W, ys.max()/H
            bw,bh = bx1-bx0, by1-by0
            nb = [max(0.0,bx0-bw*mg), max(0.0,by0-bh*mg),
                  min(1.0,bw*(1+2*mg)), min(1.0,bh*(1+2*mg))]
            if nb[2] >= 0.04 and nb[3] >= 0.04:
                box = [round(v,4) for v in nb]
    # FORME mesuree sur la SOURCE : cercle inscrit vide jusqu'a ~14.6% de profondeur de coin,
    # rect arrondi (~10% rayon) jusqu'a ~3%. Sondes 2% et 8%.
    s = _shape_src(box, t0, t1)
    if s is not None: return s, box
    # fallback blob si mesure source pas fiable
    x=int(box[0]*W);y=int(box[1]*H);w=max(8,int(box[2]*W));h=max(8,int(box[3]*H))
    x=max(0,min(W-w,x));y=max(0,min(H-h,y))
    sq=max(4,int(min(w,h)*0.15))
    sub=m[y:y+h, x:x+w]
    cs=[sub[:sq,:sq], sub[:sq,w-sq:], sub[h-sq:,:sq], sub[h-sq:,w-sq:]]
    occ=sum(float(c.mean()) if c.size else 0.0 for c in cs)/4.0
    # PAS d'ellipse au fallback blob : le blob = une personne, jamais dans les coins
    # d'une carte rect -> occ bas meme sur pip rect (4D7). Rond = seulement via sondes
    # source contrastees (_shape_src). Rect couvre un rond entierement = sur-couverture
    # benigne ; l'inverse (rond sur rect) = coins de carte a decouvert.
    if occ > 0.75: return "rect90", box
    return "rect", box

def _true_rect(box, t0, t1):
    """bords REELS de la carte dans la box. La box canonique sur-couvre souvent le fond ->
    sondes de coin hors carte = fausse ellipse + avatar trop grand (o9x8). Scan de la
    transition fond->carte au MILIEU de chaque cote (mediane 3 frames). None si douteux."""
    x = int(box[0]*W); y = int(box[1]*H); w = max(8, int(box[2]*W)); h = max(8, int(box[3]*H))
    x = max(0, min(W-w, x)); y = max(0, min(H-h, y))
    def scan(line):
        if len(line) < 10: return 0
        ref = line[:4].mean(axis=0)
        for i in range(4, len(line)):
            if np.linalg.norm(line[i]-ref) > 35: return i
        return 0
    res = []
    for frac in (0.3, 0.5, 0.7):
        cap.set(cv2.CAP_PROP_POS_MSEC, (t0+(t1-t0)*frac)*1000.0)
        ok, img = cap.read()
        if not ok: continue
        img = img.astype('float32')
        row = img[y+h//2, x:x+w]; col = img[y:y+h, x+w//2]
        res.append((scan(row[:w//2]), scan(row[::-1][:w//2]),
                    scan(col[:h//2]), scan(col[::-1][:h//2])))
    if len(res) < 2: return None
    med = [sorted(r[i] for r in res)[len(res)//2] for i in range(4)]
    L, R, T, B = med
    nw = w-L-R; nh = h-T-B
    if nw < w*0.5 or nh < h*0.5: return None   # scan delirant -> on garde la box
    if L+R+T+B == 0: return None                # deja ajustee
    return [round((x+L)/W,4), round((y+T)/H,4), round(nw/W,4), round(nh/H,4)]

def _narrator_region(box, t0, t1):
    """bbox de la PRESENCE narrateur dans la box, ancree sur le mouvement SOUTENU :
    pixel qui bouge dans >=3 paires sur 5 etalees sur la scene — un narrateur qui parle
    bouge en continu, un scroll de page est transitoire. Mesure sur eglV 90-418s :
    motion bbox [0.036,0.572,0.19,0.38] vs carte reelle [0.03,0.55,0.22,0.40] = fidele,
    alors que le blob grabcut debordait jusqu'a y=1.0 -> blob = FALLBACK seulement.
    Les scans de bords sont aveugles dans une box gonflee pleine de contenu ; pas ca.
    None si presence introuvable ou scene trop courte pour mesurer."""
    x = int(box[0]*W); y = int(box[1]*H); w = max(8, int(box[2]*W)); h = max(8, int(box[3]*H))
    x = max(0, min(W-w, x)); y = max(0, min(H-h, y))
    acc = np.zeros((H, W), np.uint8)
    npairs = 0
    for frac in (0.15, 0.3, 0.5, 0.7, 0.85):
        cap.set(cv2.CAP_PROP_POS_MSEC, (t0+(t1-t0)*frac)*1000.0); ok1, a = cap.read()
        cap.set(cv2.CAP_PROP_POS_MSEC, (t0+(t1-t0)*frac+0.5)*1000.0); ok2, b = cap.read()
        if not (ok1 and ok2): continue
        d = cv2.absdiff(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY))
        acc += (d > 18).astype(np.uint8)
        npairs += 1
    if npairs < 4: return None
    sust = acc[y:y+h, x:x+w] >= 3
    ys, xs = np.nonzero(sust)
    if len(xs) < 2000:
        m, _ = webcam_mask.seg_mask(cap, W, H, t0, t1, box, yfd)
        if m is None: return None
        ys, xs = np.nonzero(m[y:y+h, x:x+w])
        if len(xs) < 2000: return None
    return [(x+xs.min())/W, (y+ys.min())/H, (xs.max()-xs.min()+1)/W, (ys.max()-ys.min()+1)/H]

def _motion_extend(box, t0, t1):
    """etend la box vers les zones ADJACENTES qui bougent en continu (bandes 8%, max 5 pas).
    Attrape ce que les scans de BORDS ne voient pas : torse noir-sur-noir sous la cam (TzJC),
    narrateur libre sans carte. MONOTONE (grandit seulement). Page statique = zero extension."""
    pairs = []
    for frac in (0.3, 0.5, 0.7):
        cap.set(cv2.CAP_PROP_POS_MSEC, (t0+(t1-t0)*frac)*1000.0); ok1, a = cap.read()
        cap.set(cv2.CAP_PROP_POS_MSEC, (t0+(t1-t0)*frac+0.5)*1000.0); ok2, b = cap.read()
        if ok1 and ok2:
            pairs.append(cv2.absdiff(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY),
                                     cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)))
    if not pairs: return box
    dm = np.maximum.reduce(pairs).astype('float32')
    x0, y0 = box[0], box[1]; x1, y1 = box[0]+box[2], box[1]+box[3]
    def _hot(ya, yb, xa, xb):
        band = dm[max(0,int(ya*H)):min(H,int(yb*H)), max(0,int(xa*W)):min(W,int(xb*W))]
        return band.size > 200 and float((band > 18).mean()) > 0.25
    for _ in range(5):
        grown = False
        if y1 < 0.999 and _hot(y1, y1+0.08, x0, x1): y1 = min(1.0, y1+0.08); grown = True
        if y0 > 0.001 and _hot(y0-0.08, y0, x0, x1): y0 = max(0.0, y0-0.08); grown = True
        if x1 < 0.999 and _hot(y0, y1, x1, x1+0.08): x1 = min(1.0, x1+0.08); grown = True
        if x0 > 0.001 and _hot(y0, y1, x0-0.08, x0): x0 = max(0.0, x0-0.08); grown = True
        if not grown: break
    # PLUS DE SNAP AVEUGLE ici (verdict Boss 2026-07-27 : la carte source flotte avec
    # marge gauche+bas, le vert collait aux bords — eglV marge 0.0281 < 0.08 mangee).
    # Toucher un bord = decision de _ring_scene SUR PREUVE (blob personne OU carte-
    # continue), jamais de proximite seule. _motion_extend reste monotone sans snap.
    return [round(x0,4), round(y0,4), round(x1-x0,4), round(y1-y0,4)]

def _pmean(img, px, py):
    h, w = img.shape[:2]
    if px < 0 or py < 0 or px >= w or py >= h: return None
    x0 = max(0, px-2); y0 = max(0, py-2)
    p = img[y0:min(h, py+3), x0:min(w, px+3)]
    return None if p.size == 0 else p.reshape(-1, 3).mean(axis=0)

def _shape_src(box, t0, t1):
    """coin vide a 2% ET 8% de profondeur = cercle ; vide a 2% seulement = rect arrondi ;
    plein aux deux = rect90. Vote 4 coins x 3 frames vs fond exterieur / contenu centre.
    None si pas assez de coins mesurables (box collee aux bords)."""
    x = int(box[0]*W); y = int(box[1]*H); w = max(8, int(box[2]*W)); h = max(8, int(box[3]*H))
    x = max(0, min(W-w, x)); y = max(0, min(H-h, y))
    d1 = max(2, int(min(w, h)*0.02)); d2 = max(6, int(min(w, h)*0.08))
    v1 = 0; v2 = 0; valid = 0
    for frac in (0.3, 0.5, 0.7):
        cap.set(cv2.CAP_PROP_POS_MSEC, (t0+(t1-t0)*frac)*1000.0)
        ok, img = cap.read()
        if not ok: continue
        img = img.astype('float32')
        pe = _pmean(img, x+w//2, y+h//2)
        for cx, cy, sx, sy in ((x, y, 1, 1), (x+w-1, y, -1, 1), (x, y+h-1, 1, -1), (x+w-1, y+h-1, -1, -1)):
            po = _pmean(img, cx-sx*8, cy-sy*8)
            a1 = _pmean(img, cx+sx*d1, cy+sy*d1)
            a2 = _pmean(img, cx+sx*d2, cy+sy*d2)
            if po is None or pe is None or a1 is None or a2 is None: continue
            n = np.linalg.norm
            # sonde AVEUGLE si pas de contraste exterieur/carte a ce coin : pip rect a
            # coins sombres sur fond sombre = "vide" aux 2 profondeurs = fausse ellipse
            # (4D7 rond sur pip rect 90deg, verdict Boss). Pas de contraste = pas de vote.
            if n(po-pe) < 40: continue
            valid += 1
            if n(a1-po)+15 < n(a1-pe): v1 += 1
            if n(a2-po)+15 < n(a2-pe): v2 += 1
    if valid < 6: return None
    # les pips RONDS sont RARES (retour Boss) : ellipse seulement si evidence quasi unanime
    # aux DEUX profondeurs ; tout cas ambigu = rect
    # plus JAMAIS d'ellipse (regle Boss 2026-07-21) : rond couvert par rect arrondi
    if v1*2 >= valid or (v2 >= valid*0.85 and v1 >= valid*0.85): return "rect"
    return "rect90"

def draw(box, shape, idx, mg=MG):
    # MARGE FIDELE (verdict Boss 2026-07-27 : la carte source flotte, le vert collait aux
    # bords et depassait la carte). L'expansion mg par cote est CAPPEE a la moitie de
    # l'ecart au bord d'ecran : une carte a marge design garde une marge visible ; flush
    # seulement si la box touche deja (<=0.5% = touche, clamp plein autorise).
    bx0, by0 = box[0], box[1]; bx1, by1 = box[0]+box[2], box[1]+box[3]
    def _exp(amt, gap):
        if gap <= 0.005: return amt          # touche deja -> expansion libre (clamp ecran)
        return min(amt, gap*0.5)             # marge design -> on n'en mange que la moitie max
    cx0 = bx0 - _exp(box[2]*mg, bx0)
    cy0 = by0 - _exp(box[3]*mg, by0)
    cx1 = bx1 + _exp(box[2]*mg, 1.0-bx1)
    cy1 = by1 + _exp(box[3]*mg, 1.0-by1)
    cw = cx1-cx0; ch = cy1-cy0
    x=int(cx0*W);y=int(cy0*H);w=max(4,int(cw*W));h=max(4,int(ch*H))
    x=max(0,min(W-w,x));y=max(0,min(H-h,y));  w=min(w,W-x); h=min(h,H-y)
    out=np.zeros((h,w),np.uint8)
    if shape=="ellipse":
        cv2.ellipse(out,(w//2,h//2),(w//2-1,h//2-1),0,0,360,255,-1)
    elif shape=="rect90":
        # coins source a 90 degres -> rect sec, pas d'arrondi (regle Boss)
        cv2.rectangle(out,(0,0),(w,h),255,-1)
    else:
        r=max(2,int(min(w,h)*0.10))
        cv2.rectangle(out,(r,0),(w-r,h),255,-1);cv2.rectangle(out,(0,r),(w,h-r),255,-1)
        for a,b in ((r,r),(w-r,r),(r,h-r),(w-r,h-r)): cv2.circle(out,(a,b),r,255,-1)
        # coins colles a un bord d'ECRAN = CARRES : la carte y est tronquee par l'ecran,
        # l'arrondi du masque laissait un sliver au coin (og_i chemise coin bas-droit)
        eL = x <= 1; eT = y <= 1; eR = x+w >= W-1; eB = y+h >= H-1
        if eL or eT: out[:r, :r] = 255
        if eR or eT: out[:r, w-r:] = 255
        if eL or eB: out[h-r:, :r] = 255
        if eR or eB: out[h-r:, w-r:] = 255
    out=cv2.GaussianBlur(out,(5,5),0)
    p=os.path.join(mdir,"seg_%04d.png"%idx); cv2.imwrite(p,out)
    return p,[round(x/W,4),round(y/H,4),round(w/W,4),round(h/H,4)]

_narr_ref = None
_sface = None
def _narr_match(fr):
    """Vrai si un visage de la frame EST le narrateur (SFace cos >= 0.363, meme seuil
    que qc_ident). Sert de gate identite a la promotion hero des scenes src=ident."""
    global _narr_ref, _sface
    if fr is None: return False
    p = os.path.join(wd, 'narrator_feat.npy')
    if _narr_ref is None:
        if not os.path.exists(p): return True  # pas de reference narrateur -> pas de gate
        _narr_ref = np.load(p)
        _sface = cv2.FaceRecognizerSF.create('/home/boss/videogen/face_recognition_sface_2021dec.onnx', '')
    _, faces = yfd.detect(fr)
    if faces is None: return False
    for f in faces:
        if f[3]/H < 0.045: continue
        try:
            feat = _sface.feature(_sface.alignCrop(fr, f)).flatten().astype(np.float32)
        except Exception:
            continue
        c = float(np.dot(feat, _narr_ref)/(np.linalg.norm(feat)*np.linalg.norm(_narr_ref)+1e-9))
        # narrateur reconnu ET visage PLEIN CADRE. Le cos seul ne suffit pas : une PHOTO
        # du narrateur dans une carte de contenu matche aussi (photo de groupe eglV t=33 :
        # cos 0.58-0.69 mais visage 0.099-0.11 H). Un vrai hero live a un visage 0.368-0.415 H
        # (mesures eglV 39/41/908/1388/1393 ; meme echelle que qc_ident : plans serres >=0.40,
        # contenu <=0.31). Seuil 0.30 = marge des deux cotes.
        if c >= 0.363 and float(f[3])/H >= 0.30: return True
    return False

def is_hero(t0, t1, box, need_ident=False):
    """HERO = grosse cam qui remplit le cadre -> avatar plein ecran. VLM fullface x2 + garde
    geometrique (la box couvre une grande part de l'ecran : large ET haute).
    need_ident (scenes src=ident, colonnes h>0.9) : exige EN PLUS le narrateur reconnu
    SFace sur la frame — un fullface VLM sur un visage de CONTENU (photo de groupe en
    gros plan, eglV t=33) promouvait la scene hero = vert plein cadre par-dessus une
    carte a marges creme, sur-couverture invisible aux juges (doctrine monotone)."""
    bw, bh = box[2], box[3]
    big = bw > 0.5 and bh > 0.7           # box deja quasi plein cadre
    mid = (t0+t1)/2
    cap.set(cv2.CAP_PROP_POS_MSEC, mid*1000.0); ok, fr = cap.read()
    fr_mid = fr if ok else None
    ff1 = vlm_probe.fullface(fr) if ok else None
    if ff1 is not True:
        hero = big and ff1 is not False
        fr_alt = None
    else:
        cap.set(cv2.CAP_PROP_POS_MSEC, (t0+(t1-t0)*0.25)*1000.0); ok, fr = cap.read()
        fr_alt = fr if ok else None
        hero = vlm_probe.fullface(fr_alt) is True if fr_alt is not None else False
    if hero and need_ident:
        hero = _narr_match(fr_mid) or (fr_alt is not None and _narr_match(fr_alt))
    return hero

def _narr_live(t0, t1):
    """Vrai si le narrateur LIVE plein cadre est visible dans la scene (3 sondes).
    Gate des promotions pip->hero tardives : _motion_extend gonflait une CARTE de
    contenu quasi pleine frame (photo de groupe eglV 32-34, marges creme) a area>0.85
    -> promotion 'narrateur libre' sans aucun test d'identite -> vert plein cadre.
    Meme regle que le gate is_hero : reconnu SFace ET visage >=0.30 H."""
    for f in (0.5, 0.25, 0.75):
        cap.set(cv2.CAP_PROP_POS_MSEC, (t0+(t1-t0)*f)*1000.0)
        ok, fr = cap.read()
        if ok and _narr_match(fr): return True
    return False

segs=[]; prev=0.0
nhero=0
pips=[]
for sc in pin:
    if sc["start"]>prev+0.3:
        segs.append({"host":"off","start":round(prev,2),"end":round(sc["start"],2),"bbox":None})
    # src=ident : promotion hero DETERMINISTE par _narr_live (SFace calibre, cos>=0.363
    # et visage >=0.30 H). Le verdict VLM fullface est stochastique : (40,43)/(1391,95)
    # flippaient hero<->pip d une passe a l autre (A/B 27/07 05h). VLM reste pour les
    # sources sans narrator_feat.
    if sc.get("region") == "hero" or sc["box"][2]*sc["box"][3] > 0.85 \
            or (sc.get("src") == "ident"
                and ((sc["box"][3] > 0.9 and sc["box"][2] > 0.35)
                     or sc["box"][2]*sc["box"][3] > 0.5)
                and _narr_live(sc["start"], sc["end"])) \
            or (sc.get("src") != "ident"
                and is_hero(sc["start"], sc["end"], sc["box"])):
        segs.append({"host":"hero","start":round(sc["start"],2),"end":round(sc["end"],2),"bbox":None})
        nhero+=1
    else:
        seg={"host":"pip","start":round(sc["start"],2),"end":round(sc["end"],2)}
        segs.append(seg)
        pips.append({"seg":seg,"t0":sc["start"],"t1":sc["end"],"box":sc["box"],
                     "src":sc.get("src"),
                     "patched":bool(sc.get("patched")) or sc.get("region")=="patch",
                     "pident":bool(sc.get("patched_ident")) or sc.get("region")=="patch",
                     "pkeep":bool(sc.get("patched_keep"))})
    prev=sc["end"]

# CONSENSUS TEMPOREL (box_consensus.py, lance par le batch apres pinpoint3) : box+forme
# mesurees par cluster de position sur TOUTE la video (activite temporelle + bords
# persistants + anneau-juge). Prioritaire sur toute la tour heuristique par scene.
_cons = []
_cpath = os.path.join(wd, "box_consensus.json")
if os.path.exists(_cpath):
    try: _cons = json.load(open(_cpath))
    except Exception: _cons = []
def _cons_match(b):
    # match par centres proches OU par CONTENANCE : une box de scene gonflee (union
    # pinpoint3 cam+previews, AAmd 0.79x0.84) a son centre loin du cluster mais le
    # CONTIENT — le consensus doit quand meme la remplacer. Plus proche si plusieurs.
    cx = b[0]+b[2]/2; cy = b[1]+b[3]/2
    best = None; bd = 9.0
    for c in _cons:
        # cluster FAIBLE (votes nuls = jamais confirme par le vote multi-mesures,
        # ex eglV cluster 2 [0.24,0,0.55,1.0] votes [0,0,0] n=31) : PAS d autorite —
        # la scene retombe sur la mesure de carte reelle (b-roll 32-38 coupe par la
        # colonne du faux cluster, verdict Boss 27/07). Les vrais clusters votent
        # (eglV [80,0.97,0.68], 4D7 stable).
        _v = c.get("votes") or [0, 0.0, 0.0]
        if not any(_v): continue
        cb = c["box"]; ccx = cb[0]+cb[2]/2; ccy = cb[1]+cb[3]/2
        near = abs(cx-ccx) < 0.15 and abs(cy-ccy) < 0.15
        inside = (b[0]-0.02 <= ccx <= b[0]+b[2]+0.02
                  and b[1]-0.02 <= ccy <= b[1]+b[3]+0.02)
        if near or inside:
            d = abs(cx-ccx)+abs(cy-ccy)
            if d < bd: best, bd = c, d
    return best

# forme + shrink par scene (cache par box canonique non-shrinkee)
_shape_cache={}
for p in pips:
    # patched_keep (petit ajustement de bord geom sur scene consensus) : box TELLE QUELLE
    # — c'est deja consensus ∪ carte mesuree. La passer au legacy la regonflait via
    # motion_extend (Id9G 16-19s : bord ajuste -> colonne pleine hauteur)
    if p.get("pkeep") and not p.get("pident"):
        c = _cons_match(p["box"])
        p["shape"] = (c["shape"] if c is not None else "rect")
        p["abox"] = list(p["box"]); p["cons"] = True
        continue
    # consensus AUTORITAIRE sur les patches GEOM (true_rect surestime sur fond sombre,
    # 4D7 verdict Boss) ; les patches IDENTITE (vrai visage qui fuit) gardent priorite
    if not p.get("pident"):
        c = _cons_match(p["box"])
        if c is not None:
            _big = c.get("kind") == "hero" or c["box"][2]*c["box"][3] > 0.85
            if _big and (p.get("src") != "ident" or _narr_live(p["t0"], p["t1"])):
                # cluster narrateur plein ecran (aucun bord de carte, visage central)
                # -> HERO complet, jamais une box sur la tete (ADJj 0:08, verdict Boss).
                # Scenes ident : narrateur LIVE exige (une carte de contenu peut couvrir
                # l'ecran aussi — la couvrir en pip, pas la remplacer par l'avatar).
                p["seg"]["host"] = "hero"; p["seg"]["bbox"] = None; p["hero"] = True
                p["shape"], p["abox"] = "rect90", list(c["box"])
                continue
            if not _big:
                p["shape"], p["abox"] = c["shape"], list(c["box"])
                p["cons"] = True
                continue
            # cluster geant BLOQUE (scene ident, narrateur pas live) : appliquer sa box
            # mangerait le contenu (verts geants photos 32-38 / screencasts 1365-78 eglV,
    # frames 27/07 05h) — retomber sur la mesure locale de la scene
    key=tuple(p["box"])
    if key in _shape_cache and not p["patched"]:
        p["shape"],p["abox"]=_shape_cache[key]
    else:
        shp,abox=shape_of(p["t0"],p["t1"],p["box"],allow_shrink=not p["patched"])
        if p["patched"] and shp == "ellipse":
            shp = "rect"   # patch anti-fuite : l'ellipse ne couvre pas les coins de l'union
                           # (pLos popout : tete qui depasse du cercle -> boucle QC sterile)
        abox2 = _motion_extend(list(abox), p["t0"], p["t1"])
        if abox2[2]*abox2[3] > 0.85 or (p.get("pident") and abox2[2]*abox2[3] > 0.5):
            if p.get("src") != "ident" or _narr_live(p["t0"], p["t1"]):
                # l'extension revele un corps quasi plein cadre -> narrateur libre -> hero.
                p["seg"]["host"] = "hero"; p["seg"]["bbox"] = None
                p["hero"] = True
                p["shape"], p["abox"] = "rect90", abox2
                continue
            # extension geante SANS narrateur live : le motion venait du CONTENU
            # (transitions photos, scroll screencast — verts geants eglV, frames 27/07
            # 05h). Garder la mesure locale ; qc_ident rattrape toute fuite reelle.
        else:
            abox = abox2
        p["shape"],p["abox"]=shp,list(abox)
        if abox == p["box"] and not p["patched"]:
            _shape_cache[key]=(shp,list(abox))

# UNIFICATION position : le MEME pip source recoit LA MEME box finale partout.
# Sans ca, une scene garde la canonique et l'autre la version shrinkee -> saut de forme
# a la frontiere + bord du pip a decouvert (0sqC 485s, invisible au QC : pas un visage).
# Les patches GEOM (non-pident) ENTRENT dans l'union : exclus, chaque tour correctif
# laissait sa box propre par scene -> 5 masques uniques pour 1 pip physique (4D7 :
# w=0.225/0.26/0.213, SAUT-DE-BOX t=11). Union monotone = chaque patch reste couvert.
# Le matching ignore la forme (un patch geom force rect, le consensus dit rect90 ->
# le meme pip physique se splittait en 2 groupes jamais unifies) ; forme finale =
# celle du membre consensus si present, sinon celle du membre le plus long.
def _ctr(b): return (b[0]+b[2]/2, b[1]+b[3]/2)
groups=[]
for p in pips:
    if p.get("pident") or p.get("hero"): continue
    cx,cy=_ctr(p["abox"]); hit=None
    for g in groups:
        if abs(cx-g["cx"])<0.06 and abs(cy-g["cy"])<0.06:
            hit=g; break
    if hit is None:
        groups.append({"cx":cx,"cy":cy,"box":list(p["abox"]),"members":[p]})
    else:
        b=hit["box"]; nb=p["abox"]
        x0=min(b[0],nb[0]); y0=min(b[1],nb[1])
        x1=max(b[0]+b[2],nb[0]+nb[2]); y1=max(b[1]+b[3],nb[1]+nb[3])
        hit["box"]=[x0,y0,x1-x0,y1-y0]; hit["members"].append(p)
        hit["cx"],hit["cy"]=_ctr(hit["box"])
for g in groups:
    cons_m=[p for p in g["members"] if p.get("cons")]
    ref=cons_m[0] if cons_m else max(g["members"], key=lambda p: p["t1"]-p["t0"])
    g["shape"]=ref["shape"]
    for p in g["members"]:
        p["abox"]=[round(float(v),4) for v in g["box"]]
        p["shape"]=g["shape"]

# CAP TROP-GRAND terminal — UNE passe apres toute la chaine de croissance (percentiles
# pinpoint + motion_extend + unions), JAMAIS en boucle donc pas d'oscillation possible.
# Groupes avec membre patche : JAMAIS de cap (le cap resserre, un patch resserre =
# la fuite corrigee revient = oscillation). Box finale > 1.25x la presence narrateur
# reelle -> resserree a cette presence (+4% de marge), forme re-decidee dans le nouveau
# referentiel (avatar 2x trop grand : 0sq/eglV/ADJj, verdicts Boss 2026-07-20).
ncap = 0
for g in groups:
    if any(p.get("cons") or p["patched"] for p in g["members"]):
        continue   # consensus deja valide / patch monotone : ne jamais resserrer
    p0 = max(g["members"], key=lambda p: p["t1"]-p["t0"])
    reg = _narrator_region(g["box"], p0["t0"], p0["t1"])
    if reg is None: continue
    mg = 0.04
    nb = [max(0.0, reg[0]-reg[2]*mg), max(0.0, reg[1]-reg[3]*mg),
          min(1.0, reg[2]*(1+2*mg)), min(1.0, reg[3]*(1+2*mg))]
    # clamp DANS la box existante (le cap resserre, il n'etend jamais)
    x0 = max(nb[0], g["box"][0]); y0 = max(nb[1], g["box"][1])
    x1 = min(nb[0]+nb[2], g["box"][0]+g["box"][2]); y1 = min(nb[1]+nb[3], g["box"][1]+g["box"][3])
    if x1-x0 < 0.04 or y1-y0 < 0.04: continue
    nb = [round(x0,4), round(y0,4), round(x1-x0,4), round(y1-y0,4)]
    # 1.25 et pas 1.6 : une box correcte vs sa presence narrateur ratio ~1.05 (eglV) ;
    # a 1.6 le cas reel 1.4x passait. Les slivers de carte decouverts par un resserrage
    # trop zele sont rattrapes par qc_geom SOUS-COUVERTURE (union monotone vers la carte)
    if g["box"][2]*g["box"][3] <= 1.25*nb[2]*nb[3]: continue
    s = _shape_src(nb, p0["t0"], p0["t1"])
    ncap += 1
    for p in g["members"]:
        p["abox"] = list(nb)
        if s is not None: p["shape"] = s
    g["box"] = list(nb)
if ncap: print("cap trop-grand: %d groupes resserres" % ncap)

def _ring_scene(box, t0, t1):
    """ANNEAU-JUGE par SCENE : bande adjacente a la box en mouvement SOUTENU (>=3/5
    paires sur la duree de LA scene) = presence narrateur a decouvert -> etendre.
    Attrape les layouts de scene qui divergent du cluster (Id9G 16-19s : narrateur
    FLOUTE en fond de colonne droite, tete au-dessus de la box cluster — aucun visage
    detectable, ident aveugle, seule l'activite le voit). Monotone, cap 2.5x/dim."""
    acc = np.zeros((H, W), np.uint8); n = 0; fs = []; dms = []
    for frac in (0.15, 0.3, 0.5, 0.7, 0.85):
        cap.set(cv2.CAP_PROP_POS_MSEC, (t0+(t1-t0)*frac)*1000.0); ok1, a = cap.read()
        cap.set(cv2.CAP_PROP_POS_MSEC, (t0+(t1-t0)*frac+0.4)*1000.0); ok2, b = cap.read()
        if not (ok1 and ok2): continue
        fs.append(a)
        _d = cv2.absdiff(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY),
                         cv2.cvtColor(b, cv2.COLOR_BGR2GRAY))
        dms.append(_d.astype(np.float32))
        acc += (_d > 5).astype(np.uint8)
        n += 1
    if n < 4: return box
    # GARDE-SCROLL : si la frame bouge GLOBALEMENT (mediane >10% de pixels vifs =
    # scroll/animation/b-roll), aucune bande n est mesurable -> pas d extension.
    # (eglV intro 5.5-9 : 18% global, bandes a 25-34 d activite pure animation ;
    # scene stable 43-906 : 4%. Verdict Boss 27/07, marge mangee par l intro.)
    _ga = sorted(float((d > 5).mean()) for d in dms)[len(dms)//2] if dms else 0.0
    if _ga > 0.10: return box
    sust = acc >= 3
    def _alive(ya, yb, xa, xb):
        # bande VIVANTE = contenu video (carte) ; statique = marge design (2ef79aa :
        # fantomes 0.0, pire vrai bord 1.56 -> seuil 0.5). Sans vie, carte-continue
        # est aveugle au cas carte-blanche-sur-marge-creme (eglV, verdict Boss 27/07).
        vals = []
        for d in dms:
            z = d[max(0,ya):min(H,yb), max(0,xa):min(W,xb)]
            if z.size: vals.append(float(z.mean()))
        # seuil 1.0 : ombres/bruit sous la carte mesures 0.3-0.88 (eglV bande bas),
        # vraie personne calme 1.67 (4D7), torse og_i >5. 0.5 laissait passer l ombre.
        return len(vals) >= 3 and sorted(vals)[len(vals)//2] > 1.0
    bx0 = int(box[0]*W); by0 = int(box[1]*H)
    bx1 = bx0+int(box[2]*W); by1 = by0+int(box[3]*H)
    w0 = max(1, bx1-bx0); h0 = max(1, by1-by0)
    band = max(6, int(0.025*min(W, H)))
    # DIRECTIONS AUTORISEES : extension seulement VERS un bord d'ecran proche (<25%).
    # Les vrais deborde-box (torse og_i/TzJC, colonne Id9G, bas eglV) vont tous vers un
    # bord d'ecran ; une extension vers le CENTRE = grabcut/contenu qui bave (0sq :
    # cluster n=587 parfait regonfle 2x vers le centre, verdict Boss). Jamais vers l'interieur.
    aT = by0 < 0.25*H; aB = (H-by1) < 0.25*H
    aL = bx0 < 0.25*W; aR = (W-bx1) < 0.25*W
    def hot(xa, xb, ya, yb):
        # frac de pixels vifs (>0.12) ET amplitude vivante (_alive) : le flicker
        # d ombre sous une carte est ETENDU mais FAIBLE (0.45) -> bloque ; un torse/
        # narrateur bouge fort (>1.67). (eglV bas de carte tire a 1.0, Boss 27/07)
        z = sust[max(0,ya):min(H,yb), max(0,xa):min(W,xb)]
        return z.size > 100 and float(z.mean()) > 0.12 and _alive(ya, yb, xa, xb)
    for _ in range(10):
        grew = False
        if aT and by0 > 0 and (by1-by0) < 2.5*h0 and hot(bx0, bx1, by0-band, by0):
            by0 = max(0, by0-band); grew = True
        if aB and by1 < H and (by1-by0) < 2.5*h0 and hot(bx0, bx1, by1, by1+band):
            by1 = min(H, by1+band); grew = True
        if aL and bx0 > 0 and (bx1-bx0) < 2.5*w0 and hot(bx0-band, bx0, by0, by1):
            bx0 = max(0, bx0-band); grew = True
        if aR and bx1 < W and (bx1-bx0) < 2.5*w0 and hot(bx1, bx1+band, by0, by1):
            bx1 = min(W, bx1+band); grew = True
        if not grew: break
    # UNION BLOB : un torse STATIQUE en vetement uni ne bouge pas assez pour l anneau
    # (og_i chemise blanche visible sous le vert, tous les juges aveugles : tete couverte
    # = pas de visage, presence=motion = contain OK). Le grabcut segmente le statique.
    # Union bornee par le meme cap 2.5x.
    gs = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in fs]
    def _med(v): return sorted(v)[len(v)//2]
    def _cont_v(xe, lo, hi):
        dcs, vfs = [], []
        for f, g in zip(fs, gs):
            a = f[by0:by1, max(0, xe-8):xe-2].astype(np.float32)
            b = f[by0:by1, xe+2:min(W, xe+8)].astype(np.float32)
            if not a.size or not b.size: continue
            dcs.append(float(np.abs(a.mean(axis=1)-b.mean(axis=1)).mean(axis=0).max()))
            x0, x1 = max(1, lo), min(W-1, hi)
            d = np.abs(g[by0:by1, x0+1:x1+1]-g[by0:by1, x0-1:x1-1])
            vfs.append(float((d > 20).mean(axis=0).max()) if d.size else 1.0)
        return len(dcs) >= 3 and _med(dcs) < 32 and _med(vfs) < 0.45
    def _cont_h(ye, lo, hi):
        dcs, vfs = [], []
        for f, g in zip(fs, gs):
            a = f[max(0, ye-8):ye-2, bx0:bx1].astype(np.float32)
            b = f[ye+2:min(H, ye+8), bx0:bx1].astype(np.float32)
            if not a.size or not b.size: continue
            dcs.append(float(np.abs(a.mean(axis=0)-b.mean(axis=0)).mean(axis=0).max()))
            y0, y1 = max(1, lo), min(H-1, hi)
            d = np.abs(g[y0+1:y1+1, bx0:bx1]-g[y0-1:y1-1, bx0:bx1])
            vfs.append(float((d > 20).mean(axis=1).max()) if d.size else 1.0)
        return len(dcs) >= 3 and _med(dcs) < 32 and _med(vfs) < 0.45
    m, _ = webcam_mask.seg_mask(cap, W, H, t0, t1, [bx0/W, by0/H, (bx1-bx0)/W, (by1-by0)/H], yfd)
    if m is not None:
        ys, xs = np.nonzero(m)
        if len(xs) > 2000:
            # union blob CLAMPEE aux directions autorisees (vers bords d'ecran seulement)
            # chaque cote franchi exige CARTE-CONTINUE au bord de box : un blob
            # statique DANS la carte (og_i chemise) traverse un faux bord (video des
            # deux cotes -> continue) ; une bave grabcut sur la marge design traverse
            # le VRAI bord de carte (video->creme, dc enorme) -> bloquee. Boss 27/07.
            nx0 = min(bx0, int(xs.min())) if (aL and int(xs.min()) < bx0 and _cont_v(bx0, 2, bx0+5)) else bx0
            ny0 = min(by0, int(ys.min())) if (aT and int(ys.min()) < by0 and _cont_h(by0, 2, by0+5)) else by0
            nx1 = max(bx1, int(xs.max())+1) if (aR and int(xs.max())+1 > bx1 and _cont_v(bx1, bx1-5, W-2)) else bx1
            ny1 = max(by1, int(ys.max())+1) if (aB and int(ys.max())+1 > by1 and _cont_h(by1, by1-5, H-2)) else by1
            if (nx1-nx0) <= 2.5*w0 and (ny1-ny0) <= 2.5*h0:
                bx0, by0, bx1, by1 = nx0, ny0, nx1, ny1
    # PLUS AUCUN SNAP AVEUGLE (verdict Boss : les avatars collaient presque toujours a
    # 1-2 bords alors que le pip original garde sa marge). Extension au bord sur
    # PREUVE, bande restante <10% ecran : (a) pixels de personne (blob — og_i chemise
    # dans la bande 0.93-1.0) OU (b) CARTE-CONTINUE (verdict Boss 21h09 : le blob ne
    # voit que la PERSONNE — un coin de carte sans personne dedans restait un sliver,
    # 4D7 bande droite 0.955-1.0 = fauteuil/mur du narrateur, blob 0). Carte-continue
    # = continuite couleur a travers le bord de box (diff moyenne par rangee/colonne,
    # max canal : sliver mesure 22-26 vs vraie frontiere/marge 40-65 -> seuil 32) ET
    # aucune ligne franche au bord ni dans la bande (frac de pixels a gradient
    # transversal >20 par colonne/rangee : sliver <=0.40 vs bord de carte 0.51-0.77
    # -> seuil 0.45), mediane sur les frames de la scene. Un pip a marge design garde
    # sa marge (eglV gauche 2.8% : dc 57-64 -> bloque), une carte qui touche s'etend.
    def _blob(z): return z.size > 100 and float(z.mean()) > 0.05
    # les DEUX preuves (blob personne, carte-continue) exigent une bande VIVANTE :
    # grabcut bave sur une marge design statique et la declare personne (eglV creme,
    # verdict Boss 27/07) ; une vraie personne/carte video bouge (og_i chemise, 4D7 mur).
    if 0 < H-by1 < 0.10*H and _alive(by1, H, bx0, bx1) and ((m is not None and _blob(m[by1:H, bx0:bx1])) or _cont_h(by1, by1-5, H-2)): by1 = H
    if 0 < by0 < 0.10*H and _alive(0, by0, bx0, bx1) and ((m is not None and _blob(m[0:by0, bx0:bx1])) or _cont_h(by0, 2, by0+5)): by0 = 0
    if 0 < W-bx1 < 0.10*W and _alive(by0, by1, bx1, W) and ((m is not None and _blob(m[by0:by1, bx1:W])) or _cont_v(bx1, bx1-5, W-2)): bx1 = W
    if 0 < bx0 < 0.10*W and _alive(by0, by1, 0, bx0) and ((m is not None and _blob(m[by0:by1, 0:bx0])) or _cont_v(bx0, 2, bx0+5)): bx0 = 0
    return [round(bx0/W,4), round(by0/H,4), round((bx1-bx0)/W,4), round((by1-by0)/H,4)]

# anneau par scene sur TOUTES les box pip finales (consensus, pkeep, legacy)
for p in pips:
    if p.get("hero"): continue
    p["abox"] = _ring_scene(list(p["abox"]), p["t0"], p["t1"])

# RE-UNIFICATION post-anneau : l'anneau par scene etend selon le mouvement LOCAL de
# chaque scene -> micro-divergences sur le MEME pip physique (4D7 : 0.213/0.225/0.26,
# SAUT-DE-BOX t=11 au retour de hero) qui defont l'unification faite plus haut.
# Petite extension (aire <=1.35x la box commune du groupe) = bruit de mesure -> union
# appliquee a tout le groupe (monotone : chaque extension reste couverte). Grosse
# divergence (>1.35x) = vrai layout de scene (Id9G 16-19s colonne narrateur floute,
# ~2.5x) -> la scene garde sa box propre, on ne propage pas une colonne au groupe.
for g in groups:
    ga = g["box"][2]*g["box"][3]
    small = [p for p in g["members"] if not p.get("hero")
             and p["abox"][2]*p["abox"][3] <= 1.35*ga]
    if len(small) < 2: continue
    # les scenes COURTES (<3 s = transitions/scrolls) ne VOTENT pas dans l union :
    # pendant un scroll tout est vivant, leur anneau s etend jusqu aux bords et
    # propageait x=0 aux 900 s stables du groupe (eglV marge mangee, verdict Boss
    # 27/07). Elles RECOIVENT la box du groupe (couvertes), sans la dicter.
    voters = [p for p in small if p["t1"] - p["t0"] >= 3.0] or small
    x0 = min(p["abox"][0] for p in voters); y0 = min(p["abox"][1] for p in voters)
    x1 = max(p["abox"][0]+p["abox"][2] for p in voters)
    y1 = max(p["abox"][1]+p["abox"][3] for p in voters)
    ub = [round(x0,4), round(y0,4), round(x1-x0,4), round(y1-y0,4)]
    for p in small: p["abox"] = list(ub)

# dessin : UN masque par (box finale, forme)
_mask={}; mi=0
for p in pips:
    if p.get("hero"): continue
    k=(tuple(p["abox"]),p["shape"],bool(p.get("cons")))
    if k not in _mask:
        _mask[k]=draw(p["abox"],p["shape"],mi,mg=0.02 if p.get("cons") else MG); mi+=1
    mp,bb=_mask[k]
    p["seg"].update({"bbox":bb,"mask":mp,"shape":p["shape"]})
print("heros: %d / masques uniques: %d / groupes position: %d" % (nhero, len(_mask), len(groups)))
if prev<DUR-0.3:
    segs.append({"host":"off","start":round(prev,2),"end":round(DUR,2),"bbox":None})

json.dump(segs,open(os.path.join(wd,"host_map.json"),"w"),indent=2)
print("host_map: %d segs (%d pip) genere depuis pinpoint"%(len(segs),sum(1 for s in segs if s["host"]=="pip")))
cap.release()
