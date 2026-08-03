#!/usr/bin/env python3
# vf_overcover.py <workdir> [--json out.json] — LE PREMIER JUGE DE SUR-COUVERTURE.
#   Plan Efforts/VF-Plan-Robustesse.md #4. Rapport seul, aucun fixer : la doctrine
#   monotone (« trop-grand jamais flaggé ») etait un desarmement volontaire pour eviter
#   la guerre des juges — ce juge SIGNALE sans corriger, la guerre reste impossible.
#
#   PRINCIPE. Le rendu peint exactement la bbox du host_map. Donc la sur-couverture se
#   juge SANS le rendu : si une bande situee JUSTE A L'INTERIEUR d'un bord de la bbox
#   montre, dans la SOURCE, un champ statique et structure qui CONTINUE de l'autre cote
#   du bord, c'est du CONTENU de page sous le vert — pas une carte. Une carte webcam
#   s'arrete a son bord : son contenu ne continue pas dans la page.
#   Ancre SOURCE, jamais le prior (lecon vKMxg : une mesure ancree sur l'hypothese
#   qu'elle teste ne peut que la confirmer — qc_fid, ancre consensus, valide ses
#   propres boxes fautives depuis 861a5c3).
#
#   TROIS TESTS PAR BANDE, tous exiges pour le palier fort (raffinements imposes par la
#   verification adversariale du plan) :
#     1. STATIQUE   : la bande ne bouge pas sur K paires de frames (une carte webcam est
#                     un flux vivant ; une page l'est rarement — et si la page defile,
#                     l'abstention bg_live la couvre).
#     2. STRUCTUREE : gradient moyen eleve (texte, UI, icones). Le blanc uni de part et
#                     d'autre ne PROUVE rien -> palier SUR_blank, informatif seulement :
#                     l'angle mort whitespace est DECLARE, pas tu.
#     3. CONTINUITE : les statistiques (couleur Lab + densite de gradient) de la bande
#                     interieure ressemblent a celles de la bande exterieure en face.
#                     Un champ de page continue A TRAVERS le bord du vert ; le fond de
#                     piece d'une webcam, non. C'est ce test qui tue les faux positifs
#                     « chambre sombre statique » (murs, etageres — KKniWb9RKq4).
#
#   GARDE-FOUS :
#     - bandes sur le TIERS CENTRAL de chaque cote : les coins arrondis d'un pip rond
#       (eglV) montrent le fond DANS la bbox sans etre un defaut (doctrine rect-only).
#     - abstention totale si le fond hors-bbox est vivant (bg_live > 0.40 : gameplay,
#       b-roll — l'activite du contenu rendrait tout verdict douteux).
#     - cote dont le bord bbox touche le bord ecran : pas de bande exterieure, donc pas
#       de continuite mesurable -> ce cote ne produit que du SUR_blank au mieux.
#     - scenes patched_ident : jamais jugees (priorite absolue des patches identite —
#       un visage qui fuit est plus grave qu'une sur-couverture).
import sys, os, json
import cv2, numpy as np

wd = sys.argv[1]
OUTJSON = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None

hm = json.load(open(os.path.join(wd, "host_map.json")))
cap = cv2.VideoCapture(os.path.join(wd, "source.mp4"))
W = int(cap.get(3)); H = int(cap.get(4))

PAIRS = 8          # paires de frames par scene
DT = 0.4           # ecart d'une paire
HOT = 18           # seuil de difference par pixel
ACT_MAX = 0.06     # bande "statique" si < 6% de pixels chauds en moyenne
GRAD_MIN = 12.0    # bande "structuree" si gradient Sobel moyen au-dela
BG_LIVE_MAX = 0.40
EDGE = 0.01
THICK = max(24, int(0.03 * min(W, H)))
CONT_LAB = 18.0    # continuite : ecart Lab moyen max entre bande in et bande out
CONT_GRAD = 0.5    # ... et rapport de gradients dans [1/(1+r), 1+r]

def frame(t):
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, f = cap.read()
    return f if ok else None

def band_rects(box):
    """[(cote, in_rect, out_rect|None)] en px, tiers central de chaque cote."""
    x0 = int(box[0] * W); y0 = int(box[1] * H)
    x1 = x0 + int(box[2] * W); y1 = y0 + int(box[3] * H)
    cx0 = x0 + (x1 - x0) // 3; cx1 = x1 - (x1 - x0) // 3
    cy0 = y0 + (y1 - y0) // 3; cy1 = y1 - (y1 - y0) // 3
    out = []
    for side, at_edge, r_in, r_out in (
        ("G", x0 <= EDGE * W,       (x0 + 3, cy0, min(x1, x0 + 3 + THICK), cy1),
                                    (max(0, x0 - THICK), cy0, max(1, x0 - 3), cy1)),
        ("D", x1 >= (1 - EDGE) * W, (max(x0, x1 - 3 - THICK), cy0, x1 - 3, cy1),
                                    (min(W - 1, x1 + 3), cy0, min(W, x1 + 3 + THICK), cy1)),
        ("H", y0 <= EDGE * H,       (cx0, y0 + 3, cx1, min(y1, y0 + 3 + THICK)),
                                    (cx0, max(0, y0 - THICK), cx1, max(1, y0 - 3))),
        ("B", y1 >= (1 - EDGE) * H, (cx0, max(y0, y1 - 3 - THICK), cx1, y1 - 3),
                                    (cx0, min(H - 1, y1 + 3), cx1, min(H, y1 + 3 + THICK))),
    ):
        out.append((side, r_in, None if at_edge else r_out))
    return out

def crop(fr, r):
    x0, y0, x1, y1 = r
    if x1 - x0 < 8 or y1 - y0 < 8: return None
    return fr[y0:y1, x0:x1]

def stats(imgs):
    """(activite moyenne, gradient moyen, couleur Lab moyenne) sur une liste de crops
    apparies (a, b) : activite entre paires, gradient et couleur sur a."""
    acts, grads, labs = [], [], []
    for a, b in imgs:
        ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        acts.append(float((cv2.absdiff(ga, cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)) > HOT).mean()))
        gx = cv2.Sobel(ga, cv2.CV_32F, 1, 0); gy = cv2.Sobel(ga, cv2.CV_32F, 0, 1)
        grads.append(float(np.sqrt(gx * gx + gy * gy).mean()))
        labs.append(cv2.cvtColor(a, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(axis=0))
    if not acts: return None
    return (float(np.mean(acts)), float(np.mean(grads)),
            np.mean(np.stack(labs), axis=0))

flags = []
for si, s in enumerate(hm):
    if s.get("host") != "pip" or not s.get("bbox"): continue
    t0, t1 = s["start"], s["end"]; dur = t1 - t0
    if dur < 2.0: continue
    ts = np.linspace(t0 + 0.1 * dur, t1 - 0.1 * dur - DT, min(PAIRS, max(3, int(dur / 4))))
    pairs = []
    for t in ts:
        a = frame(float(t)); b = frame(float(t) + DT)
        if a is not None and b is not None: pairs.append((a, b))
    if len(pairs) < 3: continue

    # abstention : fond hors-bbox vivant (gameplay, b-roll)
    x0 = int(s["bbox"][0] * W); y0 = int(s["bbox"][1] * H)
    x1 = x0 + int(s["bbox"][2] * W); y1 = y0 + int(s["bbox"][3] * H)
    bg_act = []
    for a, b in pairs:
        m = (cv2.absdiff(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY),
                         cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)) > HOT)
        m[y0:y1, x0:x1] = False
        denom = m.size - (y1 - y0) * (x1 - x0)
        bg_act.append(m.sum() / max(1, denom))
    if float(np.mean(bg_act)) > BG_LIVE_MAX:
        continue

    for side, r_in, r_out in band_rects(s["bbox"]):
        ci = [(crop(a, r_in), crop(b, r_in)) for a, b in pairs]
        ci = [(a, b) for a, b in ci if a is not None and b is not None]
        st_in = stats(ci)
        if st_in is None: continue
        act_in, grad_in, lab_in = st_in
        if act_in > ACT_MAX: continue              # bande vivante = carte plausible, rien a dire
        if grad_in < GRAD_MIN:
            # Uni. La continuite s'applique AUSSI ici (mesure de validation : les
            # scenes-panneau N1rAC et les colonnes gnfHl ont leur PERIMETRE dans le
            # padding uni du conteneur — le texte est au milieu, jamais au bord. Le
            # champ uni qui TRAVERSE le bord — creme dedans, creme dehors, meme
            # couleur — est la signature du sur-paint sur page claire). Sans bande
            # exterieure ou sans continuite : SUR_blank simple, informatif.
            tier = "SUR_blank"
            if r_out is not None:
                co = [(crop(a, r_out), crop(b, r_out)) for a, b in pairs]
                co = [(a, b) for a, b in co if a is not None and b is not None]
                st_out = stats(co)
                if st_out is not None:
                    act_out, grad_out, lab_out = st_out
                    if (act_out <= ACT_MAX and grad_out < GRAD_MIN
                            and float(np.linalg.norm(lab_in - lab_out)) <= CONT_LAB):
                        tier = "SUR_cont"
            flags.append(dict(tier=tier, scene=si, t0=round(t0, 2), t1=round(t1, 2),
                              side=side, act=round(act_in, 3), grad=round(grad_in, 1)))
            continue
        if r_out is None: continue
        co = [(crop(a, r_out), crop(b, r_out)) for a, b in pairs]
        co = [(a, b) for a, b in co if a is not None and b is not None]
        st_out = stats(co)
        if st_out is None: continue
        act_out, grad_out, lab_out = st_out
        cont = (act_out <= ACT_MAX
                and float(np.linalg.norm(lab_in - lab_out)) <= CONT_LAB
                and (1.0 / (1.0 + CONT_GRAD)) <= (grad_in / max(grad_out, 1e-6)) <= (1.0 + CONT_GRAD))
        if cont:
            flags.append(dict(tier="SUR_struct", scene=si, t0=round(t0, 2), t1=round(t1, 2),
                              side=side, act=round(act_in, 3), grad=round(grad_in, 1),
                              dlab=round(float(np.linalg.norm(lab_in - lab_out)), 1)))
cap.release()

vid = os.path.basename(os.path.abspath(wd)).replace("wk_b_", "")
strong = [f for f in flags if f["tier"] == "SUR_struct"]
cont = [f for f in flags if f["tier"] == "SUR_cont"]
blank = [f for f in flags if f["tier"] == "SUR_blank"]
# Validation 2026-08-02 sur 5 videos archivees : SUR_cont a UN seul cote = deja la
# signature exacte du defaut (8/8 vrais positifs — les 4 scenes-panneau N1rAC cote G,
# les 4 conteneurs gnfHl cote D — et 0 faux positif sur eglV/KKniW/4D7 ; les marges
# design d'eglV ne declenchent pas la continuite, leur couleur differe de la page).
from collections import Counter
cc = Counter(f["scene"] for f in cont)
cont2 = sorted(s for s, n in cc.items() if n >= 2)
print("%s : %d SUR_struct, %d SUR_cont (%d scenes a >=2 cotes), %d SUR_blank"
      % (vid, len(strong), len(cont), len(cont2), len(blank)))
for f in strong:
    print("  SUR_struct  scene %8.2f-%8.2f cote %s  act=%.3f grad=%.1f dLab=%.1f"
          % (f["t0"], f["t1"], f["side"], f["act"], f["grad"], f["dlab"]))
for f in cont:
    m = "  <<< scene multi-cotes" if f["scene"] in cont2 else ""
    print("  SUR_cont    scene %8.2f-%8.2f cote %s  (uni continu a travers le bord)%s"
          % (f["t0"], f["t1"], f["side"], m))
for f in blank[:4]:
    print("  SUR_blank   scene %8.2f-%8.2f cote %s  (informatif)" % (f["t0"], f["t1"], f["side"]))
if len(blank) > 4: print("  ... +%d SUR_blank" % (len(blank) - 4))
if OUTJSON:
    json.dump({"video": vid, "flags": flags}, open(OUTJSON, "w"), indent=1)
sys.exit(1 if (strong or cont) else 0)
