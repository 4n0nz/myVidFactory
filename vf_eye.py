#!/usr/bin/env python3
# vf_eye.py <workdir> [--probes-long N] [--limit N] [--json out.json]
#   L'OEIL : regarde ce que Boss regarde, automatiquement. Verifie la doctrine des bords
#   DANS LES DEUX SENS, scene pip par scene pip, par oracle VLM local (gemma3:12b).
#
# POURQUOI CE FICHIER (2026-08-02, question Boss « je fais juste te copier coller le
# screenshot alors je comprend pas pkoi tu a besoin de moi ?! ») : tous les juges
# mesurent AUTOUR de l'image (visages, gradients, aires) mais personne ne la REGARDE.
# Chaque defaut de la famille dominante (le prior mesure le CONTENEUR : o9x8 fenetre
# navigateur, gnfHl colonne layout, N1rAC panneau UI, itWI/w_Px4 bord ecran) a ete
# trouve par l'oeil de Boss, jamais par un juge.
#
# DESIGN — constat vlm_probe 2026-07-17, reutilise de vlm_box.py : « box VLM directe =
# pas fiable ; CROP + question FERMEE = fiable ». Le VLM avait ete retrograde des
# decisions parce que ses verdicts ouverts (hero/pip) flippaient d'une passe a l'autre.
# Ici : questions fermees sur des BANDES, temperature 0, et role de DETECTEUR seulement
# — un flag de l'oeil pointe la mesure au bon endroit, il ne decide rien. La regle
# « defaut confirme aux frames avant tout fix » reste entiere.
#
# L'ASTUCE QUI EVITE DE MESURER LA CARTE : le rendu peint EXACTEMENT la bbox du
# host_map. Donc on ne cherche pas les bords de la carte (vf_cardscan est mort de ca —
# il n'existe PAS de signal universel « carte » : frontiere franche sur gnfHl, AUCUNE
# sur KKniW ou la webcam est incrustee a ras dans une capture). On teste des bandes de
# la SOURCE de part et d'autre de chaque bord de la bbox :
#   - bande juste DEDANS  : doit etre webcam  -> sinon le vert couvre du contenu = SURCOUVERTURE
#   - bande juste DEHORS  : doit etre page    -> sinon de la carte reste a nu   = SOUSCOUVERTURE
# L'oracle reconnait « webcam vs page » sans avoir besoin d'un gradient : le cas KKniW
# passe. La doctrine « le vert touche un bord ssi la carte le touche » est testee dans
# les deux sens — la SUR-couverture, invisible a tous les juges (doctrine monotone),
# devient enfin detectable sans relancer la guerre des juges (aucun fixer branche).
#
# GARDE-FOUS APPRIS SUR LE CORPUS :
#   - bandes sur le TIERS CENTRAL de chaque cote seulement : les coins arrondis d'une
#     carte ronde (eglV pip rond) montrent le fond DANS la bbox sans etre un defaut.
#   - cote dont la bbox touche le bord ecran (<=1% du cadre) : pas de bande exterieure
#     (rien a tester dehors). La bande INTERIEURE reste testee. itWI (carte a x=0, vert
#     a x=80) est bien attrape : sa bbox ne touche PAS le bord, la bande exterieure a
#     x<80 montre la webcam -> SOUSCOUVERTURE.
#   - bande exterieure decalee de ~1.5% du cadre au-dela du bord : le halo voulu
#     (draw(...,mg) cape a MG_PX, verdict Boss 27/07) n'est pas un defaut.
#   - scenes off >= 2 s : question fermee « talking head plein cadre ? » sur la frame
#     source reduite + porte de MOUVEMENT (une photo/banniere statique du narrateur est
#     du CONTENU, doctrine). Attrape les heroes rates (N1rAC 9,6 s, itWI 20,2 s de
#     visage a nu sur scene off) qu'aucun pavage ne peut voir.
#   - scenes hero : rien a faire ici, le plein-cadre vert se verifie au pixel (greenscan).
#   - reponse VLM invalide/timeout -> sonde IGNOREE (comptee unknown), jamais un flag.
#
# COUT : ~8-24 appels VLM par scene pip (4 cotes x 2 bandes x 1-3 sondes), ~2-4 s
# l'appel sur gemma3:12b (8 Go VRAM, coexiste avec NVENC dans les 16 Go). Une video
# type = 5-15 min — tient dans le tick de 55 min comme etape de JUGEMENT apres un END.
import sys, os, json, base64, urllib.request
import cv2, numpy as np

wd = sys.argv[1]
PROBES_LONG = int(sys.argv[sys.argv.index("--probes-long") + 1]) if "--probes-long" in sys.argv else 3
LIMIT = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
OUTJSON = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None

OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = os.environ.get("VLM_MODEL", "gemma3:12b")

# Bande NUE abandonnee apres smoke-test (itWI, 8 faux flags sur 8) : une bande de 32 px
# de mur sombre est indecidable SANS contexte — l'oracle repondait « page » sur de la
# chambre de streamer. La bande contextuelle regle ca : frame ENTIERE + rectangle rouge
# marquant la zone, question fermee « webcam ou page ». On ne demande jamais au VLM de
# PRODUIRE une box (constat vlm_probe : pas fiable) — on lui en DONNE une a classifier.
PROMPT_REGION = (
    "A red rectangle is drawn on this video frame. The video has a webcam overlay "
    "showing the narrator (a real person filmed by their camera, with their room as "
    "background). Question: what is INSIDE the red rectangle — part of the webcam "
    "overlay (the narrator or their room), or page/screen content (game, browser, "
    "code, slides, desktop, text)? Only answer webcam or page if you can actually "
    "tell from what is visible inside the rectangle. Answer with ONE word: webcam, "
    "page, or unclear.")

PROMPT_TALKING = (
    "This is a frame from a YouTube video. Question: is this a full-screen shot of a "
    "person talking to the camera (a talking head filling most of the frame, filmed "
    "live by a camera)? Screenshots, slides, web pages, photos with a person IN them, "
    "small webcam overlays in a corner do NOT count. Answer with ONE word: yes or no.")

def _call(img, prompt):
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok: return None
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False,
                       "images": [base64.b64encode(jpg.tobytes()).decode()],
                       "options": {"temperature": 0}}).encode()
    try:
        r = urllib.request.urlopen(urllib.request.Request(
            OLLAMA, body, {"Content-Type": "application/json"}), timeout=90)
        return json.loads(r.read())["response"].strip().lower()
    except Exception:
        return None

def ask(img, prompt):
    a = _call(img, prompt)
    if a is None: return None
    if a.startswith("yes"): return True
    if a.startswith("no"): return False
    return None

def ask_region(fr, rect, ctx):
    """True = webcam, False = page, None = indecidable.
    Le VLM recoit une FENETRE DE CONTEXTE (la carte + ses alentours) avec la zone
    marquee en rouge — pas la frame entiere : reduite a 768 px, une bande de 32 px
    devenait ~13 px, illisible, et l'oracle devinait par proximite avec la carte
    (smoke itWI : bande de ciel de jeu declaree webcam). La fenetre garde la carte
    visible (le contexte qui permet de decider) et la bande y reste lisible."""
    x0, y0, x1, y1 = rect
    bx0, by0, bx1, by1 = ctx
    mx = int(0.5 * (bx1 - bx0)); my = int(0.5 * (by1 - by0))
    wx0 = max(0, min(x0, bx0) - mx); wy0 = max(0, min(y0, by0) - my)
    wx1 = min(W, max(x1, bx1) + mx); wy1 = min(H, max(y1, by1) + my)
    im = fr[wy0:wy1, wx0:wx1].copy()
    if im.size == 0: return None
    cv2.rectangle(im, (x0 - wx0, y0 - wy0), (x1 - wx0, y1 - wy0), (0, 0, 255),
                  max(3, (wx1 - wx0) // 250))
    sw = min(768, im.shape[1])
    im = cv2.resize(im, (sw, max(8, int(sw * im.shape[0] / im.shape[1]))))
    a = _call(im, PROMPT_REGION)
    if a is None: return None
    if a.startswith("webcam"): return True
    if a.startswith("page"): return False
    return None

hm = json.load(open(os.path.join(wd, "host_map.json")))
cap = cv2.VideoCapture(os.path.join(wd, "source.mp4"))
W = int(cap.get(3)); H = int(cap.get(4))

def frame(t):
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, f = cap.read()
    return f if ok else None

THICK = max(24, int(0.03 * min(W, H)))     # meme epaisseur de bande que vlm_box
HALO = int(0.015 * max(W, H))              # marge du halo voulu avant la bande ext.
EDGE = 0.01                                # bbox a <=1% du cadre = touche le bord

def strips(box):
    """[(cote, bande_interieure_rect, bande_exterieure_rect_ou_None), ...] en px.
    Bandes sur le tiers central du cote (coins arrondis = pas un defaut)."""
    x0 = int(box[0] * W); y0 = int(box[1] * H)
    x1 = x0 + int(box[2] * W); y1 = y0 + int(box[3] * H)
    cx0 = x0 + (x1 - x0) // 3; cx1 = x1 - (x1 - x0) // 3
    cy0 = y0 + (y1 - y0) // 3; cy1 = y1 - (y1 - y0) // 3
    out = []
    for side, at_edge, inner, outer in (
        ("G", x0 <= EDGE * W,       (x0 + 4, cy0, x0 + 4 + THICK, cy1),
                                    (max(0, x0 - HALO - THICK), cy0, max(1, x0 - HALO), cy1)),
        ("D", x1 >= (1 - EDGE) * W, (x1 - 4 - THICK, cy0, x1 - 4, cy1),
                                    (min(W - 1, x1 + HALO), cy0, min(W, x1 + HALO + THICK), cy1)),
        ("H", y0 <= EDGE * H,       (cx0, y0 + 4, cx1, y0 + 4 + THICK),
                                    (cx0, max(0, y0 - HALO - THICK), cx1, max(1, y0 - HALO))),
        ("B", y1 >= (1 - EDGE) * H, (cx0, y1 - 4 - THICK, cx1, y1 - 4),
                                    (cx0, min(H - 1, y1 + HALO), cx1, min(H, y1 + HALO + THICK))),
    ):
        out.append((side, inner, None if at_edge else outer))
    return out

def crop(fr, r):
    x0, y0, x1, y1 = r
    if x1 - x0 < 8 or y1 - y0 < 8: return None
    return fr[y0:y1, x0:x1]

def moving(t, r):
    a = frame(t); b = frame(t + 0.5)
    if a is None or b is None: return True
    pa, pb = crop(a, r), crop(b, r)
    if pa is None or pb is None or pa.shape != pb.shape: return True
    return float((cv2.absdiff(cv2.cvtColor(pa, cv2.COLOR_BGR2GRAY),
                              cv2.cvtColor(pb, cv2.COLOR_BGR2GRAY)) > 18).mean()) >= 0.10

flags = []; nq = 0
scenes = [s for s in hm if s.get("host") in ("pip", "off")]
if LIMIT: scenes = scenes[:LIMIT]
for s in scenes:
    t0, t1 = s["start"], s["end"]; dur = t1 - t0
    if s["host"] == "pip" and s.get("bbox"):
        np_ = PROBES_LONG if dur > 60 else 1
        ts = [t0 + dur * f for f in ([0.5] if np_ == 1 else np.linspace(0.15, 0.85, np_))]
        for t in ts:
            fr = frame(float(t))
            if fr is None: continue
            bpx = (int(s["bbox"][0] * W), int(s["bbox"][1] * H),
                   int((s["bbox"][0] + s["bbox"][2]) * W), int((s["bbox"][1] + s["bbox"][3]) * H))
            for side, inner, outer in strips(s["bbox"]):
                if inner[2] - inner[0] >= 8 and inner[3] - inner[1] >= 8:
                    nq += 1
                    if ask_region(fr, inner, bpx) is False:
                        flags.append(dict(kind="SURCOUVERTURE", t=round(float(t), 1),
                                          scene=[round(t0, 2), round(t1, 2)], side=side))
                if outer is not None and outer[2] - outer[0] >= 8 and outer[3] - outer[1] >= 8:
                    nq += 1
                    if ask_region(fr, outer, bpx) is True:
                        flags.append(dict(kind="SOUSCOUVERTURE", t=round(float(t), 1),
                                          scene=[round(t0, 2), round(t1, 2)], side=side))
    elif s["host"] == "off" and dur >= 2.0:
        for t in ([t0 + dur * 0.5] if dur < 20 else [t0 + dur * 0.25, t0 + dur * 0.75]):
            fr = frame(float(t))
            if fr is None: continue
            small = cv2.resize(fr, (768, int(768 * H / W)))
            nq += 1
            if ask(small, PROMPT_TALKING) is True and moving(float(t), (0, 0, W, H)):
                flags.append(dict(kind="HERO_RATE", t=round(float(t), 1),
                                  scene=[round(t0, 2), round(t1, 2)], side="-"))
cap.release()

vid = os.path.basename(os.path.abspath(wd)).replace("wk_b_", "")
# agrege par (scene, kind) : N sondes concordantes = plus fort qu'une
from collections import Counter
agg = Counter((tuple(f["scene"]), f["kind"], f["side"]) for f in flags)
print("%s : %d scenes regardees, %d questions, %d flag(s)" % (vid, len(scenes), nq, len(agg)))
for (scene, kind, side), n in sorted(agg.items()):
    print("  %-15s scene %8.2f-%8.2f  cote %s  (%d sonde(s))" % (kind, scene[0], scene[1], side, n))
if OUTJSON:
    json.dump({"video": vid, "flags": flags, "questions": nq}, open(OUTJSON, "w"), indent=1)
sys.exit(1 if agg else 0)
