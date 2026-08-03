#!/usr/bin/env python3
# vf_invariants.py <workdir> [--quiet] — controle TOTAL du host_map final.
#
# POURQUOI CE FICHIER EXISTE (2026-08-01, apres la fuite de frontiere).
# Les juges qc_ident / qc_geom / qc_fid sont PERCEPTUELS : ils regardent des pixels,
# donc ils echantillonnent, donc ils ont des seuils, donc ils ont des angles morts. La
# preuve payee cette nuit : qc_ident echantillonne 1 frame/s a t=X.5 et accorde +-0.6 s
# de tolerance autour de chaque scene dans covered_by_avatar(). Les deux rembourrages
# voisins se recouvrent donc sur 1.2 s : AUCUN trou de frontiere sous 1.2 s ne peut
# produire un flag, jamais. 71 trous "off" sur 19 videos (41.9 s) + 36 discontinuites
# sur 11 videos (2.2 s) sont passes CLEAN, dont 0.97 s de vrai visage plein cadre sur
# Id9GbZ4k4R4 (98.03-99.00) et 1.46 s sur XzEgfmesG8c (45.54-47.00) ENTRE DEUX HERO.
#
# Ici on ne regarde AUCUN pixel. On verifie des proprietes vraies de TOUTE sortie
# correcte, sur la totalite du host_map : pas d'echantillonnage, donc pas d'angle mort
# par construction. C'est la meme famille de controle que le `dur=OK` de vf_one.sh, qui
# est le SEUL juge de la chaine qui n'ait jamais menti (il a attrape le chevauchement de
# scenes -> DESYNC(1401.6) sur une source de 1395 s). Il ratait le trou pour une raison
# purement mecanique : un chevauchement allonge la piste, un trou ne la raccourcit pas,
# build_seg laissant passer le temps non couvert. On surveillait donc un seul cote de la
# meme classe de bug. Ce fichier ferme l'autre.
#
# REGLE DE CALIBRATION : un invariant ne se negocie pas avec un seuil perceptuel. Tout
# ce qui est ici est soit exact, soit tolerance a la frame pres (TOL). Si un controle
# demande un seuil "au jugé", il n'a rien a faire dans ce fichier — il appartient aux
# juges perceptuels. Mieux vaut cinq invariants incontestables que dix discutables : un
# invariant qui cree un faux positif detruit la confiance dans TOUS les autres.
import sys, os, json

wd = sys.argv[1]
QUIET = "--quiet" in sys.argv
HM = os.path.join(wd, "host_map.json")

def dur_source():
    """Duree de la source. cv2 d'abord (deja une dependance partout), ffprobe en repli."""
    try:
        import cv2
        c = cv2.VideoCapture(os.path.join(wd, "source.mp4"))
        fps = c.get(cv2.CAP_PROP_FPS) or 0
        n = c.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        c.release()
        if fps > 1 and n > 1: return n / fps, fps
    except Exception:
        pass
    try:
        import subprocess
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", os.path.join(wd, "source.mp4")], text=True)
        return float(out.strip()), 30.0
    except Exception:
        return None, 30.0

DUR, FPS = dur_source()
TOL = max(0.04, 1.5 / (FPS or 30))   # tolerance a la frame pres, jamais plus

hm = json.load(open(HM))
V = []
def bad(code, msg): V.append((code, msg))

# --- 0. structure minimale -------------------------------------------------
if not isinstance(hm, list) or not hm:
    bad("VIDE", "host_map vide ou pas une liste")
    print("INVARIANTS : 1 violation\n  VIDE : host_map vide ou pas une liste")
    sys.exit(1)

for i, s in enumerate(hm):
    tag = "scene[%d] %s %.2f-%.2f" % (i, s.get("host"), s.get("start", -1), s.get("end", -1))

    # --- 1. bornes coherentes ----------------------------------------------
    if s.get("start") is None or s.get("end") is None:
        bad("BORNE", tag + " : borne manquante"); continue
    if s["end"] <= s["start"]:
        bad("BORNE", tag + " : end <= start")

    # --- 2. host connu ------------------------------------------------------
    if s.get("host") not in ("hero", "pip", "off"):
        bad("HOST", tag + " : host inconnu %r" % s.get("host"))

    # --- 3. pip <=> bbox ----------------------------------------------------
    # Un pip sans box n'est pas peignable ; un hero/off avec une box est une
    # incoherence qui a deja produit du vert la ou il ne fallait pas.
    b = s.get("bbox")
    if s.get("host") == "pip":
        if not (isinstance(b, (list, tuple)) and len(b) == 4):
            bad("BBOX-MANQUANTE", tag + " : pip sans bbox exploitable (%r)" % (b,))
        else:
            x, y, w, h = [float(v) for v in b]
            # --- 4. bbox dans le cadre -------------------------------------
            if x < -1e-6 or y < -1e-6 or x + w > 1.0 + 1e-6 or y + h > 1.0 + 1e-6:
                bad("BBOX-HORS-CADRE",
                    tag + " : bbox [%.4f,%.4f,%.4f,%.4f] sort du cadre" % (x, y, w, h))
            # --- 5. bbox non degeneree -------------------------------------
            if w <= 0.01 or h <= 0.01:
                bad("BBOX-DEGENEREE",
                    tag + " : bbox [%.4f,%.4f,%.4f,%.4f] plate" % (x, y, w, h))
    elif b is not None:
        bad("BBOX-PARASITE", tag + " : %s ne devrait pas porter de bbox" % s.get("host"))

# --- 6. PAVAGE : tri, pas de trou, pas de chevauchement --------------------
# LE controle qui manquait. Un trou = des frames que personne ne peint (fuite
# d'identite) ; un chevauchement = des frames dupliquees au montage (DESYNC).
for a, b in zip(hm, hm[1:]):
    if b["start"] < a["start"]:
        bad("ORDRE", "scenes non triees : %.2f apres %.2f" % (b["start"], a["start"]))
        continue
    d = b["start"] - a["end"]
    if d > TOL:
        bad("TROU", "TROU de %.3f s entre %s %.2f-%.2f et %s %.2f-%.2f — frames non peintes"
            % (d, a["host"], a["start"], a["end"], b["host"], b["start"], b["end"]))
    elif d < -TOL:
        bad("CHEVAUCHEMENT", "CHEVAUCHEMENT de %.3f s entre %s %.2f-%.2f et %s %.2f-%.2f"
            % (-d, a["host"], a["start"], a["end"], b["host"], b["start"], b["end"]))

# --- 7. bords : la couverture part de 0 et va jusqu'a la fin ---------------
if hm[0]["start"] > TOL:
    bad("BORD-DEBUT", "couverture demarre a %.3f s, pas a 0" % hm[0]["start"])
if DUR and (DUR - hm[-1]["end"]) > TOL:
    bad("BORD-FIN", "couverture s'arrete a %.2f s pour une source de %.2f s (%.2f s a nu)"
        % (hm[-1]["end"], DUR, DUR - hm[-1]["end"]))
if DUR and (hm[-1]["end"] - DUR) > TOL:
    bad("BORD-FIN", "couverture depasse la source : %.2f s pour %.2f s"
        % (hm[-1]["end"], DUR))

# --- 8. SUSPECTS : structurel, mais avec un seuil -> pas fatal --------------
# Un invariant ne se negocie pas ; ce qui suit repose sur un seuil, donc ca se SIGNALE
# et ca ne tue pas la passe. Separer les deux paliers est deliberé : un invariant qui
# produit un faux positif detruit la confiance dans tous les autres.
#
# Le seul signal structurel fiable trouve a ce jour : une scene "off" COURTE coincee
# entre deux scenes ou le narrateur est present. Le temps est bien couvert — donc les
# invariants de pavage sont mueTs — mais la scene ne peint RIEN. C'est la signature de
# l'artefact de frontiere (fin prouvee a la sonde fine, debut suivant reste sur la
# grille STEP=1.0) que pin_render ferme depuis 12746b4. Le narrateur ne se teleporte
# pas : s'il est la avant et apres, il est la pendant.
# Mesure 2026-08-01 : 71 de ces off sur 19 videos, 41.9 s cumulees, tous passes CLEAN.
#
# CE QUI N'EST PAS ICI, ET POURQUOI. Un seuil sur la TAILLE de box (attraper la
# sur-couverture type o9x8kycf3Wo, vert 2x plus haut que la carte) a ete mesure puis
# REJETE : sur les 389 box pip du corpus, p95 = 0.846 de hauteur et p99 = 1.000, parce
# que les colonnes pleine hauteur sont un layout legitime et courant (QU-fGu6imlE 43
# pips en [0.667,0,0.332,1.0], gnfHlIoh34Q en [0.033,0,0.302,1.0]). La mauvaise box
# d'o9x8 est [0.165,0.044,0.241,0.924] : geometriquement INDISCERNABLE d'une colonne
# correcte. Un seuil de hauteur aurait flagge 22 scenes dont la majorite sont bonnes.
# La sur-couverture ne se voit qu'en mesurant les bords de la carte DANS LA SOURCE ;
# elle n'appartient ni aux invariants ni aux suspects, mais a un juge perceptuel — et
# ancre sur la source, jamais sur le prior (leçon vKMxg : une fenetre de recherche
# centree sur l'hypothese testee ne peut que la confirmer).
SUSP_MAX = 1.0
S = []
for i, s in enumerate(hm):
    if s.get("host") != "off" or i == 0 or i + 1 >= len(hm): continue
    if (s["end"] - s["start"]) >= SUSP_MAX: continue
    a, b = hm[i-1], hm[i+1]
    if a.get("host") == "off" or b.get("host") == "off": continue
    S.append("off NON PEINT %.2f-%.2f (%.2fs) entre %s et %s — narrateur present des deux cotes"
             % (s["start"], s["end"], s["end"] - s["start"], a["host"], b["host"]))

# --- 9. ARBITRAGE : une scene pip qui CONTIENT un cluster consensus fort ---------
# Le defaut dominant du corpus (mesure 2026-08-02, 6 videos) : le prior mesure le
# CONTENEUR (panneau UI, colonne de layout, fenetre) au lieu de la carte, et la mesure
# brute n'est jamais arbitree ALORS QUE le bon cluster existe (N1rACQTJepA : cluster
# n=238 servant 12 scenes, 4 scenes-panneau a 4.3x son aire jamais rappelees a l ordre).
# Regle relationnelle (JSON pur, host_map + box_consensus) : une scene pip dont la bbox
# contient (epsilon EPS_ARB — le bon cluster de N1rAC DEBORDE de 9 px de la box
# fautive) la box d un cluster pip a forte fraction des votes, avec un ratio d aires
# eleve, n est probablement pas une carte : c est ce qu il y a AUTOUR.
# SEUILS PAR INVENTAIRE (269 paires mesurees sur les archives des runs 1-2, jamais a
# priori) : l extension d anneau legitime monte a 1.6-1.8x (bgMs 1.61 fracvotes 1.00,
# video CLEAN), donc pas de vallee sur le ratio seul ; le COUPLE ratio x fracvotes
# separe. FORT = ratio>=3.0 ET fracvotes>=0.10 : attrape TOUS les defauts verifies
# (N1rAC 3.97-4.45, gnfHl 7.82, OfrZE 4.18/13.42, T-chq 4.12/8.86) et AUCUNE video
# saine (les 5.72/4.52/3.32 du corpus sont tous a fracvotes<=0.03). INFO = ratio>=2.0
# ET fracvotes>=0.30 : zone grise documentee (T-chq 2.01, w_Px4 2.81), signalee sans
# certitude. Deux paliers SUSPECT, jamais fatal : rapport seul, aucun fixer.
EPS_ARB = 0.012
ARB_FORT = (3.0, 0.10)
ARB_INFO = (2.0, 0.30)
bc_p = os.path.join(wd, "box_consensus.json")
if os.path.exists(bc_p):
    try:
        bc = json.load(open(bc_p))
    except Exception:
        bc = []
    pcl = [c for c in bc if c.get("kind") == "pip" and c.get("box")]
    tot_votes = sum(c.get("n", 0) for c in pcl) or 1
    for s in hm:
        if s.get("host") != "pip" or not s.get("bbox"): continue
        bx0, by0, bw, bh = [float(v) for v in s["bbox"]]
        bx1, by1 = bx0 + bw, by0 + bh
        sa = bw * bh
        best = None
        for c in pcl:
            cx0, cy0, cw, ch = [float(v) for v in c["box"]]
            cx1, cy1 = cx0 + cw, cy0 + ch
            ca = cw * ch
            if ca <= 0 or sa <= ca: continue
            if not (cx0 >= bx0 - EPS_ARB and cy0 >= by0 - EPS_ARB
                    and cx1 <= bx1 + EPS_ARB and cy1 <= by1 + EPS_ARB): continue
            same = (abs(bx0-cx0) < EPS_ARB and abs(by0-cy0) < EPS_ARB
                    and abs(bx1-cx1) < EPS_ARB and abs(by1-cy1) < EPS_ARB)
            if same: continue
            ratio = sa / ca; frac = c.get("n", 0) / tot_votes
            if best is None or ratio * frac > best[0] * best[1]:
                best = (ratio, frac, c)
        if best is None: continue
        ratio, frac, c = best
        tier = None
        if ratio >= ARB_FORT[0] and frac >= ARB_FORT[1]: tier = "ARBITRAGE-FORT"
        elif ratio >= ARB_INFO[0] and frac >= ARB_INFO[1]: tier = "ARBITRAGE-INFO"
        if tier:
            S.append("%s scene pip %.2f-%.2f contient le cluster n=%d (%.0f%% des votes) "
                     "a %.1fx son aire — box probablement = conteneur, pas carte"
                     % (tier, s["start"], s["end"], c.get("n", 0), 100 * frac, ratio))

# --- rapport ---------------------------------------------------------------
from collections import Counter
if S:
    print("SUSPECTS : %d" % len(S))
    for m in S: print("  SUSPECT          %s" % m)

if not V:
    if not QUIET:
        print("INVARIANTS : OK (%d scenes, %.2f s couvertes sans trou ni chevauchement)"
              % (len(hm), hm[-1]["end"] - hm[0]["start"]))
    sys.exit(0)

print("INVARIANTS : %d violation(s)" % len(V))
for code, n in Counter(c for c, _ in V).most_common():
    print("  %-16s x%d" % (code, n))
for code, msg in V:
    print("  %-16s %s" % (code, msg))
sys.exit(1)
