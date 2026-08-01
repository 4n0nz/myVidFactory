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
