#!/bin/bash
# vf_finalize.sh <video_id> — composite le master vert devant le background anime.
#   Sortie : out/final_<id>.mp4 = 1,5 s de fond seul + video a 75% + 1,5 s de fond seul.
#   La position se lit dans host_map.json (centre pondere par la duree des scenes pip),
#   sur LES DEUX AXES, et la video fuit le cote du pip pour que le fond respire la ou
#   il vit (ordres Boss 2026-08-03/04, marge 50 px quand on colle un bord) :
#     cx > 0.58 -> pips a droite -> video a GAUCHE (x=50) ; cx < 0.42 -> a DROITE (x=430)
#     cy > 0.58 -> pips en bas   -> video en HAUT   (y=50) ; cy < 0.42 -> en BAS   (y=220)
#     zone neutre 0.42-0.58 (ou aucun pip) -> centree sur cet axe (x=240 / y=135)
#   Video 1440x810 (75%). Le master b_<id>.mp4 reste intact — c est lui que l avatar
#   consommera ; final_<id>.mp4 est la presentation.
id=$1
[ -n "$id" ] || { echo "usage: vf_finalize.sh <video_id>"; exit 2; }
VG=/home/boss/videogen
WD=$VG/wk_b_$id
SRC=$VG/out/b_${id}.mp4
# branche avatar : la presentation se construit sur la version AVATAR quand elle existe
[ -f "$VG/out/av_${id}.mp4" ] && SRC=$VG/out/av_${id}.mp4
OUT=$VG/out/final_${id}.mp4
# VF_BACKGROUND : choisir un autre decor sans toucher a assets/background.mp4, qui reste
# le defaut. Meme logique que --avatar sur vf_avatar.py.
BG=${VF_BACKGROUND:-$VG/assets/background.mp4}
INTRO=$VG/assets/pipintro.mp4
PY=$VG/.venv/bin/python
[ -f "$SRC" ] || { echo "pas de rendu $SRC"; exit 1; }
[ -f "$BG" ]  || { echo "pas de background $BG"; exit 1; }
[ -f "$INTRO" ] || { echo "pas d intro $INTRO"; exit 1; }

read cx cy bh <<< $($PY - "$WD" <<'PYEOF'
import json, sys, os
try:
    m = json.load(open(os.path.join(sys.argv[1], "host_map.json")))
    tot = wx = wy = wh = 0.0
    for s in m:
        if s.get("host") == "pip" and s.get("bbox"):
            d = s["end"] - s["start"]
            wx += (s["bbox"][0] + s["bbox"][2] / 2.0) * d
            wy += (s["bbox"][1] + s["bbox"][3] / 2.0) * d
            wh += s["bbox"][3] * d
            tot += d
    if tot:
        print(round(wx / tot, 3), round(wy / tot, 3), round(wh / tot, 3))
    else:
        print(0.5, 0.5, 0.0)
except Exception:
    print(0.5, 0.5, 0.0)
PYEOF
)
X=240; Y=135   # centre par defaut : (1920-1440)/2, (1080-810)/2
sx="centre"; sy="centre"
if $PY -c "exit(0 if float('$cx') > 0.58 else 1)"; then X=50;  sx="gauche (pips a droite, cx=$cx)"; fi
if $PY -c "exit(0 if float('$cx') < 0.42 else 1)"; then X=430; sx="droite (pips a gauche, cx=$cx)"; fi
if $PY -c "exit(0 if float('$cy') > 0.58 else 1)"; then Y=50;  sy="haut (pips en bas, cy=$cy)"; fi
if $PY -c "exit(0 if float('$cy') < 0.42 else 1)"; then Y=220; sy="bas (pips en haut, cy=$cy)"; fi
# Colonne pleine hauteur (detail Boss 2026-08-05) : un pip qui occupe toute la hauteur
# (hauteur ponderee >= 0.85 — QU-fG 0.962, gnfHl 1.0, contre 0.45 pour un coin) fait
# partie du LAYOUT, pas d un coin a fuir : la video reste centree sur les deux axes.
if $PY -c "exit(0 if float('$bh') >= 0.85 else 1)"; then X=240; Y=135; sx="centre (colonne pleine hauteur, h=$bh)"; sy="centre"; fi

DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$SRC")
TOT=$($PY -c "print(round(float('$DUR')+3.0, 3))")

# Avatar POP-OUT (ordre Boss 2026-08-05) : l avatar est pose ICI, par-dessus la fenetre
# 75%, box 5:4 = plus petite box couvrant la zone verte projetee, puis AV_SCALE — il
# couvre le vert a 100% et DEBORDE de la fenetre video sur le background. Timing decale
# de +1.5 s (la neige d ouverture). En dessous, av_<id>.mp4 garde son pip exact.
# CENTRAGE ENSEMBLE (ordre Boss 2026-08-05) : quand le pop-out est actif, la position
# de la fenetre est recalculee pour que l ENSEMBLE fenetre+avatar soit centre sur le
# background — remplace la logique fuis-le-pip (le python renvoie X Y sur la 1re ligne).
AVATAR=$VG/public/avatar.mp4
AVCHAIN=";[vwin]null[v]"; AVIN=()
if [ -f "$AVATAR" ]; then
  AVOUT=$($PY - "$WD" "$X" "$Y" <<'PYEOF'
import json, sys, os
from collections import OrderedDict
AV_SCALE = 1.25
# ratio de la box pop-out (essais Boss 2026-08-05 : 5:4 retenu ; 848,464 = natif 16:9)
AW, AH = 5, 4
wd, X, Y = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
VW, VH, CW, CH = 1440, 810, 1920, 1080
def bail():
    print("%d %d" % (X, Y)); print(";[vwin]null[v]"); sys.exit()
try:
    m = json.load(open(os.path.join(wd, "host_map.json")))
except Exception:
    bail()
groups = OrderedDict()
for s in m:
    if s.get("host") == "pip" and s.get("bbox"):
        groups.setdefault(tuple(s["bbox"]), []).append(s)
if not groups:
    bail()
# 1re passe : geometrie des pop-out en coordonnees RELATIVES a la fenetre (sans clamp)
rel = []
for bbox, segs in groups.items():
    # colonne pleine hauteur (h >= 0.85, meme seuil que la doctrine de centrage) :
    # PAS de pop-out — l av_ en dessous couvre deja la colonne exactement, et toute
    # box x1.25 dessus mangerait la video (crash QU-fG 2026-08-06)
    if bbox[3] >= 0.85:
        continue
    rx = int(round(bbox[0] * VW)); ry = int(round(bbox[1] * VH))
    pw = max(2, int(round(bbox[2] * VW))); ph = max(2, int(round(bbox[3] * VH)))
    # ratio ADAPTATIF (verdicts Boss 2026-08-05/06) : cible = bande [5:4, 16:9], mais
    # SEULEMENT si l elargissement reste <= 40% de la zone (un grand portrait force
    # en 5:4 = +268% de largeur, il garde son ratio) ; plafond 16:9 natif
    zr = pw / float(ph)
    ar = min(max(zr, AW / float(AH)), 848 / 464.0)
    if ar / zr > 1.4:
        ar = zr
    # scale DEGRESSIF selon la part de fenetre occupee : x1.25 pour un petit pip,
    # glisse vers x1.08 quand la zone approche 30% de la fenetre (gros pip = deja
    # proeminent, le surplus mange la video)
    f = (pw * ph) / float(VW * VH)
    sc = AV_SCALE - (AV_SCALE - 1.08) * min(1.0, max(0.0, (f - 0.10) / 0.20))
    w2 = min(CW, int(round(max(pw, ph * ar) * sc))) // 2 * 2
    h2 = min(CH, int(round(w2 / ar))) // 2 * 2
    # ancrage : surplus vers l EXTERIEUR de la fenetre (cote background), jamais
    # vers l interieur de la video ; zone neutre (0.45-0.55) -> centre
    bcx = bbox[0] + bbox[2] / 2.0; bcy = bbox[1] + bbox[3] / 2.0
    if bcx > 0.55:   dx = rx
    elif bcx < 0.45: dx = rx + pw - w2
    else:            dx = rx - (w2 - pw) // 2
    if bcy > 0.55:   dy = ry
    elif bcy < 0.45: dy = ry + ph - h2
    else:            dy = ry - (h2 - ph) // 2
    rel.append((dx, dy, w2, h2, rx, ry, pw, ph, segs))
# ensemble = fenetre ∪ pop-out SIGNIFICATIFS (>= 20% du temps pip total) -> position
# de fenetre qui centre l ensemble. Les groupes rares (ex. MS7 : 20 s a gauche contre
# 411 s a droite) sont rendus quand meme mais ne comptent pas dans le centrage, sinon
# ils decentrent toute la video pour quelques secondes d ecran.
tot = sum(sum(s["end"] - s["start"] for s in r[8]) for r in rel)
big = [r for r in rel if sum(s["end"] - s["start"] for s in r[8]) >= 0.2 * tot] or rel
minx = min([0] + [r[0] for r in big]); maxx = max([VW] + [r[0] + r[2] for r in big])
miny = min([0] + [r[1] for r in big]); maxy = max([VH] + [r[1] + r[3] for r in big])
X = max(0, min(CW - VW, int(round((CW - (maxx - minx)) / 2.0 - minx))))
Y = max(0, min(CH - VH, int(round((CH - (maxy - miny)) / 2.0 - miny))))
flt = []; chain = "[vwin]"; step = 0
for dx, dy, w2, h2, rx, ry, pw, ph, segs in rel:
    x = X + rx; y = Y + ry
    # clamp au canvas dans l intervalle qui garantit vert ⊂ box avatar
    x2 = min(max(X + dx, max(0, x + pw - w2)), min(x, CW - w2))
    y2 = min(max(Y + dy, max(0, y + ph - h2)), min(y, CH - h2))
    e = "+".join("between(t,%.3f,%.3f)" % (s["start"] + 1.5, s["end"] + 1.5) for s in segs)
    flt.append("[3:v]scale=%d:%d:force_original_aspect_ratio=increase,crop=%d:%d,setsar=1[af%d]"
               % (w2, h2, w2, h2, step))
    flt.append("%s[af%d]overlay=%d:%d:enable='%s'[ao%d]" % (chain, step, x2, y2, e, step))
    bx = max(0, x2 - 1); by = max(0, y2 - 1)
    bw = min(CW - bx, w2 + 2); bh = min(CH - by, h2 + 2)
    flt.append("[ao%d]drawbox=%d:%d:%d:%d:color=0x00FF00@1:t=1:enable='%s'[av%d]"
               % (step, bx, by, bw, bh, e, step))
    chain = "[av%d]" % step; step += 1
flt.append("%snull[v]" % chain)
print("%d %d" % (X, Y))
print(";" + ";".join(flt))
PYEOF
)
  read X Y <<< $(echo "$AVOUT" | sed -n 1p)
  AVCHAIN=$(echo "$AVOUT" | sed -n 2p)
  if [ "$AVCHAIN" != ";[vwin]null[v]" ]; then
    AVIN=(-stream_loop -1 -i "$AVATAR")
    sx="centre ensemble fenetre+avatar"; sy="centre ensemble"
  fi
fi

# encodeur : NVENC si vivant, sinon CPU (meme repli que vf_one.sh)
ENC="h264_nvenc -preset p4 -b:v 8M"
ffmpeg -y -v error -f lavfi -i color=c=red:s=320x180:r=30 -t 1 -c:v h264_nvenc /tmp/nvenc_probe.mp4 2>/dev/null || ENC="libx264 -preset fast -crf 20"

# Montage (ordre Boss 2026-08-04, corrige apres verdict Boss : la neige va DANS LA
# FENETRE de la video traitee, pas plein ecran) : le background anime tourne en continu
# du debut a la fin ; le cadre 75% (1440x810, position auto 2 axes) joue neige TV 1,5 s
# (pipintro trime a 3,0 s puis 2x) -> video traitee -> neige TV 1,5 s. Duree totale =
# video + 3 s. Audio : burst accelere, puis piste video, puis burst. concat exige des
# formats identiques : chaque branche est amenee a 1440x810 / SAR 1 / 30 fps cote
# video et 48 kHz stereo cote audio.
echo "finalize $id : x=$X ($sx), y=$Y ($sy), duree ${TOT}s (neige dans la fenetre, 2x1.5s, avatar pop-out $([ ${#AVIN[@]} -gt 0 ] && echo ON || echo OFF))"
ffmpeg -y -v error -stream_loop -1 -i "$BG" -i "$SRC" -i "$INTRO" "${AVIN[@]}" -filter_complex \
  "[2:v]trim=duration=3,setpts=(PTS-STARTPTS)/2,scale=1440:810,setsar=1,fps=30,split[sn0][sn1];\
[2:a]atrim=duration=3,atempo=2,aresample=48000,aformat=channel_layouts=stereo,asplit[sa0][sa1];\
[1:v]scale=1440:810,setsar=1,fps=30,setpts=PTS-STARTPTS[fv];\
[1:a]aresample=48000,aformat=channel_layouts=stereo[fa];\
[sn0][fv][sn1]concat=n=3:v=1:a=0[fgall];\
[sa0][fa][sa1]concat=n=3:v=0:a=1[a];\
[0:v]fps=30,scale=1920:1080,setsar=1,trim=duration=${TOT},setpts=PTS-STARTPTS[bg];\
[bg][fgall]overlay=${X}:${Y}:eof_action=pass[ov];[ov]drawbox=$((X-1)):$((Y-1)):1442:812:color=0x00FF00@1:t=1[vwin]${AVCHAIN}" \
  -map "[v]" -map "[a]" -t "$TOT" -c:v $ENC -c:a aac "$OUT"
[ -s "$OUT" ] && echo "OK $OUT" || { echo "FINALIZE_FAIL"; exit 1; }
