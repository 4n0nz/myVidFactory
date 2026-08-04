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
OUT=$VG/out/final_${id}.mp4
BG=$VG/assets/background.mp4
INTRO=$VG/assets/pipintro.mp4
PY=$VG/.venv/bin/python
[ -f "$SRC" ] || { echo "pas de rendu $SRC"; exit 1; }
[ -f "$BG" ]  || { echo "pas de background $BG"; exit 1; }
[ -f "$INTRO" ] || { echo "pas d intro $INTRO"; exit 1; }

read cx cy <<< $($PY - "$WD" <<'PYEOF'
import json, sys, os
try:
    m = json.load(open(os.path.join(sys.argv[1], "host_map.json")))
    tot = wx = wy = 0.0
    for s in m:
        if s.get("host") == "pip" and s.get("bbox"):
            d = s["end"] - s["start"]
            wx += (s["bbox"][0] + s["bbox"][2] / 2.0) * d
            wy += (s["bbox"][1] + s["bbox"][3] / 2.0) * d
            tot += d
    if tot:
        print(round(wx / tot, 3), round(wy / tot, 3))
    else:
        print(0.5, 0.5)
except Exception:
    print(0.5, 0.5)
PYEOF
)
X=240; Y=135   # centre par defaut : (1920-1440)/2, (1080-810)/2
sx="centre"; sy="centre"
if $PY -c "exit(0 if float('$cx') > 0.58 else 1)"; then X=50;  sx="gauche (pips a droite, cx=$cx)"; fi
if $PY -c "exit(0 if float('$cx') < 0.42 else 1)"; then X=430; sx="droite (pips a gauche, cx=$cx)"; fi
if $PY -c "exit(0 if float('$cy') > 0.58 else 1)"; then Y=50;  sy="haut (pips en bas, cy=$cy)"; fi
if $PY -c "exit(0 if float('$cy') < 0.42 else 1)"; then Y=220; sy="bas (pips en haut, cy=$cy)"; fi

DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$SRC")
TOT=$($PY -c "print(round(float('$DUR')+3.0, 3))")

# encodeur : NVENC si vivant, sinon CPU (meme repli que vf_one.sh)
ENC="h264_nvenc -preset p4 -b:v 8M"
ffmpeg -y -v error -f lavfi -i color=c=red:s=320x180:r=30 -t 1 -c:v h264_nvenc /tmp/nvenc_probe.mp4 2>/dev/null || ENC="libx264 -preset fast -crf 20"

# Montage (ordre Boss 2026-08-04) : la neige TV (pipintro, trimee a 3,0 s puis jouee a
# 2x = 1,50 s pile) OCCUPE les 1,5 s d ouverture et de fermeture — elle remplace le
# fond nu des premieres versions, meme duree totale = video + 3 s. Le burst garde son
# audio (accelere d autant). concat exige des formats identiques : tout est amene a
# 1920x1080 / 30 fps / 48 kHz stereo avant.
echo "finalize $id : x=$X ($sx), y=$Y ($sy), duree ${TOT}s (dont 2x1.5s de neige)"
ffmpeg -y -v error -stream_loop -1 -i "$BG" -i "$SRC" -i "$INTRO" -filter_complex \
  "[2:v]trim=duration=3,setpts=(PTS-STARTPTS)/2,scale=1920:1080,setsar=1,fps=30,split[in0][in1];\
[2:a]atrim=duration=3,atempo=2,aresample=48000,aformat=channel_layouts=stereo,asplit[ia0][ia1];\
[0:v]fps=30,scale=1920:1080,setsar=1,trim=duration=${DUR},setpts=PTS-STARTPTS[bg];\
[1:v]scale=1440:810,setsar=1,setpts=PTS-STARTPTS[fg];\
[bg][fg]overlay=${X}:${Y}:eof_action=pass[mv];\
[1:a]aresample=48000,aformat=channel_layouts=stereo[ma];\
[in0][ia0][mv][ma][in1][ia1]concat=n=3:v=1:a=1[v][a]" \
  -map "[v]" -map "[a]" -t "$TOT" -c:v $ENC -c:a aac "$OUT"
[ -s "$OUT" ] && echo "OK $OUT" || { echo "FINALIZE_FAIL"; exit 1; }
