#!/bin/bash
# vf_finalize.sh <video_id> — composite le master vert devant le background anime.
#   Sortie : out/final_<id>.mp4 = 1,5 s de fond seul + video a 75% + 1,5 s de fond seul.
#   Le cote se lit dans host_map.json (centre pondere par la duree des scenes pip) :
#     cx > 0.58 -> pips a droite  -> video decalee a GAUCHE (le fond respire cote pip)
#     cx < 0.42 -> pips a gauche  -> video decalee a DROITE
#     sinon (pips centraux ou absents) -> video centree
#   Geometrie validee par Boss le 2026-08-03 (bg_sample2/4) : 1440x810, marge 115 px,
#   verticalement centree. Le master b_<id>.mp4 reste intact — c est lui que l avatar
#   consommera ; final_<id>.mp4 est la presentation.
id=$1
[ -n "$id" ] || { echo "usage: vf_finalize.sh <video_id>"; exit 2; }
VG=/home/boss/videogen
WD=$VG/wk_b_$id
SRC=$VG/out/b_${id}.mp4
OUT=$VG/out/final_${id}.mp4
BG=$VG/assets/background.mp4
PY=$VG/.venv/bin/python
[ -f "$SRC" ] || { echo "pas de rendu $SRC"; exit 1; }
[ -f "$BG" ]  || { echo "pas de background $BG"; exit 1; }

cx=$($PY - "$WD" <<'PYEOF'
import json, sys, os
try:
    m = json.load(open(os.path.join(sys.argv[1], "host_map.json")))
    tot = wx = 0.0
    for s in m:
        if s.get("host") == "pip" and s.get("bbox"):
            d = s["end"] - s["start"]
            wx += (s["bbox"][0] + s["bbox"][2] / 2.0) * d
            tot += d
    print(round(wx / tot, 3) if tot else 0.5)
except Exception:
    print(0.5)
PYEOF
)
X=240   # centre par defaut : (1920-1440)/2
side="centre"
if $PY -c "exit(0 if float('$cx') > 0.58 else 1)"; then X=115; side="gauche (pips a droite, cx=$cx)"; fi
if $PY -c "exit(0 if float('$cx') < 0.42 else 1)"; then X=365; side="droite (pips a gauche, cx=$cx)"; fi

DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$SRC")
TOT=$($PY -c "print(round(float('$DUR')+3.0, 3))")

# encodeur : NVENC si vivant, sinon CPU (meme repli que vf_one.sh)
ENC="h264_nvenc -preset p4 -b:v 8M"
ffmpeg -y -v error -f lavfi -i color=c=red:s=320x180:r=30 -t 1 -c:v h264_nvenc /tmp/nvenc_probe.mp4 2>/dev/null || ENC="libx264 -preset fast -crf 20"

echo "finalize $id : position $side, duree ${TOT}s"
ffmpeg -y -v error -stream_loop -1 -i "$BG" -i "$SRC" -filter_complex \
  "[0:v]fps=30,scale=1920:1080,trim=duration=${TOT},setpts=PTS-STARTPTS[bg];[1:v]scale=1440:810,setpts=PTS-STARTPTS+1.5/TB[fg];[bg][fg]overlay=${X}:135:eof_action=pass[v];[1:a]adelay=1500|1500,apad=pad_dur=1.5[a]" \
  -map "[v]" -map "[a]" -t "$TOT" -c:v $ENC -c:a aac "$OUT"
[ -s "$OUT" ] && echo "OK $OUT" || { echo "FINALIZE_FAIL"; exit 1; }
