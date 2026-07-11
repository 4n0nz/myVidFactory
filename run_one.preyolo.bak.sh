#!/bin/bash
# Usage: run_one.sh <workdir> <url> <outname.mp4> [pip_rect="x,y,w,h"]
#   - l'extension .mp4 dans <outname> est OBLIGATOIRE (sinon le muxer plante a la fin).
#   - [pip_rect] optionnel : position/taille de l'avatar dans la webcam pip (specifique a chaque
#     video, se mesure a l'oeil sur 1 frame). Defaut = valeur de la video TkT.
set -e
WD="$1"; URL="$2"; OUT="$3"
VG=/home/boss/videogen
if [ -n "$4" ]; then export PIP_RECT="$4"; fi
mkdir -p "$WD"; WD="$(cd "$WD" && pwd)"; cd "$WD"
mkdir -p "$VG/out"
source $VG/.venv/bin/activate
echo "=== [$OUT] DOWNLOAD ==="
rm -f source.mp4 source_dl.* source.webm source.mkv 2>/dev/null || true
python3 -m yt_dlp -f 'bestvideo[height<=1080][vcodec^=avc1]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]' \
  --merge-output-format mp4 -o 'source_dl.%(ext)s' "$URL"
DL=$(ls source_dl.* | head -1)
H=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$DL")
if [ "$H" -gt 1080 ]; then
  echo "=== normalize ${H}p -> 1080p ==="
  ffmpeg -y -i "$DL" -vf scale=-2:1080 -c:v libx264 -preset fast -crf 20 -c:a aac -b:a 160k source.mp4
else
  mv "$DL" source.mp4
fi
DIMS=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 source.mp4)
echo "WORKING DIMS=$DIMS"
echo "=== [$OUT] HOST DETECTION (v2 YuNet + live-motion, full-res) ==="
python3 $VG/agent_yt/analyze_host2.py "$WD/source.mp4"
echo "=== [$OUT] BUILD COMPOSITOR (segmente, rapide) ==="
python3 $VG/build_seg.py "$WD" "$OUT"
echo "=== [$OUT] RENDER (NVENC segmente) ==="
bash "$WD/run_seg.sh"
echo "=== [$OUT] DONE ==="
ls -lh $VG/out/"$OUT"
