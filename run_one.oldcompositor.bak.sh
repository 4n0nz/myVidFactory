#!/bin/bash
# Usage: run_one.sh <workdir> <url> <outname>
set -e
WD="$1"; URL="$2"; OUT="$3"
VG=/home/boss/videogen
mkdir -p "$WD"; WD="$(cd "$WD" && pwd)"; cd "$WD"
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
echo "=== [$OUT] TRANSCODE 854x480 (detection) ==="
ffmpeg -y -i source.mp4 -vf scale=854:480 -c:v libx264 -preset ultrafast -an source_h264.mp4
echo "=== [$OUT] HOST DETECTION ==="
python3 $VG/agent_yt/analyze_host.py "$WD/source_h264.mp4"
echo "=== [$OUT] BUILD COMPOSITOR ==="
python3 $VG/agent_yt3/build_compositor.py "$WD" "$OUT"
echo "=== [$OUT] RENDER ==="
bash "$WD/run_compositor.sh"
echo "=== [$OUT] DONE ==="
ls -lh $VG/out/"$OUT"
