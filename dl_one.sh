#!/bin/bash
set -e
WD="$1"; URL="$2"
mkdir -p "$WD"; cd "$WD"
source /home/boss/videogen/.venv/bin/activate
rm -f source_dl.*
python3 -m yt_dlp --js-runtimes bun:/home/boss/.bun/bin/bun --remote-components ejs:github   -f 'bestvideo[height<=1080][vcodec^=avc1]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]'   --merge-output-format mp4 -o 'source_dl.%(ext)s' "$URL"
DL=$(ls source_dl.* | head -1)
H=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$DL")
if [ "$H" -gt 1080 ]; then
  ffmpeg -y -i "$DL" -vf scale=-2:1080 -c:v libx264 -preset fast -crf 20 -c:a aac -b:a 160k source.mp4
else
  mv "$DL" source.mp4
fi
echo DL_DONE
