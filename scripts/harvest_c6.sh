#!/bin/bash
# Harvest du createur au PiP spherique (c6), en parallele du pipeline v3.
set -uo pipefail
wd="$HOME/videogen/wk_hv_c6"; mkdir -p "$wd"; cd "$wd"
if [ ! -f source.mp4 ]; then
  "$HOME/videogen/.venv/bin/python" -m yt_dlp \
    -f 'bestvideo[height<=720][vcodec^=avc1]/best[height<=720][ext=mp4]/best[height<=720]' \
    -o 'source.%(ext)s' 'https://www.youtube.com/watch?v=mWLDn49_8HA' 2>&1 | tail -2
  f=$(ls source.* 2>/dev/null | grep -vi '\.part' | head -1)
  [ -n "$f" ] && [ "$f" != source.mp4 ] && mv "$f" source.mp4
fi
[ -f source.mp4 ] && "$HOME/videogen/.venv/bin/python" "$HOME/yolo/scripts/harvest_crops.py" "$wd/source.mp4" c6
echo C6_HARVEST_DONE
