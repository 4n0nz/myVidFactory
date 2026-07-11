#!/bin/bash
# Download (720p, video-only) + harvest crops facecam pour 5 nouveaux createurs + les 3 existants.
set -uo pipefail
HVPY=~/videogen/.venv/bin/python
HS=~/yolo/scripts/harvest_crops.py
dl_harvest(){
  local url="$1" name="$2" wd="$HOME/videogen/wk_hv_$2"
  mkdir -p "$wd"; cd "$wd"
  if [ ! -f source.mp4 ]; then
    echo "=== download $name ==="
    $HVPY -m yt_dlp -f 'bestvideo[height<=720][vcodec^=avc1]/best[height<=720][ext=mp4]/best[height<=720]' \
      -o 'source.%(ext)s' "$url" 2>&1 | tail -2
    f=$(ls source.* 2>/dev/null | grep -vi '\.part' | head -1)
    [ -n "$f" ] && [ "$f" != "source.mp4" ] && mv "$f" source.mp4
  fi
  [ -f source.mp4 ] && $HVPY "$HS" "$wd/source.mp4" "$name" || echo "SKIP $name (pas de source)"
}
dl_harvest 'https://youtu.be/53AqruupjvQ' c1
dl_harvest 'https://youtu.be/itWI5CDVVfQ' c2
dl_harvest 'https://www.youtube.com/watch?v=ADJj3VMFWmM' c3
dl_harvest 'https://www.youtube.com/watch?v=QQEgIo4Juxg' c4
dl_harvest 'https://www.youtube.com/watch?v=gb5TlGw6Uks' c5
# existants (deja telecharges)
$HVPY "$HS" ~/videogen/wk_lastvideo/source.mp4 lastvideo
$HVPY "$HS" ~/videogen/wk_ofr/source.mp4 ofr
echo "=== crops par personne ==="
for d in ~/yolo/data/crops/*/; do echo "  $(basename $d): $(ls $d | wc -l)"; done
echo HARVEST_ALL_DONE
