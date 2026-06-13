#!/usr/bin/env bash
# Télécharge du B-roll libre de droits (Pexels + Mixkit), filtre <= 60s.
set -u
UA='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36'
OUT=/home/anon/videogen/broll
mkdir -p "$OUT"
cd "$OUT"

dl () { # url outfile
  curl -s -L -A "$UA" -o "$2" "$1" || return 1
}

keep_if_short () { # file -> keep if <=60s and is a real video
  local f="$1"
  local d
  d=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f" 2>/dev/null)
  if [ -z "$d" ]; then echo "  ✗ pas une vidéo, supprimé: $f"; rm -f "$f"; return; fi
  awk "BEGIN{exit !($d<=60)}" || { echo "  ✗ ${d}s >60s, supprimé: $f"; rm -f "$f"; return; }
  printf "  ✓ %-28s %5.1fs\n" "$(basename "$f")" "$d"
}

echo "===== PEXELS ====="
for id in 5377525 5377684 5377682 5377521 5377517 5377683 5377519 5377699 \
          5377690 5377685 5377688 32831021 32336447 13375774 5377994 5377987 \
          5377701 6331006; do
  f="pexels_${id}.mp4"
  [ -s "$f" ] && { echo "  (déjà) $f"; continue; }
  dl "https://www.pexels.com/download/video/${id}/" "$f" && keep_if_short "$f"
done

echo "===== MIXKIT ====="
# scrape plusieurs recherches, garde la meilleure résolution par id
ids=$(for q in hacker anonymous cyber-security hacking matrix code-data; do
        curl -s -A "$UA" "https://mixkit.co/free-stock-video/$q/"
      done | grep -oE 'assets\.mixkit\.co/videos/[0-9]+/' | grep -oE '[0-9]+' | sort -u)
for id in $ids; do
  f="mixkit_${id}.mp4"
  [ -s "$f" ] && { echo "  (déjà) $f"; continue; }
  for res in 1080 720 360; do
    url="https://assets.mixkit.co/videos/${id}/${id}-${res}.mp4"
    if curl -s -L -A "$UA" -o "$f" "$url" && [ -s "$f" ] && head -c4 "$f" | grep -q . ; then
      # vérifie que c'est bien un mp4 (pas une page 404)
      if ffprobe -v error "$f" >/dev/null 2>&1; then break; else rm -f "$f"; fi
    fi
  done
  [ -s "$f" ] && keep_if_short "$f"
done

echo "===== RÉSUMÉ ====="
echo "clips gardés: $(ls -1 *.mp4 2>/dev/null | wc -l)"
du -sh "$OUT"
