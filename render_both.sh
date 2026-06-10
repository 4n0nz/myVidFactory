#!/usr/bin/env bash
# Render the QC master in two voices (Thierry + Sylvie), with whisper subs each.
set -e
cd /home/anon/videogen
. .venv/bin/activate

render_one () {
  local VOICE="$1"; local TAG="$2"
  echo "==================== $TAG ($VOICE) ===================="
  python build_audio.py script_qc.json "$VOICE"
  rm -f "out/master_${TAG}.mp4"
  npx remotion render src/index.ts Tutorial "out/master_${TAG}.mp4" --concurrency=3 --log=error
  # whisper synced subtitles from the rendered audio
  ffmpeg -y -i "out/master_${TAG}.mp4" -ar 16000 -ac 1 -c:a pcm_s16le "out/a16_${TAG}.wav" 2>/dev/null
  /home/anon/whisper.cpp/build/bin/whisper-cli \
    -m /home/anon/whisper.cpp/models/ggml-small.bin \
    -f "out/a16_${TAG}.wav" -l fr -osrt -of "out/master_${TAG}" --max-len 48 -sow 2>/dev/null
  echo "[$TAG] done: $(ffprobe -v error -show_entries format=duration -of csv=p=0 out/master_${TAG}.mp4)s"
}

render_one fr-CA-ThierryNeural thierry
render_one fr-CA-SylvieNeural sylvie

echo "==================== ALL DONE ===================="
ls -la out/master_thierry.mp4 out/master_sylvie.mp4 out/master_thierry.srt out/master_sylvie.srt
