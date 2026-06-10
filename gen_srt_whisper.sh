#!/usr/bin/env bash
# Word-level synced subtitles via whisper.cpp.
# Transcribes the actual rendered audio -> SRT calé sur ce qu'on entend.
set -e
ROOT=/home/anon/videogen
WHISPER=/home/anon/whisper.cpp
MODEL="${WHISPER}/models/ggml-small.bin"
SRC="${1:-$ROOT/out/tutorial.mp4}"   # clean (no-subs) video by default

cd "$ROOT"
# 16kHz mono WAV = ce que whisper attend
ffmpeg -y -i "$SRC" -ar 16000 -ac 1 -c:a pcm_s16le out/audio16k.wav 2>/dev/null

# transcription FR, sortie SRT, lignes courtes lisibles
"${WHISPER}/build/bin/whisper-cli" \
  -m "$MODEL" -f out/audio16k.wav \
  -l fr -osrt -of out/subs_whisper \
  --max-len 48 -sow 2>&1 | tail -4

echo "=== SRT head ==="
head -12 out/subs_whisper.srt
