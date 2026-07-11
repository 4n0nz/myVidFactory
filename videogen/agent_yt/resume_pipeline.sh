#!/bin/bash
set -e
cd /home/anon/videogen/agent_yt
exec >> pipeline.log 2>&1

source /home/anon/videogen/.venv-xtts/bin/activate

echo ''
echo '=== 2/3 EXTRACT AUDIO ==='
ffmpeg -i source.mp4 -ar 16000 -ac 1 -c:a pcm_s16le source_audio.wav -y

echo ''
echo '=== 3/3 WHISPER ==='
python3 /home/anon/videogen/agent_yt/whisper_transcribe.py

echo ''
echo '==== TRANSCRIPT DONE ===='
