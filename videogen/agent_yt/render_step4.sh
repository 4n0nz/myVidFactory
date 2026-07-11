#!/bin/bash
set -e
cd /home/anon/videogen
exec > /home/anon/videogen/agent_yt/render_yt.log 2>&1

echo '==== RENDU REMOTION ===='
npx remotion render src/index.ts Tutorial out/yt_botnet.mp4 --concurrency=3 --log=error

echo '==== DONE ===='
ls -lh out/yt_botnet.mp4
ffprobe -v error -show_entries format=duration -of csv=p=0 out/yt_botnet.mp4
