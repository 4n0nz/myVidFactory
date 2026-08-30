#!/bin/bash
exec > /home/anon/videogen/agent_yt/compositor.log 2>&1
echo '=== COMPOSITOR dE-I5xKtmso ==='
ffmpeg -y -i /home/anon/videogen/agent_yt/source.mp4 -stream_loop -1 -t 1490 -i /home/anon/videogen/public/avatar.mp4 -filter_complex "[1:v] fps=60,split=2 [av_a][av_b];[av_a] scale=1280:720 [av_hero];[av_b] scale=320:180 [av_pip];[0:v][av_hero] overlay=0:0:enable='0' [v1];[v1][av_pip] overlay=W-w-20:H-h-20:enable='between(t,4.0,214.0)+between(t,214.0,934.0)+between(t,934.0,1488.7666666666667)' [vout]" -map "[vout]" -map 0:a -c:v libx264 -preset fast -crf 20 -c:a copy /home/anon/videogen/out/dE-I5xKtmso_remix.mp4
echo '==== DONE ===='
ls -lh /home/anon/videogen/out/dE-I5xKtmso_remix.mp4
ffprobe -v error -show_entries format=duration -of csv=p=0 /home/anon/videogen/out/dE-I5xKtmso_remix.mp4
