#!/bin/bash
until grep -q '==== DONE ====' /home/anon/videogen/agent_yt3/compositor.log 2>/dev/null; do sleep 20; done
echo "Compositor done, checking file..."
ls -lh /home/anon/videogen/out/KzObeom88Y_remix.mp4
echo "==== WATCHER DONE ===="
