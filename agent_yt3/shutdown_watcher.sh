#!/bin/bash
until grep -q '==== DONE ====' /home/anon/videogen/agent_yt3/compositor.log 2>/dev/null; do sleep 20; done
echo "$(date) - compositor done, shutting down..." >> /home/anon/videogen/agent_yt3/watcher.log
echo '1q4r7u8i9o6y3e' | sudo -S shutdown -h now
