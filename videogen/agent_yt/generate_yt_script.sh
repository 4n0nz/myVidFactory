#!/bin/bash
set -e
VIDGEN=/home/anon/videogen
WORKDIR=$VIDGEN/agent_yt
cd $WORKDIR

echo '=== NARRATOR AGENT (YT) ==='

~/.npm-global/bin/claude -p "$(cat $VIDGEN/SPEC_yt_narrator.md)

=== source_audio.json (transcription whisper) ===
$(cat $WORKDIR/source_audio.json)

=== host_map.json (detection narrateur) ===
$(cat $WORKDIR/host_map.json)

Ecris script_qc.json dans le repertoire courant." \n  --dangerously-skip-permissions > narrator_yt.log 2>&1

if [ ! -f script_qc.json ]; then
  echo 'ERREUR: script_qc.json non genere'
  tail -20 narrator_yt.log
  exit 1
fi

echo 'script_qc.json genere:'
python3 -c "import json; d=json.load(open('script_qc.json')); scenes=d['scenes'] if isinstance(d,dict) else d; print(f'{len(scenes)} scenes')"
echo '==== SCRIPT DONE ===='
