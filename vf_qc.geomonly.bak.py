#!/usr/bin/env python3
# vf_qc.py <host_map.json> — QC auto du placement pip.
# Le detecteur v2 place l'avatar sur la region live-visage ; le QC flag les segments
# ou la detection elle-meme est douteuse (une vraie webcam = COIN + PERSISTANTE).
# Flags = pip au centre (probable video-dans-contenu) OU pip bref hors-coin.
# exit 0 = clean, 1 = des flags a reviewer. Ecrit aussi <dir>/qc_report.txt.
import json, sys, os

hm_path = sys.argv[1]
hm = json.load(open(hm_path))
flags = []
for s in hm:
    if s.get('host') != 'pip' or not s.get('bbox'):
        continue
    bx, by, bw, bh = s['bbox']
    cx = bx + bw / 2; cy = by + bh / 2
    dur = s['end'] - s['start']
    corner = (bx > 0.66 or bx + bw < 0.34 or by < 0.12 or by + bh > 0.88)
    if 0.34 <= cx <= 0.66:
        flags.append((s, "pip au CENTRE (cx=%.2f) — webcam attendue en coin, probable video-dans-contenu" % cx))
    elif dur < 3.0 and not corner:
        flags.append((s, "pip bref (%.1fs) hors-coin — probable faux positif transitoire" % dur))

npip = sum(1 for s in hm if s.get('host') == 'pip')
nhero = sum(1 for s in hm if s.get('host') == 'hero')
lines = ["QC host_map: %d segs (hero=%d pip=%d), flags=%d" % (len(hm), nhero, npip, len(flags))]
for s, why in flags:
    lines.append("  FLAG %.1f-%.1fs — %s" % (s['start'], s['end'], why))
report = "\n".join(lines)
print(report)
open(os.path.join(os.path.dirname(os.path.abspath(hm_path)), 'qc_report.txt'), 'w').write(report + "\n")
sys.exit(1 if flags else 0)
