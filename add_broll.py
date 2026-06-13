#!/usr/bin/env python3
"""Insère un B-roll AVANT chaque chapitre = pont de transition entre sections.
Moments déterministes (changements de chapitre), pas après chaque petite scène."""
import json, sys, os

ROOT = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(ROOT, sys.argv[1] if len(sys.argv) > 1 else "script_qc.json")
NCLIPS = 16
DUR = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0

data = json.load(open(src, encoding="utf-8"))
scenes = data["scenes"]
clips = [f"broll_{i:02d}.mp4" for i in range(NCLIPS)]

out, ci = [], 0
for idx, sc in enumerate(scenes):
    # broll juste avant chaque carte de chapitre (sauf tout au début)
    if sc["type"] == "chapter" and idx > 1:
        out.append({
            "id": f"broll_{ci:02d}", "type": "broll", "host": "off",
            "narration": "",
            "props": {"clip": clips[ci % len(clips)], "durationSec": DUR},
        })
        ci += 1
    out.append(sc)

data["scenes"] = out
json.dump(data, open(src, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"{ci} ponts B-roll insérés (avant chapitres) -> {len(out)} scènes au total")
