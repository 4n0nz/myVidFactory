#!/usr/bin/env python3
"""Build an SRT from narration, timed per-scene using manifest frame counts.
Sentence-level, proportional to char length within each scene window.
Production can swap this for whisper.cpp word-level timing."""
import json, os, re

ROOT = os.path.dirname(os.path.abspath(__file__))
manifest = json.load(open(os.path.join(ROOT, "render-manifest.json"), encoding="utf-8"))
script = json.load(open(os.path.join(ROOT, "script.json"), encoding="utf-8"))
fps = manifest["meta"]["fps"]
narr = {s["id"]: s["narration"] for s in script["scenes"]}

def fmt(t):
    h = int(t // 3600); m = int((t % 3600) // 60); s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")

def split_sentences(txt):
    parts = re.split(r"(?<=[.!?:])\s+", txt.strip())
    return [p for p in parts if p]

lines = []
idx = 1
cursor = 0.0
for sc in manifest["scenes"]:
    dur = sc["durationInFrames"] / fps
    text = narr.get(sc["id"], "")
    sents = split_sentences(text)
    # reserve the ~0.7s tail (no speech)
    speech = max(dur - 0.7, 0.5)
    total_chars = sum(len(s) for s in sents) or 1
    t = cursor
    for s in sents:
        seg = speech * (len(s) / total_chars)
        start, end = t, t + seg
        lines.append(f"{idx}\n{fmt(start)} --> {fmt(end)}\n{s}\n")
        idx += 1
        t = end
    cursor += dur

open(os.path.join(ROOT, "out", "subs.srt"), "w", encoding="utf-8").write("\n".join(lines))
print(f"wrote {idx-1} subtitle cues")
