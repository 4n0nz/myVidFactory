#!/usr/bin/env python3
"""Parse an SRT into captions.json = [{start, end, text}] (seconds)."""
import json, re, sys, os

ROOT = os.path.dirname(os.path.abspath(__file__))
src = sys.argv[1] if len(sys.argv) > 1 else "out/apercu_intro.srt"
out = sys.argv[2] if len(sys.argv) > 2 else "captions.json"

def to_sec(ts):
    h, m, rest = ts.split(":")
    s, ms = rest.split(",")
    return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000

txt = open(os.path.join(ROOT, src), encoding="utf-8").read()
blocks = re.split(r"\n\s*\n", txt.strip())
caps = []
for b in blocks:
    lines = [l for l in b.splitlines() if l.strip()]
    if len(lines) < 2:
        continue
    m = re.search(r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})", lines[1])
    if not m:
        continue
    text = " ".join(lines[2:]).strip()
    caps.append({"start": to_sec(m.group(1)), "end": to_sec(m.group(2)), "text": text})

json.dump(caps, open(os.path.join(ROOT, out), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"wrote {len(caps)} captions to {out}; last end = {caps[-1]['end'] if caps else 0}s")
