#!/usr/bin/env python3
"""Clean-text captions for the terminal PiP.
Uses the EXACT script narration (no whisper typos), split into short lines and
timed per-scene from the manifest frame counts. Output: captions.json."""
import json, os, re, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
script_path = sys.argv[1] if len(sys.argv) > 1 else "script_qc.json"
script = json.load(open(os.path.join(ROOT, script_path), encoding="utf-8"))
manifest = json.load(open(os.path.join(ROOT, "render-manifest.json"), encoding="utf-8"))
fps = manifest["meta"]["fps"]
TAIL = 0.7          # trailing silence per scene (matches build_audio)
MAXLEN = 52         # max chars per terminal line

narr = {s["id"]: s["narration"] for s in script["scenes"]}

def segments(text):
    # split into sentences, then break long ones on commas/clauses
    sents = re.split(r"(?<=[.!?:])\s+", text.strip())
    out = []
    for s in sents:
        s = s.strip()
        if not s:
            continue
        if len(s) <= MAXLEN:
            out.append(s)
            continue
        # break on commas / connectors, packing words up to MAXLEN
        words = re.split(r"(\s+)", s)
        cur = ""
        for w in words:
            if len(cur) + len(w) > MAXLEN and cur.strip():
                out.append(cur.strip())
                cur = w.lstrip()
            else:
                cur += w
        if cur.strip():
            out.append(cur.strip())
    return out

caps = []
cursor = 0.0
for sc in manifest["scenes"]:
    dur = sc["durationInFrames"] / fps
    speech = max(dur - TAIL, 0.5)
    segs = segments(narr.get(sc["id"], ""))
    total = sum(len(s) for s in segs) or 1
    t = cursor
    for s in segs:
        seg_dur = speech * (len(s) / total)
        caps.append({"start": round(t, 3), "end": round(t + seg_dur, 3), "text": s})
        t += seg_dur
    cursor += dur

json.dump(caps, open(os.path.join(ROOT, "captions.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print(f"wrote {len(caps)} clean captions; last end = {caps[-1]['end'] if caps else 0}s")
