#!/usr/bin/env python3
# vlm_box.py <workdir> [t0 t1 ...] — raffine la box pip par ORACLE VLM sur crops.
#
# Constat valide (vlm_probe 2026-07-17) : box VLM directe = pas fiable ; CROP + question
# FERMEE = fiable. Donc : bisection PAR COTE. Pour un cote donne, on teste une bande crop
# a la position candidate : "cette bande appartient-elle entierement a la vue webcam ?"
# Bande dans la carte -> oui ; bande sur la page -> non. La frontiere oui/non = le bord.
# Depart = box consensus (ou ancre visage). 3 frames, mediane par cote.
import sys, os, json, base64, urllib.request, cv2, numpy as np

VG = "/home/boss/videogen"
OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = os.environ.get("VLM_MODEL", "gemma3:12b")

PROMPT_BAND = (
    "This image is a thin strip cropped from a YouTube video frame. The video has a "
    "webcam overlay showing the narrator (a real person filmed by their webcam, with "
    "their room as background). Question: is this ENTIRE strip part of the webcam "
    "overlay (narrator or their room background)? If ANY part of the strip shows page "
    "content, desktop, code, text document or anything that is not the webcam view, "
    "answer no. Answer with ONE word: yes or no.")

def _ask(img):
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok: return None
    body = json.dumps({"model": MODEL, "prompt": PROMPT_BAND, "stream": False,
                       "images": [base64.b64encode(jpg.tobytes()).decode()],
                       "options": {"temperature": 0}}).encode()
    try:
        r = urllib.request.urlopen(urllib.request.Request(
            OLLAMA, body, {"Content-Type": "application/json"}), timeout=60)
        a = json.loads(r.read())["response"].strip().lower()
    except Exception:
        return None
    if a.startswith("yes"): return True
    if a.startswith("no"): return False
    return None

def _band(fr, side, pos, box, W, H, thick):
    x0, y0, x1, y1 = box
    if side in ("L", "R"):
        a = max(0, pos - thick//2); b = min(W, pos + thick//2)
        return fr[max(0,y0):y1, a:b]
    a = max(0, pos - thick//2); b = min(H, pos + thick//2)
    return fr[a:b, max(0,x0):x1]

def refine(frames, box_n, W, H):
    """box_n normalisee -> box raffinee par bisection oracle, par cote."""
    x0 = int(box_n[0]*W); y0 = int(box_n[1]*H)
    x1 = x0 + int(box_n[2]*W); y1 = y0 + int(box_n[3]*H)
    thick = max(24, int(0.03*min(W, H)))
    reach_out = {"L": int(0.12*W), "R": int(0.12*W), "T": int(0.10*H), "B": int(0.15*H)}
    reach_in  = {"L": int(0.10*W), "R": int(0.10*W), "T": int(0.08*H), "B": int(0.12*H)}
    res = {}
    for side in ("L", "R", "T", "B"):
        cur = {"L": x0, "R": x1, "T": y0, "B": y1}[side]
        sgn_out = -1 if side in ("L", "T") else 1
        lo = cur - sgn_out*reach_in[side]     # position surement DANS la carte
        hi = cur + sgn_out*reach_out[side]    # position surement DEHORS
        lo = max(0, min({"L": W, "R": W, "T": H, "B": H}[side]-1, lo))
        hi = max(0, min({"L": W, "R": W, "T": H, "B": H}[side]-1, hi))
        votes = []
        for fr in frames:
            a, b = lo, hi
            for _ in range(4):
                mid = (a + b)//2
                band = _band(fr, side, mid, (x0, y0, x1, y1), W, H, thick)
                if band.size < 500: break
                r = _ask(band)
                if r is None: break
                if r: a = mid          # bande dans la carte -> bord plus loin dehors
                else: b = mid          # bande sur la page -> bord plus proche dedans
            votes.append((a + b)//2)
        if votes:
            res[side] = int(np.median(votes))
        else:
            res[side] = cur
    nx0, nx1 = min(res["L"], res["R"]-8), max(res["R"], res["L"]+8)
    ny0, ny1 = min(res["T"], res["B"]-8), max(res["B"], res["T"]+8)
    return [round(nx0/W, 4), round(ny0/H, 4), round((nx1-nx0)/W, 4), round((ny1-ny0)/H, 4)]

if __name__ == "__main__":
    wd = sys.argv[1].rstrip("/")
    cons = json.load(open(os.path.join(wd, "box_consensus.json")))
    cap = cv2.VideoCapture(os.path.join(wd, "source.mp4"))
    W = int(cap.get(3)); H = int(cap.get(4)); DUR = cap.get(7)/(cap.get(5) or 30)
    ts = [float(t) for t in sys.argv[2:]] or [DUR*0.3, DUR*0.5, DUR*0.7]
    frames = []
    for t in ts:
        cap.set(cv2.CAP_PROP_POS_MSEC, t*1000.0); ok, fr = cap.read()
        if ok: frames.append(fr)
    out = []
    for c in cons:
        if c.get("kind") == "hero" or c["box"][2]*c["box"][3] > 0.5:
            out.append(c); continue
        nb = refine(frames, c["box"], W, H)
        c2 = dict(c); c2["box_vlm"] = nb
        out.append(c2)
        print("cluster", c["box"], "-> VLM", nb)
    json.dump(out, open(os.path.join(wd, "box_vlm.json"), "w"), indent=2)
