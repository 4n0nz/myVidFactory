#!/usr/bin/env python3
# vlm_probe.py — arbitre VISION (VLM local via Ollama) pour le placement avatar.
# Ne sur des tests batch20 (2026-07-17) : gemma3:12b juge correctement la SCENE la ou
# YOLO/YuNet voient des motifs — 3/3 sur les frames pieges (XzEg cam ratee -> bottom-left,
# Iup vignette decoy -> none, mCE screen-share -> none). Box precise = pas fiable ;
# REGION grossiere = fiable. ~2s/frame une fois le modele charge.
#
# API :
#   region(frame)   -> "top-left"|"top-right"|"bottom-left"|"bottom-right"|"center"|"none"|None
#                      ou est la webcam overlay du NARRATEUR (none = pas d'overlay ;
#                      ignore les visages du contenu/vignettes). None = reponse inexploitable.
#   fullface(frame) -> True|False|None : frame = talking-head plein ecran (vrai hero) ?
# frame = chemin jpg OU ndarray cv2 (downscale 960 + jpg q85 avant envoi).
# Env : VLM_MODEL (def gemma3:12b), OLLAMA_URL (def localhost:11434).
import json, base64, os, urllib.request

OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = os.environ.get("VLM_MODEL", "gemma3:12b")
REGIONS = ("top-left", "top-right", "bottom-left", "bottom-right", "center", "none")

_PROMPT_REGION = (
    "Frame from a YouTube video. There may be a small webcam overlay showing the real "
    "narrator (a human talking to camera). Ignore faces that are part of the page content, "
    "thumbnails, or embedded videos - only the narrator webcam overlay counts. "
    "In which region of the image is the webcam overlay? "
    "Answer with ONE of: top-left, top-right, bottom-left, bottom-right, center, none.")

_PROMPT_HERO = (
    "Does this frame show a person's face or upper body filling most of the frame "
    "(a full-screen talking-head shot)? Answer with ONE word: yes or no.")


def _b64(frame):
    if isinstance(frame, str):
        return base64.b64encode(open(frame, "rb").read()).decode()
    import cv2
    h, w = frame.shape[:2]
    if w > 960:
        frame = cv2.resize(frame, (960, int(h * 960 / w)))
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError("imencode")
    return base64.b64encode(buf.tobytes()).decode()


def _ask(prompt, frame, timeout=120):
    req = urllib.request.Request(OLLAMA, json.dumps({
        "model": MODEL, "prompt": prompt, "images": [_b64(frame)],
        "stream": False, "options": {"temperature": 0}}).encode(),
        {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    return r["response"].strip().lower()


def region(frame):
    out = _ask(_PROMPT_REGION, frame)
    for reg in REGIONS:
        if reg in out:
            return reg
    return None


def fullface(frame):
    out = _ask(_PROMPT_HERO, frame)
    if out.startswith("yes"):
        return True
    if out.startswith("no"):
        return False
    return None


if __name__ == "__main__":
    import sys, cv2
    path = sys.argv[1]
    if path.endswith((".jpg", ".png", ".jpeg")):
        fr = path
    else:
        t = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, fr = cap.read()
        cap.release()
        if not ok:
            sys.exit("frame illisible")
    print("region  :", region(fr))
    print("fullface:", fullface(fr))
