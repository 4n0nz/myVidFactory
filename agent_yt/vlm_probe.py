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

# Test le plus fiable (valide vkmx 439.5, page truffee de vignettes-pieges) : CROP de la box
# + question fermee. La recherche ouverte de region se fait distraire par les visages du
# contenu ; la question orientee "y a-t-il une cam en bas-gauche ?" = complaisance ("yes"
# partout). Le crop isole la zone -> discrimination nette webcam live vs vignette/graphique.
_PROMPT_CROP = (
    "This is a CROP from a YouTube video frame. Does this crop show a LIVE webcam view of "
    "the video narrator (a real human filmed by their webcam)? A stylized thumbnail, a course "
    "cover image, a graphic with text, or page content is NOT a webcam view. "
    "Answer with ONE word: yes or no.")


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


def webcam_crop(frame, bbox, margin=0.10):
    """frame = ndarray cv2, bbox = [x,y,w,h] fractions. True si le crop (box + marge)
    est une webcam live du narrateur, False si vignette/contenu, None si inexploitable."""
    import cv2
    H, W = frame.shape[:2]
    bx, by, bw, bh = bbox
    mx, my = bw * margin, bh * margin
    x0 = max(0, int((bx - mx) * W)); y0 = max(0, int((by - my) * H))
    x1 = min(W, int((bx + bw + mx) * W)); y1 = min(H, int((by + bh + my) * H))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    out = _ask(_PROMPT_CROP, frame[y0:y1, x0:x1])
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
