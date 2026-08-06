#!/usr/bin/env python3
# smartseek.py — SmartCap : VideoCapture avec seeks courts convertis en lecture avant.
#
#   Un cap.set(CAP_PROP_POS_MSEC)+read() coute ~98 ms sur la iABox (seek keyframe +
#   decode d'approche) ; une lecture sequentielle coute ~1.5 ms/frame. Or les scripts
#   de la chaine (pin_render, qc_geom, qc_fid) balaient les scenes DANS L'ORDRE :
#   presque tous leurs seeks avancent de quelques secondes. SmartCap detecte ce cas
#   et AVANCE en lisant (grab) au lieu de seeker ; un saut arriere ou lointain reste
#   un vrai seek.
#
#   Equivalence : l'ancien chemin retourne la premiere frame de pts >= t. SmartCap
#   avance en grab() en surveillant CAP_PROP_POS_MSEC (= pts de la PROCHAINE frame a
#   decoder chez OpenCV/FFmpeg) et s'arrete des que ce pts depasse t : la frame alors
#   retrieve()e est exactement celle du seek. Verifie bitwise sur b_/source (30 fps
#   et VFR yt-dlp) par smartseek_test.py avant adoption.
#
#   Usage (2 lignes par script) :
#       from smartseek import SmartCap
#       cap = SmartCap(path)        # a la place de cv2.VideoCapture(path)
import cv2

FWD_MAX_S = 5.0   # au-dela de ce bond en avant, un vrai seek redevient rentable

class SmartCap:
    def __init__(self, path):
        self._c = cv2.VideoCapture(path)
        self._pending = None          # cible POS_MSEC posee par set(), consommee par read()
        self._fps = self._c.get(cv2.CAP_PROP_FPS) or 30.0

    def set(self, prop, val):
        if prop == cv2.CAP_PROP_POS_MSEC:
            self._pending = float(val)
            return True
        self._pending = None
        return self._c.set(prop, val)

    def read(self):
        if self._pending is None:
            return self._c.read()
        t = self._pending; self._pending = None
        # REGLE MESUREE (iABox, 11 points, poussiere des deux cotes + demi-frames
        # exactes) : seek(t)+read() d'OpenCV retourne la frame d'index
        # int(t*fps/1000 + 0.5) - 1 — arrondi HALF-UP (cap_ffmpeg fait +0.5 puis
        # troncature ; round() de Python arrondit .5 au pair et rate 8.55s@30fps).
        # On avance en grab() jusqu'a ce que l'index de la PROCHAINE frame decodable
        # (POS_MSEC -> index) depasse la cible ; la derniere grab()ee est la bonne.
        tgt = int(t * self._fps / 1000.0 + 0.5) - 1
        cur = self._c.get(cv2.CAP_PROP_POS_MSEC)
        cur_idx = round((cur or 0.0) * self._fps / 1000.0)
        if 0 <= tgt - cur_idx < FWD_MAX_S * self._fps:
            grabbed = False
            while round(self._c.get(cv2.CAP_PROP_POS_MSEC) * self._fps / 1000.0) <= tgt:
                if not self._c.grab():
                    return False, None
                grabbed = True
            if grabbed:
                return self._c.retrieve()
        self._c.set(cv2.CAP_PROP_POS_MSEC, t)
        return self._c.read()

    def get(self, prop):
        return self._c.get(prop)

    def release(self):
        return self._c.release()

    def isOpened(self):
        return self._c.isOpened()
