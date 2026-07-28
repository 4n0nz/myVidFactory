#!/usr/bin/env python3
# narr_role.py — « le narrateur est-il le SUJET de ce plan, et ou ? »
#
# Complete cardness.py (« est-ce une carte ? ») pour donner les 4 cas de la doctrine
# Boss, sans jamais dependre d un seuil de taille de visage seul :
#
#   carte + narrateur DANS la carte      -> PIP, box = la carte           (fidelite)
#   carte + narrateur ABSENT de la carte -> OFF, image intacte            (b-roll, photo)
#   pas de carte + narrateur DOMINANT    -> HERO plein cadre              (XzEg)
#   pas de carte + narrateur absent      -> OFF, image intacte
#
# Le seuil de taille (0.30 H) qui faisait tout basculer devient inutile ici : c est
# cardness qui separe carte/pas-carte, et la DOMINANCE (plus grand visage de la frame)
# qui separe hero/contenu. XzEg (visage 0.26 H, plein cadre) tombait du mauvais cote.
import os
import cv2
import numpy as np

COS_MIN = 0.363     # meme seuil identite que qc_ident / _narr_match
FACE_MIN = 0.10     # un visage plus petit que 10% de H n est pas un sujet de plan
YUNET = '/home/boss/videogen/face_detection_yunet_2023mar.onnx'
SFACE = '/home/boss/videogen/face_recognition_sface_2021dec.onnx'


class NarrRole(object):
    def __init__(self, wd, W, H):
        self.W = W; self.H = H
        self.ref = None
        p = os.path.join(wd, 'narrator_feat.npy')
        if os.path.exists(p):
            self.ref = np.load(p)
        self.yfd = cv2.FaceDetectorYN.create(YUNET, '', (W, H), score_threshold=0.6)
        self.sface = cv2.FaceRecognizerSF.create(SFACE, '')

    def _faces(self, fr):
        _, faces = self.yfd.detect(fr)
        return [] if faces is None else list(faces)

    def _is_narr(self, fr, f):
        if self.ref is None: return False
        try:
            feat = self.sface.feature(self.sface.alignCrop(fr, f)).flatten().astype(np.float32)
        except Exception:
            return False
        c = float(np.dot(feat, self.ref) / (np.linalg.norm(feat) * np.linalg.norm(self.ref) + 1e-9))
        return c >= COS_MIN

    def probe(self, cap, t0, t1, rect=None, n=3):
        """Renvoie (present, dominant, faceH) median sur n sondes.
        present  : le narrateur est visible (dans `rect` si fourni)
        dominant : son visage est le PLUS GRAND de la frame entiere
        faceH    : hauteur de son visage en fraction de l ecran"""
        pres = []; dom = []; hs = []
        for frac in np.linspace(0.25, 0.75, n):
            cap.set(cv2.CAP_PROP_POS_MSEC, (t0 + (t1 - t0) * float(frac)) * 1000.0)
            ok, fr = cap.read()
            if not ok: continue
            faces = self._faces(fr)
            if not faces:
                pres.append(0); dom.append(0); hs.append(0.0); continue
            biggest = max(float(f[3]) for f in faces)
            found = 0.0
            for f in faces:
                fh = float(f[3]) / self.H
                if fh < FACE_MIN: continue
                if rect is not None:
                    cx = float(f[0]) + float(f[2]) / 2.0
                    cy = float(f[1]) + float(f[3]) / 2.0
                    if not (rect[0] <= cx <= rect[2] and rect[1] <= cy <= rect[3]): continue
                if self._is_narr(fr, f) and fh > found: found = fh
            pres.append(1 if found > 0 else 0)
            hs.append(found)
            dom.append(1 if (found > 0 and found * self.H >= 0.85 * biggest) else 0)
        if not pres: return (False, False, 0.0)
        med = lambda v: sorted(v)[len(v) // 2]
        return (med(pres) == 1, med(dom) == 1, med(hs))
