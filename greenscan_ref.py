#!/usr/bin/env python3
# greenscan : pour chaque scene du host_map, mesure le rectangle vert PEINT dans le
# rendu et signale s'il touche un bord de l'ecran. Compare a la carte source mesuree.
import json, sys
import cv2, numpy as np

wd = '/home/boss/videogen/wk_b_eglVxLaWRUU'
out = '/home/boss/videogen/out/b_eglVxLaWRUU.mp4'
hm = json.load(open(wd + '/host_map.json'))
cap = cv2.VideoCapture(out)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def green_box(t):
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, f = cap.read()
    if not ok: return None
    b, g, r = f[:, :, 0].astype(int), f[:, :, 1].astype(int), f[:, :, 2].astype(int)
    m = (g > 140) & (r < 110) & (b < 110)
    if m.sum() < 500: return 'no-green'
    ys, xs = np.nonzero(m)
    return [xs.min() / W, ys.min() / H, (xs.max() + 1) / W, (ys.max() + 1) / H]

print('scene            host   vert x0,y0,x1,y1                 bords touches')
for e in hm:
    if e['host'] == 'off': continue
    t = (e['start'] + e['end']) / 2.0
    gb = green_box(t)
    if gb is None: continue
    if gb == 'no-green':
        print(f"{e['start']:7.1f}-{e['end']:7.1f} {e['host']:5s} PAS DE VERT")
        continue
    x0, y0, x1, y1 = gb
    touch = []
    if x0 <= 0.004: touch.append('GAUCHE')
    if y0 <= 0.004: touch.append('HAUT')
    if x1 >= 0.996: touch.append('DROITE')
    if y1 >= 0.996: touch.append('BAS')
    area = (x1 - x0) * (y1 - y0)
    flag = ','.join(touch) if touch else '-'
    if e['host'] == 'hero' and len(touch) == 4: flag = '(hero plein cadre, normal)'
    print(f"{e['start']:7.1f}-{e['end']:7.1f} {e['host']:5s} [{x0:.4f},{y0:.4f},{x1:.4f},{y1:.4f}] aire={area*100:5.1f}%  {flag}")
