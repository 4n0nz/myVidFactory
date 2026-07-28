#!/usr/bin/env python3
# greenscan.py <video_id> — mesure le vert PEINT (plus gros bloc connexe) au milieu de
# chaque scene du host_map. NE PAS LIRE SEUL : comparer a la carte SOURCE de la meme
# frame (lecon Boss 2026-07-27 19h05 — a t=8.3 la carte touchait les bords et le vert
# non : sous-couverture invisible a un test 'le vert touche-t-il un bord').
import json, sys
import cv2, numpy as np
vid = sys.argv[1]
wd = '/home/boss/videogen/wk_b_' + vid
out = '/home/boss/videogen/out/b_%s.mp4' % vid
hm = json.load(open(wd + '/host_map.json'))
cap = cv2.VideoCapture(out)
W = int(cap.get(3)); H = int(cap.get(4))
def blob(t):
    cap.set(cv2.CAP_PROP_POS_MSEC, t*1000.0)
    ok, f = cap.read()
    if not ok: return None
    b,g,r = f[:,:,0].astype(int), f[:,:,1].astype(int), f[:,:,2].astype(int)
    m = ((g>140)&(r<110)&(b<110)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n < 2: return 'no-green'
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    x,y,w,h,a = st[i]
    if a < 500: return 'no-green'
    return (x, y, x+w, y+h, a)
print('%s  %dx%d' % (vid, W, H))
print('scene              host  vert px(x0,y0)-(x1,y1)          norm                          aire')
for e in hm:
    if e['host'] == 'off': continue
    t = (e['start'] + e['end'])/2.0
    gb = blob(t)
    if gb is None: continue
    if gb == 'no-green':
        print('%8.1f-%8.1f %-5s PAS DE VERT' % (e['start'], e['end'], e['host'])); continue
    x0,y0,x1,y1,a = gb
    print('%8.1f-%8.1f %-5s px(%4d,%4d)-(%4d,%4d)  [%.4f,%.4f,%.4f,%.4f]  %4.1f%%' % (
        e['start'], e['end'], e['host'], x0,y0,x1,y1, x0/W,y0/H,x1/W,y1/H, a*100.0/(W*H)))
