#!/usr/bin/env python3
# Compositor PER-ERA: un masque arrondi + bordure verte par segment pip, a sa PROPRE bbox.
# (vs l'ancien cluster 4x4 qui moyennait les positions). Hero = avatar plein ecran.
import json, subprocess, os, sys, statistics
from PIL import Image, ImageDraw

workdir  = sys.argv[1] if len(sys.argv) > 1 else '/home/boss/videogen/agent_yt3'
out_name = sys.argv[2] if len(sys.argv) > 2 else 'KzObeom88Y_remix.mp4'
hmap   = json.load(open(workdir+'/host_map.json'))
avatar = '/home/boss/videogen/public/avatar.mp4'
source = workdir+'/source.mp4'
outp   = '/home/boss/videogen/out/'+out_name
mdir   = workdir+'/masks'; os.makedirs(mdir, exist_ok=True)
for f in os.listdir(mdir):
    if f.endswith('.png'): os.remove(mdir+'/'+f)

r = subprocess.check_output('ffprobe -v error -select_streams v:0 -show_entries stream=width,height,duration -of csv=p=0 '+source, shell=True).decode().strip().split(',')
W, H, dur = int(r[0]), int(r[1]), float(r[2])+2

hero_segs = [s for s in hmap if s['host'] == 'hero']
pip_segs  = [s for s in hmap if s['host'] == 'pip' and s.get('bbox')]

def ee(segs):
    return '+'.join("between(t,%s,%s)" % (s['start'], s['end']) for s in segs) if segs else '0'
def cover(w, h):
    return "scale=%d:%d:force_original_aspect_ratio=increase,crop=%d:%d" % (w, h, w, h)

hero_en = ee(hero_segs)

# un cluster = un seg pip (position propre). pad +6% pour couvrir le cadre orange source.
G = 10
clusters = []
for s in pip_segs:
    fx, fy, fw, fh = s['bbox']
    x = int(fx*W); y = int(fy*H); w = int(fw*W); h = int(fh*H)
    pad = int(min(w, h)*0.10)
    # bias directionnel: le glow source est souvent sous-detecte du cote OPPOSE au bord
    # que le pip touche (bord sombre). On etend plus vers le centre de l'ecran.
    cx, cy = x+w/2, y+h/2
    extra = int(min(w, h)*0.14)
    top    = pad + (extra if cy > 0.55*H else 0)   # pip en bas -> etend vers le haut
    bottom = pad + (extra if cy < 0.45*H else 0)   # pip en haut -> etend vers le bas
    left   = pad + (extra if cx > 0.55*W else 0)   # pip a droite -> etend vers la gauche
    right  = pad + (extra if cx < 0.45*W else 0)
    nx = max(0, x-left); ny = max(0, y-top)
    nw = min(w+left+right, W-nx); nh = min(h+top+bottom, H-ny)
    nw -= nw % 2; nh -= nh % 2
    if nw < 16 or nh < 16: continue
    clusters.append({'x': nx, 'y': ny, 'w': nw, 'h': nh, 'seg': s})

# masque arrondi + bordure verte glow par cluster
for i, c in enumerate(clusters):
    w, h = c['w'], c['h']; rad = int(min(w, h)*0.14)
    m = Image.new('L', (w, h), 0); ImageDraw.Draw(m).rounded_rectangle([0, 0, w-1, h-1], radius=rad, fill=255)
    m.save('%s/mask%d.png' % (mdir, i))
    bw, bh = w+2*G, h+2*G; b = Image.new('RGBA', (bw, bh), (0, 0, 0, 0)); db = ImageDraw.Draw(b)
    db.rounded_rectangle([0, 0, bw-1, bh-1], radius=rad+G, outline=(0, 255, 0, 40), width=10)
    db.rounded_rectangle([G-3, G-3, bw-G+2, bh-G+2], radius=rad+3, outline=(0, 255, 0, 120), width=5)
    db.rounded_rectangle([G, G, bw-G-1, bh-G-1], radius=rad, outline=(0, 255, 0, 235), width=3)
    b.save('%s/border%d.png' % (mdir, i))

nc = len(clusters)
inputs = "-i %s -stream_loop -1 -t %d -i %s" % (source, int(dur), avatar)
idx = 2; mask_idx = []; bord_idx = []
for i in range(nc):
    inputs += " -i %s/mask%d.png -i %s/border%d.png" % (mdir, i, mdir, i)
    mask_idx.append(idx); bord_idx.append(idx+1); idx += 2

fc = "[1:v] fps=60,split=%d %s;" % (1+nc, ''.join("[a%d]" % i for i in range(1+nc)))
fc += "[a0] %s [av_hero];" % cover(W, H)
for i, c in enumerate(clusters):
    fc += "[a%d] %s,format=rgba [sc%d];" % (i+1, cover(c['w'], c['h']), i)
    fc += "[sc%d][%d:v] alphamerge [pr%d];" % (i, mask_idx[i], i)

fc += "[0:v][av_hero] overlay=0:0:enable='%s' [v0];" % hero_en
last = 'v0'
for i, c in enumerate(clusters):
    en = ee([c['seg']]); x, y = c['x'], c['y']
    t1 = "vr%d" % i; t2 = "vb%d" % i
    fc += "[%s][pr%d] overlay=%d:%d:enable='%s' [%s];" % (last, i, x, y, en, t1)
    fc += "[%s][%d:v] overlay=%d:%d:enable='%s' [%s];" % (t1, bord_idx[i], x-G, y-G, en, t2)
    last = t2
fc += "[%s] null [vout]" % last

cmd = ("ffmpeg -y "+inputs+" -filter_complex \""+fc+"\" -map \"[vout]\" -map 0:a "
       "-c:v h264_nvenc -preset p4 -rc vbr -cq 23 -b:v 0 -c:a copy "+outp)
open(workdir+'/run_compositor.sh', 'w').write(
    "#!/bin/bash\nset -e\nexec > "+workdir+"/compositor.log 2>&1\necho '=== START ==='\n"+cmd+
    "\necho '==== DONE ===='\nls -lh "+outp+"\n")
os.chmod(workdir+'/run_compositor.sh', 0o755)
print("PER-ERA: hero=%d pip=%d clusters=%d" % (len(hero_segs), len(pip_segs), nc))
for i, c in enumerate(clusters):
    s = c['seg']; print("  c%d %.1f-%.1fs %dx%d @(%d,%d)" % (i, s['start'], s['end'], c['w'], c['h'], c['x'], c['y']))
