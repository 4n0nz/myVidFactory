
import json, subprocess, os, sys, statistics
from PIL import Image, ImageDraw

workdir  = sys.argv[1] if len(sys.argv)>1 else '/home/anon/videogen/agent_yt3'
out_name = sys.argv[2] if len(sys.argv)>2 else 'remix.mp4'
hmap   = json.load(open(workdir+'/host_map.json'))
avatar = '/home/anon/videogen/public/avatar.mp4'
source = workdir+'/source.mp4'
outp   = '/home/anon/videogen/out/'+out_name
mdir   = workdir+'/masks'; os.makedirs(mdir, exist_ok=True)

r = subprocess.check_output('ffprobe -v error -select_streams v:0 -show_entries stream=width,height,duration -of csv=p=0 '+source, shell=True).decode().strip().split(',')
W,H,dur = int(r[0]),int(r[1]),float(r[2])+2
hero_segs=[s for s in hmap if s['host']=='hero']
pip_segs =[s for s in hmap if s['host']=='pip' and s.get('bbox')]
def ee(segs):
    p=["between(t,%s,%s)"%(s['start'],s['end']) for s in segs]; return '+'.join(p) if p else '0'
hero_en=ee(hero_segs)
def cover(w,h): return "scale=%d:%d:force_original_aspect_ratio=increase,crop=%d:%d"%(w,h,w,h)

# clusters: median bbox per center-cell (stable, pas de jitter/balloon)
cells={}
for s in pip_segs:
    fx,fy,fw,fh=s['bbox']; cx=fx+fw/2; cy=fy+fh/2
    cell=(min(3,int(cx*4)), min(3,int(cy*4)))
    cells.setdefault(cell,[]).append(s)
med=statistics.median
clusters=[]
for cell,segs in cells.items():
    fx=med([s['bbox'][0] for s in segs]); fy=med([s['bbox'][1] for s in segs])
    fw=med([s['bbox'][2] for s in segs]); fh=med([s['bbox'][3] for s in segs])
    x=int(fx*W); y=int(fy*H); w=int(fw*W); h=int(fh*H)
    pad=int(min(w,h)*0.06)
    x=max(0,x-pad); y=max(0,y-pad); w=min(w+2*pad,W-x); h=min(h+2*pad,H-y)
    w-=w%2; h-=h%2
    clusters.append({'x':x,'y':y,'w':w,'h':h,'segs':segs})

G=10  # marge glow exterieur
# generer masque arrondi (alpha) + bordure verte arrondie par cluster
for i,c in enumerate(clusters):
    w,h=c['w'],c['h']; rad=int(min(w,h)*0.14)
    # masque alpha: blanc arrondi sur noir (L)
    m=Image.new('L',(w,h),0); d=ImageDraw.Draw(m)
    d.rounded_rectangle([0,0,w-1,h-1], radius=rad, fill=255)
    m.save('%s/mask%d.png'%(mdir,i))
    # bordure verte arrondie avec glow (RGBA), taille w+2G x h+2G
    bw,bh=w+2*G,h+2*G; b=Image.new('RGBA',(bw,bh),(0,0,0,0)); db=ImageDraw.Draw(b)
    rad2=rad+G
    # halo exterieur faible -> bright interieur
    db.rounded_rectangle([0,0,bw-1,bh-1], radius=rad2, outline=(0,255,0,40), width=10)
    db.rounded_rectangle([G-3,G-3,bw-G+2,bh-G+2], radius=rad+3, outline=(0,255,0,120), width=5)
    db.rounded_rectangle([G,G,bw-G-1,bh-G-1], radius=rad, outline=(0,255,0,235), width=3)
    b.save('%s/border%d.png'%(mdir,i))

nc=len(clusters)
# inputs: 0=source 1=avatar  puis pour chaque cluster: mask, border
inputs = "-i %s -stream_loop -1 -t %d -i %s" % (source, int(dur), avatar)
idx=2; mask_idx=[]; bord_idx=[]
for i in range(nc):
    inputs += " -i %s/mask%d.png -i %s/border%d.png" % (mdir,i,mdir,i)
    mask_idx.append(idx); bord_idx.append(idx+1); idx+=2

fc = "[1:v] fps=60,split=%d %s;" % (1+nc, ''.join("[a%d]"%i for i in range(1+nc)))
fc += "[a0] %s [av_hero];" % cover(W,H)
for i,c in enumerate(clusters):
    # avatar cover-crop puis applique le masque arrondi
    fc += "[a%d] %s,format=rgba [sc%d];" % (i+1, cover(c['w'],c['h']), i)
    fc += "[sc%d][%d:v] alphamerge [pr%d];" % (i, mask_idx[i], i)

fc += "[0:v][av_hero] overlay=0:0:enable='%s' [v0];" % hero_en
last='v0'
for i,c in enumerate(clusters):
    en=ee(c['segs']); x,y=c['x'],c['y']
    t1="vr%d"%i; t2="vb%d"%i
    fc += "[%s][pr%d] overlay=%d:%d:enable='%s' [%s];" % (last,i,x,y,en,t1)
    fc += "[%s][%d:v] overlay=%d:%d:enable='%s' [%s];" % (t1, bord_idx[i], x-G, y-G, en, t2)
    last=t2
fc += "[%s] null [vout]" % last

cmd=("ffmpeg -y "+inputs+
     " -filter_complex \""+fc+"\" -map \"[vout]\" -map 0:a -c:v libx264 -preset fast -crf 20 -c:a copy "+outp)
open(workdir+'/run_compositor.sh','w').write("#!/bin/bash\nset -e\nexec > "+workdir+"/compositor.log 2>&1\necho '=== START ==='\n"+cmd+"\necho '==== DONE ===='\nls -lh "+outp+"\n")
os.chmod(workdir+'/run_compositor.sh',0o755)
print("hero=%d pip=%d clusters=%d ROUNDED corners + green border"%(len(hero_segs),len(pip_segs),nc))
for i,c in enumerate(clusters): print("  c%d %dx%d @(%d,%d) r=%d"%(i,c['w'],c['h'],c['x'],c['y'],int(min(c['w'],c['h'])*0.14)))
