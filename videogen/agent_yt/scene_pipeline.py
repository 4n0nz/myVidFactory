
#!/usr/bin/env python3
# Pipeline scene-par-scene : detect coupes -> split -> classe -> traite -> concat -> mux audio
import subprocess, json, os, sys, statistics, re
import cv2
import mediapipe as mp
from mediapipe.tasks import python as P
from mediapipe.tasks.python import vision as V
from PIL import Image, ImageDraw

WORK   = sys.argv[1] if len(sys.argv)>1 else '/home/anon/videogen/agent_yt3'
OUTNAME= sys.argv[2] if len(sys.argv)>2 else 'KzObeom88Y_remix.mp4'
SRC    = WORK+'/source.mp4'
H264   = WORK+'/source_h264.mp4'           # 854x480, decode rapide pour detection
AVATAR = '/home/anon/videogen/public/avatar.mp4'
OUT    = '/home/anon/videogen/out/'+OUTNAME
SD     = WORK+'/scenes'; os.makedirs(SD, exist_ok=True)
MD     = WORK+'/masks';  os.makedirs(MD, exist_ok=True)
MODEL  = '/home/anon/videogen/efficientdet.tflite'

def probe(path, stream='v:0', fields='width,height,r_frame_rate,duration'):
    return subprocess.check_output('ffprobe -v error -select_streams %s -show_entries stream=%s -of csv=p=0 %s'%(stream,fields,path), shell=True).decode().strip()

vr = probe(SRC).split(',')
W,H = int(vr[0]), int(vr[1])
num,den = vr[2].split('/'); FPS = round(float(num)/float(den), 3)
DUR = float(subprocess.check_output('ffprobe -v error -show_entries format=duration -of csv=p=0 '+SRC, shell=True).decode().strip())
print("SRC %dx%d %.3ffps %.1fs"%(W,H,FPS,DUR))

STAGE = sys.argv[3] if len(sys.argv)>3 else 'all'

# ---------- STAGE 1: detection des coupes ----------
def detect_cuts(thresh=0.30):
    print("[1] scene detection...")
    cmd = "ffmpeg -i %s -filter:v \"select='gt(scene,%s)',showinfo\" -f null - 2>&1" % (H264, thresh)
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True).stderr + subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout
    # re-run capturing stderr properly
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    txt = p.stdout + p.stderr
    cuts = sorted(set(float(m) for m in re.findall(r'pts_time:([0-9.]+)', txt)))
    # boundaries: 0 .. cuts .. DUR ; drop cuts too close (<0.8s apart)
    bounds=[0.0]
    for c in cuts:
        if c-bounds[-1] >= 0.8: bounds.append(round(c,3))
    if DUR-bounds[-1] >= 0.5: bounds.append(round(DUR,3))
    else: bounds[-1]=round(DUR,3)
    json.dump(bounds, open(WORK+'/cuts.json','w'))
    print("    %d scenes"%(len(bounds)-1))
    return bounds

# ---------- STAGE 2: split en clips video (un seul passage) ----------
def split_scenes(bounds):
    print("[2] split scenes (one pass)...")
    times = ",".join("%.3f"%b for b in bounds[1:-1])  # internal cut points
    for f in os.listdir(SD):
        if f.startswith('s_') or f.startswith('o_'): os.remove(SD+'/'+f)
    cmd = ("ffmpeg -y -i %s -an -c:v libx264 -preset fast -crf 20 -pix_fmt yuv420p "
           "-force_key_frames %s -f segment -segment_times %s -reset_timestamps 1 "
           "-r %s %s/s_%%03d.mp4" % (SRC, times, times, FPS, SD))
    subprocess.run(cmd, shell=True, check=True)
    clips = sorted([f for f in os.listdir(SD) if f.startswith('s_')])
    print("    %d clips"%len(clips))
    return clips

# ---------- STAGE 3: classer chaque scene ----------
def classify(bounds):
    print("[3] classify scenes...")
    det=V.ObjectDetector.create_from_options(V.ObjectDetectorOptions(
        base_options=P.BaseOptions(model_asset_path=MODEL), score_threshold=0.30,
        category_allowlist=['person'], max_results=5))
    cap=cv2.VideoCapture(H264); fps=cap.get(cv2.CAP_PROP_FPS) or 30
    hw=int(cap.get(3)); hh=int(cap.get(4))
    scenes=[]
    for i in range(len(bounds)-1):
        s0,s1=bounds[i],bounds[i+1]
        # echantillonne jusqu'a 5 frames au milieu de la scene
        ts=[s0+(s1-s0)*f for f in (0.3,0.45,0.55,0.7,0.85)]
        votes=[]; bbs=[]
        for t in ts:
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(t*fps)); ok,fr=cap.read()
            if not ok: continue
            rgb=cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)
            res=det.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb))
            if res.detections:
                b=max(res.detections,key=lambda d:d.bounding_box.width*d.bounding_box.height).bounding_box
                fx=max(0,b.origin_x/hw); fy=max(0,b.origin_y/hh); fw=min(b.width/hw,1-fx); fh=min(b.height/hh,1-fy)
                area=fw*fh; cx=fx+fw/2
                if area>=0.28: h='hero'
                elif 0.28<=cx<=0.72 and area>=0.06: h='hero'
                elif 0.015<=area<0.22: h='pip'
                else: h='off'
                votes.append(h); bbs.append((fx,fy,fw,fh,h))
            else:
                votes.append('off')
        host = max(set(votes), key=votes.count) if votes else 'off'
        bbox=None
        if host=='pip':
            pbs=[b for b in bbs if b[4]=='pip']
            if pbs:
                bbox=[round(statistics.median([b[j] for b in pbs]),4) for j in range(4)]
        scenes.append({'idx':i,'start':s0,'end':s1,'host':host,'bbox':bbox})
    cap.release(); det.close()
    json.dump(scenes, open(WORK+'/scenes.json','w'), indent=2)
    hc=sum(1 for s in scenes if s['host']=='hero'); pc=sum(1 for s in scenes if s['host']=='pip'); oc=sum(1 for s in scenes if s['host']=='off')
    print("    %d scenes  hero=%d pip=%d off=%d"%(len(scenes),hc,pc,oc))
    return scenes

# ---------- masque arrondi + bordure ----------
def make_mask(w,h,idx):
    rad=int(min(w,h)*0.14); G=10
    m=Image.new('L',(w,h),0); ImageDraw.Draw(m).rounded_rectangle([0,0,w-1,h-1],radius=rad,fill=255)
    m.save('%s/mask%d.png'%(MD,idx))
    bw,bh=w+2*G,h+2*G; b=Image.new('RGBA',(bw,bh),(0,0,0,0)); db=ImageDraw.Draw(b)
    db.rounded_rectangle([0,0,bw-1,bh-1],radius=rad+G,outline=(0,255,0,40),width=10)
    db.rounded_rectangle([G-3,G-3,bw-G+2,bh-G+2],radius=rad+3,outline=(0,255,0,120),width=5)
    db.rounded_rectangle([G,G,bw-G-1,bh-G-1],radius=rad,outline=(0,255,0,235),width=3)
    b.save('%s/border%d.png'%(MD,idx))
    return G

# ---------- STAGE 4: traiter chaque scene ----------
def process(scenes):
    print("[4] process scenes...")
    enc="-c:v libx264 -preset fast -crf 20 -pix_fmt yuv420p -r %s -video_track_timescale 90000"%FPS
    for s in scenes:
        i=s['idx']; dur=s['end']-s['start']; src_clip="%s/s_%03d.mp4"%(SD,i); outc="%s/o_%03d.mp4"%(SD,i)
        if s['host']=='hero':
            # avatar plein ecran (cover-crop centre)
            cmd=("ffmpeg -y -stream_loop -1 -t %.3f -i %s -vf \"fps=%s,scale=%d:%d:force_original_aspect_ratio=increase,crop=%d:%d\" %s %s"
                 %(dur,AVATAR,FPS,W,H,W,H,enc,outc))
        elif s['host']=='pip' and s['bbox']:
            fx,fy,fw,fh=s['bbox']; x=int(fx*W); y=int(fy*H); w=int(fw*W); h=int(fh*H)
            pad=int(min(w,h)*0.06); x=max(0,x-pad); y=max(0,y-pad); w=min(w+2*pad,W-x); h=min(h+2*pad,H-y); w-=w%2; h-=h%2
            G=make_mask(w,h,i)
            fc=("[1:v]fps=%s,scale=%d:%d:force_original_aspect_ratio=increase,crop=%d:%d,format=rgba[av];"
                "[av][2:v]alphamerge[pr];"
                "[0:v][pr]overlay=%d:%d[v1];[v1][3:v]overlay=%d:%d[vout]"
                %(FPS,w,h,w,h,x,y,x-G,y-G))
            cmd=("ffmpeg -y -i %s -stream_loop -1 -t %.3f -i %s -i %s/mask%d.png -i %s/border%d.png "
                 "-filter_complex \"%s\" -map \"[vout]\" %s %s"
                 %(src_clip,dur,AVATAR,MD,i,MD,i,fc,enc,outc))
        else:  # off -> copie le clip source (re-encode identique pour concat safe)
            cmd="ffmpeg -y -i %s %s %s"%(src_clip,enc,outc)
        r=subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if r.returncode!=0:
            print("   ERR scene %d (%s): %s"%(i,s['host'],r.stderr[-200:]))
        else:
            print("   scene %03d %-4s %.1f-%.1fs OK"%(i,s['host'],s['start'],s['end']))

# ---------- STAGE 5: concat + audio ----------
def assemble(scenes):
    print("[5] concat + audio...")
    lst=WORK+'/concat.txt'
    with open(lst,'w') as f:
        for s in scenes:
            f.write("file '%s/o_%03d.mp4'\n"%(SD,s['idx']))
    subprocess.run("ffmpeg -y -f concat -safe 0 -i %s -c copy %s/_video.mp4"%(lst,WORK), shell=True, check=True)
    subprocess.run("ffmpeg -y -i %s/_video.mp4 -i %s -map 0:v -map 1:a -c copy %s"%(WORK,SRC,OUT), shell=True, check=True)
    print("    DONE -> "+OUT)

# ---------- main ----------
if STAGE in ('all','cuts'):
    bounds=detect_cuts()
else:
    bounds=json.load(open(WORK+'/cuts.json'))
if STAGE in ('all','split'): split_scenes(bounds)
if STAGE in ('all','classify'): scenes=classify(bounds)
else:
    try: scenes=json.load(open(WORK+'/scenes.json'))
    except: scenes=None
if STAGE in ('all','process') and scenes: process(scenes)
if STAGE in ('all','assemble') and scenes: assemble(scenes)
print("==== STAGE %s DONE ===="%STAGE)
