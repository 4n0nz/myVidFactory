#!/usr/bin/env python3
# pin_render.py <workdir> — construit host_map.json a partir de host_map_pin.json (box stables
# pinpoint) : chaque scene -> segment pip avec masque FORME PROPRE (ellipse/rect-arrondi) dessine
# au box pinpoint + marge de sur-couverture. Forme decidee par grabcut fill_ratio (1 passe).
# Trous entre scenes -> 'off'. Puis build_seg rend.
import sys, os, json, cv2, numpy as np
sys.path.insert(0, "/home/boss/videogen/agent_yt")
sys.path.insert(0, "/home/boss/yolo/scripts")
import webcam_mask, vlm_probe
YUNET = "/home/boss/videogen/face_detection_yunet_2023mar.onnx"
MG = 0.05

wd = sys.argv[1]
pin = json.load(open(os.path.join(wd, "host_map_pin.json")))
src = os.path.join(wd, "source.mp4")
cap = cv2.VideoCapture(src); W=int(cap.get(3)); H=int(cap.get(4)); DUR=cap.get(7)/(cap.get(5) or 30)
yfd = cv2.FaceDetectorYN.create(YUNET, "", (W, H), score_threshold=0.6)
mdir = os.path.join(wd, "masks"); os.makedirs(mdir, exist_ok=True)

def shape_of(t0, t1, box, allow_shrink=True):
    """(shape, box) : ellipse si le blob grabcut remplit ~78% du box (rond), sinon rect.
    fill < 0.45 = la box detectee SUR-couvre largement le blob reel (card_extent qui deborde
    sur page blanche : cercle rendu en rect geant, 0sqC 492-531s) -> RETRECIR la box au
    bounding-box du blob (+marge) et re-decider la forme. allow_shrink=False pour les scenes
    patchees par qc_fix (le shrink annulait leurs elargissements -> boucle QC sterile 6-6-6)."""
    m, _ = webcam_mask.seg_mask(cap, W, H, t0, t1, box, yfd)
    if m is None: return "rect", box
    def _fill(b):
        x=int(b[0]*W);y=int(b[1]*H);w=max(4,int(b[2]*W));h=max(4,int(b[3]*H))
        x=max(0,min(W-w,x));y=max(0,min(H-h,y))
        return float(m[y:y+h, x:x+w].sum())/max(1,w*h)
    fr = _fill(box)
    if fr < 0.45 and allow_shrink:
        ys, xs = np.nonzero(m)
        if len(xs) > 2000:
            mg = 0.05
            bx0,by0 = xs.min()/W, ys.min()/H; bx1,by1 = xs.max()/W, ys.max()/H
            bw,bh = bx1-bx0, by1-by0
            nb = [max(0.0,bx0-bw*mg), max(0.0,by0-bh*mg),
                  min(1.0,bw*(1+2*mg)), min(1.0,bh*(1+2*mg))]
            if nb[2] >= 0.04 and nb[3] >= 0.04:
                box = [round(v,4) for v in nb]; fr = _fill(box)
    return ("ellipse" if 0.55 < fr < 0.86 else "rect"), box

def draw(box, shape, idx):
    cx0=box[0]-box[2]*MG; cy0=box[1]-box[3]*MG; cw=box[2]*(1+2*MG); ch=box[3]*(1+2*MG)
    x=int(cx0*W);y=int(cy0*H);w=max(4,int(cw*W));h=max(4,int(ch*H))
    x=max(0,min(W-w,x));y=max(0,min(H-h,y));  w=min(w,W-x); h=min(h,H-y)
    out=np.zeros((h,w),np.uint8)
    if shape=="ellipse":
        cv2.ellipse(out,(w//2,h//2),(w//2-1,h//2-1),0,0,360,255,-1)
    else:
        r=max(2,int(min(w,h)*0.10))
        cv2.rectangle(out,(r,0),(w-r,h),255,-1);cv2.rectangle(out,(0,r),(w,h-r),255,-1)
        for a,b in ((r,r),(w-r,r),(r,h-r),(w-r,h-r)): cv2.circle(out,(a,b),r,255,-1)
    out=cv2.GaussianBlur(out,(5,5),0)
    p=os.path.join(mdir,"seg_%04d.png"%idx); cv2.imwrite(p,out)
    return p,[round(x/W,4),round(y/H,4),round(w/W,4),round(h/H,4)]

def is_hero(t0, t1, box):
    """HERO = grosse cam qui remplit le cadre -> avatar plein ecran. VLM fullface x2 + garde
    geometrique (la box couvre une grande part de l'ecran : large ET haute)."""
    bw, bh = box[2], box[3]
    big = bw > 0.5 and bh > 0.7           # box deja quasi plein cadre
    mid = (t0+t1)/2
    cap.set(cv2.CAP_PROP_POS_MSEC, mid*1000.0); ok, fr = cap.read()
    ff1 = vlm_probe.fullface(fr) if ok else None
    if ff1 is not True:
        return big and ff1 is not False
    cap.set(cv2.CAP_PROP_POS_MSEC, (t0+(t1-t0)*0.25)*1000.0); ok, fr = cap.read()
    ff2 = vlm_probe.fullface(fr) if ok else None
    return ff2 is True

segs=[]; prev=0.0
nhero=0
pips=[]
for sc in pin:
    if sc["start"]>prev+0.3:
        segs.append({"host":"off","start":round(prev,2),"end":round(sc["start"],2),"bbox":None})
    if sc.get("region") == "hero" or (sc.get("src") != "ident" and is_hero(sc["start"], sc["end"], sc["box"])):
        segs.append({"host":"hero","start":round(sc["start"],2),"end":round(sc["end"],2),"bbox":None})
        nhero+=1
    else:
        seg={"host":"pip","start":round(sc["start"],2),"end":round(sc["end"],2)}
        segs.append(seg)
        pips.append({"seg":seg,"t0":sc["start"],"t1":sc["end"],"box":sc["box"],
                     "patched":bool(sc.get("patched")) or sc.get("region")=="patch"})
    prev=sc["end"]

# forme + shrink par scene (cache par box canonique non-shrinkee)
_shape_cache={}
for p in pips:
    key=tuple(p["box"])
    if key in _shape_cache and not p["patched"]:
        p["shape"],p["abox"]=_shape_cache[key]
    else:
        shp,abox=shape_of(p["t0"],p["t1"],p["box"],allow_shrink=not p["patched"])
        p["shape"],p["abox"]=shp,list(abox)
        if abox == p["box"] and not p["patched"]:
            _shape_cache[key]=(shp,list(abox))

# UNIFICATION position+forme : le MEME pip source recoit LA MEME box finale partout.
# Sans ca, une scene garde la canonique et l'autre la version shrinkee -> saut de forme
# a la frontiere + bord du pip a decouvert (0sqC 485s, invisible au QC : pas un visage).
def _ctr(b): return (b[0]+b[2]/2, b[1]+b[3]/2)
groups=[]
for p in pips:
    if p["patched"]: continue
    cx,cy=_ctr(p["abox"]); hit=None
    for g in groups:
        if g["shape"]==p["shape"] and abs(cx-g["cx"])<0.06 and abs(cy-g["cy"])<0.06:
            hit=g; break
    if hit is None:
        groups.append({"shape":p["shape"],"cx":cx,"cy":cy,"box":list(p["abox"]),"members":[p]})
    else:
        b=hit["box"]; nb=p["abox"]
        x0=min(b[0],nb[0]); y0=min(b[1],nb[1])
        x1=max(b[0]+b[2],nb[0]+nb[2]); y1=max(b[1]+b[3],nb[1]+nb[3])
        hit["box"]=[x0,y0,x1-x0,y1-y0]; hit["members"].append(p)
        hit["cx"],hit["cy"]=_ctr(hit["box"])
for g in groups:
    for p in g["members"]:
        p["abox"]=[round(float(v),4) for v in g["box"]]

# dessin : UN masque par (box finale, forme)
_mask={}; mi=0
for p in pips:
    k=(tuple(p["abox"]),p["shape"])
    if k not in _mask:
        _mask[k]=draw(p["abox"],p["shape"],mi); mi+=1
    mp,bb=_mask[k]
    p["seg"].update({"bbox":bb,"mask":mp,"shape":p["shape"]})
print("heros: %d / masques uniques: %d / groupes position: %d" % (nhero, len(_mask), len(groups)))
if prev<DUR-0.3:
    segs.append({"host":"off","start":round(prev,2),"end":round(DUR,2),"bbox":None})

json.dump(segs,open(os.path.join(wd,"host_map.json"),"w"),indent=2)
print("host_map: %d segs (%d pip) genere depuis pinpoint"%(len(segs),sum(1 for s in segs if s["host"]=="pip")))
cap.release()
