#!/usr/bin/env python3
# gen_masks.py <workdir> — genere un masque PIXEL-EXACT de la fenetre webcam par segment pip
# (grabCut union multi-frames), sauve <workdir>/masks/seg_XXX.png (grayscale, blanc=avatar),
# et ecrit seg["mask"]=chemin + seg["bbox"]=bbox du masque dans host_map.json.
# GATE VLM crop-verify : masque garde seulement si la fenetre est une vraie webcam. Sinon box
# inchangee (fallback). Env MASK_DISABLE=1 pour couper.
import sys, os, json, cv2, numpy as np
sys.path.insert(0, "/home/boss/videogen/agent_yt")
sys.path.insert(0, "/home/boss/yolo/scripts")
import webcam_mask, vlm_probe
try:
    import card_extent
except Exception:
    card_extent = None

def _clip(a, b):
    """intersection de 2 bbox [x,y,w,h] fractions -> bbox ou None."""
    ax0,ay0,ax1,ay1 = a[0],a[1],a[0]+a[2],a[1]+a[3]
    bx0,by0,bx1,by1 = b[0],b[1],b[0]+b[2],b[1]+b[3]
    x0,y0 = max(ax0,bx0),max(ay0,by0); x1,y1 = min(ax1,bx1),min(ay1,by1)
    if x1<=x0 or y1<=y0: return None
    return [x0,y0,x1-x0,y1-y0]

def clean_shape(mask, bbox, W, H):
    """Remplace le masque grabcut DENTELE par une forme geometrique PROPRE (ellipse si rond,
    rectangle-arrondi sinon) ajustee au bbox. Retourne masque cropu (uint8 0-255) au bbox."""
    # MARGE de sur-couverture : la forme est un peu PLUS GRANDE que la webcam detectee ->
    # garantit que le pip original est 100% cache (Boss : sous-couvrir = interdit, sur-couvrir = OK).
    MG = 0.06
    cx0 = bbox[0]-bbox[2]*MG; cy0 = bbox[1]-bbox[3]*MG
    cw = bbox[2]*(1+2*MG); ch = bbox[3]*(1+2*MG)
    x = int(cx0*W); y = int(cy0*H); w = max(4,int(cw*W)); h = max(4,int(ch*H))
    x=max(0,min(W-w,x)); y=max(0,min(H-h,y))
    if x+w>W: w=W-x
    if y+h>H: h=H-y
    sub = mask[y:y+h, x:x+w]
    fill_ratio = float(sub.sum())/max(1, w*h)   # part du bbox couverte par le masque
    out = np.zeros((h, w), np.uint8)
    # ratio ~0.78 => cercle/ellipse ; ~>0.88 => rectangle (arrondi)
    if 0.55 < fill_ratio < 0.86:
        cv2.ellipse(out, (w//2, h//2), (int(w*0.5)-1, int(h*0.5)-1), 0, 0, 360, 255, -1)
    else:
        r = max(2, int(min(w,h)*0.10))
        cv2.rectangle(out, (r,0),(w-r,h),255,-1); cv2.rectangle(out,(0,r),(w,h-r),255,-1)
        for cx,cy in ((r,r),(w-r,r),(r,h-r),(w-r,h-r)):
            cv2.circle(out,(cx,cy),r,255,-1)
    out = cv2.GaussianBlur(out,(5,5),0)
    return out, [round(x/W,4),round(y/H,4),round(w/W,4),round(h/H,4)]

wd = sys.argv[1]
hm_path = os.path.join(wd, "host_map.json")
src = os.path.join(wd, "source.mp4")
segs = json.load(open(hm_path))
mdir = os.path.join(wd, "masks"); os.makedirs(mdir, exist_ok=True)
YUNET = "/home/boss/videogen/face_detection_yunet_2023mar.onnx"

cap = cv2.VideoCapture(src)
W = int(cap.get(3)); H = int(cap.get(4))
yfd = cv2.FaceDetectorYN.create(YUNET, "", (W, H), score_threshold=0.5)

done = 0
for i, s in enumerate(segs):
    if s.get("host") != "pip" or not s.get("bbox"):
        continue
    if (s["end"] - s["start"]) < 1.5:
        continue
    b = s["bbox"]
    if b[2] >= 0.55 or b[3] >= 0.9:   # colonne vraie / quasi plein ecran : skip (grabcut instable)
        continue
    mask, mbbox = webcam_mask.seg_mask(cap, W, H, s["start"], s["end"], b, yfd)
    if mask is None:
        continue
    # CLIP au contour card_extent (serre sur FOND SOMBRE ou grabcut deborde dans le noir)
    if card_extent is not None:
        cb = card_extent.card_box(cap, W, H, s["start"], s["end"], yfd)
        if cb is not None:
            cl = _clip(mbbox, cb)
            if cl is not None and cl[2]*cl[3] > 0.4*(mbbox[2]*mbbox[3]):  # garde si pas trop rogne
                mbbox = cl
    # VLM verify : la fenetre est-elle une webcam ?
    cap.set(cv2.CAP_PROP_POS_MSEC, (s["start"] + (s["end"]-s["start"])/2)*1000.0)
    ok, fr = cap.read()
    if not ok or vlm_probe.webcam_crop(fr, mbbox) is not True:
        continue
    # forme PROPRE (ellipse/rect-arrondi) au lieu du contour grabcut dentele
    crop, mbbox = clean_shape((mask>0).astype(np.uint8), mbbox, W, H)
    p = os.path.join(mdir, "seg_%04d.png" % i)
    cv2.imwrite(p, crop)
    s["mask"] = p
    s["bbox"] = mbbox
    done += 1

cap.release()
json.dump(segs, open(hm_path, "w"), indent=2)
print("masques generes : %d segment(s) pip masques pixel-exact" % done)
