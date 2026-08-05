#!/usr/bin/env python3
# vf_avatar.py <video_id> [--out nom.mp4] — remplace le vert par L'AVATAR (branche avatar).
#
#   Descendant direct de agent_yt/build_compositor.py (le compositor du debut du projet,
#   meme avatar public/avatar.mp4), reecrit pour la generation pinpoint :
#   - la BASE n est plus la source mais le MASTER VERT b_<id>.mp4 — la geometrie a deja
#     passe les 3 boucles QC et les juges ; l avatar se pose PILE sur le vert, et tout
#     vert residuel (desalignement, scene ratee) reste VISIBLE = signal de bug gratuit.
#   - les boxes ne sont plus des clusters medians recalcules : host_map.json fait foi,
#     et chaque groupe (bbox, masque) reutilise le PNG d alpha exact de build_seg
#     (coins arrondis compris) — zero jitter, zero desaccord avec le rendu vert.
#   - hero = avatar plein cadre en cover (comme a l epoque) ; pip = avatar en cover
#     dans la box, masque par l alpha de la scene.
#   L avatar (848x464, 147.9 s, muet) est boucle sur toute la duree ; l audio vient du
#   master. Sortie : out/av_<id>.mp4.
import json, os, subprocess, sys
from collections import OrderedDict

vid = sys.argv[1]
OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "av_%s.mp4" % vid
VG = "/home/boss/videogen"
WD = os.path.join(VG, "wk_b_" + vid)
BASE = os.path.join(VG, "out", "b_%s.mp4" % vid)
AVATAR = os.path.join(VG, "public", "avatar.mp4")
outp = os.path.join(VG, "out", OUT)

hm = json.load(open(os.path.join(WD, "host_map.json")))
r = subprocess.check_output(
    ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
     "stream=width,height", "-of", "csv=p=0", BASE], text=True).strip().split(",")
W, H = int(r[0]), int(r[1])

def en(segs):
    return "+".join("between(t,%.3f,%.3f)" % (s["start"], s["end"]) for s in segs)

# groupes pip par (bbox, masque) — un overlay par groupe, actif sur l union des scenes
groups = OrderedDict()
heroes = []
for s in hm:
    if s.get("host") == "pip" and s.get("bbox"):
        k = (tuple(s["bbox"]), s.get("mask") or "")
        groups.setdefault(k, []).append(s)
    elif s.get("host") == "hero":
        heroes.append(s)

inputs = ["-i", BASE, "-stream_loop", "-1", "-i", AVATAR]
flt = []
idx = 2
chain = "[0:v]"
step = 0
for (bbox, mask), segs in groups.items():
    x = int(bbox[0] * W); y = int(bbox[1] * H)
    w = max(2, int(bbox[2] * W)) // 2 * 2; h = max(2, int(bbox[3] * H)) // 2 * 2
    cover = "scale=%d:%d:force_original_aspect_ratio=increase,crop=%d:%d" % (w, h, w, h)
    if mask and os.path.exists(mask):
        inputs += ["-loop", "1", "-i", mask]
        flt.append("[1:v]%s,setsar=1[pv%d]" % (cover, step))
        flt.append("[%d:v]scale=%d:%d,format=gray[pm%d]" % (idx, w, h, step))
        flt.append("[pv%d][pm%d]alphamerge[pf%d]" % (step, step, step))
        idx += 1
    else:
        flt.append("[1:v]%s,setsar=1[pf%d]" % (cover, step))
    e = en(segs)
    flt.append("%s[pf%d]overlay=%d:%d:enable='%s'[o%d]" % (chain, step, x, y, e, step))
    # bordure 1 px #00ff00 autour de l avatar (ordre Boss 2026-08-05 — echo de la
    # bordure verte du compositor d origine), active sur les memes intervalles
    bx = max(0, x - 1); by = max(0, y - 1)
    bw = min(W - bx, w + 2); bh = min(H - by, h + 2)
    flt.append("[o%d]drawbox=%d:%d:%d:%d:color=0x00FF00@1:t=1:enable='%s'[v%d]"
               % (step, bx, by, bw, bh, e, step))
    chain = "[v%d]" % step
    step += 1
if heroes:
    e = en(heroes)
    cover = "scale=%d:%d:force_original_aspect_ratio=increase,crop=%d:%d" % (W, H, W, H)
    flt.append("[1:v]%s,setsar=1[hf]" % cover)
    flt.append("%s[hf]overlay=0:0:enable='%s'[oh]" % (chain, e))
    flt.append("[oh]drawbox=0:0:%d:%d:color=0x00FF00@1:t=1:enable='%s'[v%d]"
               % (W, H, e, step))
    chain = "[v%d]" % step

enc = ["-c:v", "h264_nvenc", "-preset", "p4", "-b:v", "8M"]
probe = subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                        "color=c=red:s=320x180:r=30", "-t", "1", "-c:v", "h264_nvenc",
                        "/tmp/nvenc_probe.mp4"], capture_output=True)
if probe.returncode != 0:
    enc = ["-c:v", "libx264", "-preset", "fast", "-crf", "20"]

print("avatar %s : %d groupes pip, %d scenes hero" % (vid, len(groups), len(heroes)))
cmd = (["ffmpeg", "-y", "-v", "error"] + inputs +
       ["-filter_complex", ";".join(flt), "-map", chain, "-map", "0:a?",
        "-shortest"] + enc + ["-c:a", "copy", outp])
rc = subprocess.run(cmd).returncode
if rc == 0 and os.path.getsize(outp) > 0:
    print("OK", outp)
else:
    print("AVATAR_FAIL rc=%d" % rc); sys.exit(1)
