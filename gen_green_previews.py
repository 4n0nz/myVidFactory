#!/usr/bin/env python3
# gen_green_previews.py <vid> [...] — extrait 30s par video avec le/les pip couverts en
# vert chroma (#00ff00) aux boxes CONSENSUS (box_consensus.json, a generer avant).
# Fenetre choisie sur la plus longue scene pip de host_map_pin qui matche un cluster.
import sys, os, json, subprocess

VG = "/home/boss/videogen"

def rounded_expr(w, h, r):
    W1 = w-r; H1 = h-r
    return ("255*(1-("
            "lt(X,%d)*lt(Y,%d)*gt(hypot(%d-X,%d-Y),%d)+"
            "gt(X,%d)*lt(Y,%d)*gt(hypot(X-%d,%d-Y),%d)+"
            "lt(X,%d)*gt(Y,%d)*gt(hypot(%d-X,Y-%d),%d)+"
            "gt(X,%d)*gt(Y,%d)*gt(hypot(X-%d,Y-%d),%d)))"
            % (r,r,r,r,r, W1,r,W1,r,r, r,H1,r,H1,r, W1,H1,W1,H1,r))

for vid in sys.argv[1:]:
    wd = "%s/wk_b_%s" % (VG, vid)
    try:
        cons = json.load(open(wd+"/box_consensus.json"))
        pin = json.load(open(wd+"/host_map_pin.json"))
    except Exception as e:
        print(vid, "SKIP", e); continue
    pips = [c for c in cons if c["box"][2]*c["box"][3] < 0.5]
    if not pips:
        print(vid, "SKIP aucun cluster pip"); continue
    # fenetre : plus longue scene non-hero qui matche un cluster pip
    best = None
    for s in pin:
        if s.get("region") == "hero": continue
        scx = s["box"][0]+s["box"][2]/2; scy = s["box"][1]+s["box"][3]/2
        for c in pips:
            ccx = c["box"][0]+c["box"][2]/2; ccy = c["box"][1]+c["box"][3]/2
            if abs(scx-ccx) < 0.25 and abs(scy-ccy) < 0.25:
                d = s["end"]-s["start"]
                if best is None or d > best[1]: best = (s["start"]+min(5, d*0.1), d)
    if best is None:
        print(vid, "SKIP aucune scene pip"); continue
    t0 = best[0]; dur = min(30, max(10, best[1]))
    r = subprocess.check_output(
        "ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 %s/source.mp4" % wd,
        shell=True).decode().strip().split(",")
    W, H = int(r[0]), int(r[1])
    inputs = "-ss %s -t %s -i %s/source.mp4" % (t0, dur, wd)
    fc = []; last = "0:v"
    for i, c in enumerate(pips):
        b = c["box"]
        x = int(b[0]*W); y = int(b[1]*H); w = int(b[2]*W)//2*2; h = int(b[3]*H)//2*2
        inputs += " -f lavfi -i color=c=0x00FF00:s=%dx%d:r=30" % (w, h)
        src = "%d:v" % (i+1)
        if c["shape"] == "rect":
            rr = max(2, int(min(w, h)*0.08))
            fc.append("[%s]format=rgba[g%d];color=c=black:s=%dx%d:r=30,format=gray,"
                      "geq=lum='%s',loop=loop=-1:size=1[m%d];[g%d][m%d]alphamerge[av%d]"
                      % (src, i, w, h, rounded_expr(w, h, rr), i, i, i, i))
        elif c["shape"] == "ellipse":
            fc.append("[%s]format=rgba[g%d];color=c=black:s=%dx%d:r=30,format=gray,"
                      "geq=lum='255*lt(hypot((X-%d)/%d,(Y-%d)/%d),1)',loop=loop=-1:size=1[m%d];"
                      "[g%d][m%d]alphamerge[av%d]"
                      % (src, i, w, h, w//2, w//2, h//2, h//2, i, i, i, i))
        else:
            fc.append("[%s]format=rgba[av%d]" % (src, i))
        fc.append("[%s][av%d]overlay=%d:%d:shortest=1[v%d]" % (last, i, x, y, i))
        last = "v%d" % i
    cmd = ('ffmpeg -y -v error %s -filter_complex "%s" -map [%s] -c:v libx264 -preset fast -crf 20 -an /tmp/green_%s.mp4'
           % (inputs, ";".join(fc), last, vid))
    subprocess.run(cmd, shell=True, check=True)
    print(vid, "OK t0=%.0f dur=%.0f boxes=%d" % (t0, dur, len(pips)))
