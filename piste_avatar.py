#!/usr/bin/env python3
# piste_avatar.py <sortie.mp4> <audio> --clips a.webm,b.webm --dur S [--mode banque|boucle]
#                 [--offset S] [--force 1.0]
#
# Produit une piste avatar de la duree EXACTE d une video, pilotee par la voix.
#
# Pourquoi une piste et pas une boucle : vf_avatar.py fait "-stream_loop -1" sur
# public/avatar.mp4 — l avatar tourne en rond, a vitesse fixe, sans aucun lien avec
# l audio. En fournissant une piste deja synchronisee ET plus longue que la video, la
# boucle infinie ne se declenche jamais : le pipeline n a pas besoin d etre modifie.
#
# Deux modes a comparer :
#   banque : le narrateur CHANGE de pose au fil de la video (mains jointes, clavier...)
#   boucle : une seule pose du debut a la fin, seule la vitesse suit la voix
#
# --offset : vf_finalize insere 1,5 s de neige AVANT l image. L audio de reference (pris
# sur le montage final) est donc en avance de 1,5 s sur le master vert ou l avatar se pose.
#
# Le fond est compose en NOIR : vf_avatar pose l avatar en cover dans les masques, donc
# son fond est visible ; l alpha reste le master pour composer ailleurs.
import os, subprocess, sys, wave
import cv2, numpy as np

DST, AUDIO = sys.argv[1], sys.argv[2]


def arg(n, d=None):
    return sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d


CLIPS = [c for c in (arg('--clips', '') or '').split(',') if c]
DUR = float(arg('--dur', '60'))
MODE = arg('--mode', 'banque')
OFFSET = float(arg('--offset', '0'))
FORCE = float(arg('--force', '1.0'))
# --fond : decor pose DERRIERE l avatar, dans sa fenetre. Sans lui, fond noir comme avant.
# A ne pas confondre avec le background du montage (Matrix), gere par vf_finalize.
FOND = arg('--fond')
DECOR = cv2.imread(FOND) if FOND else None
if FOND and DECOR is None:
    sys.exit('decor illisible : %s' % FOND)
FPS, SR = 30, 16000
if not CLIPS:
    sys.exit('--clips requis')

# ---------------------------------------------------------------- enveloppe de la voix
wav = '/tmp/piste_%d.wav' % os.getpid()
ss = ['-ss', '%.3f' % OFFSET] if OFFSET > 0 else []
subprocess.run(['ffmpeg', '-y', '-v', 'error'] + ss + ['-i', AUDIO, '-t', '%.3f' % DUR,
                '-ac', '1', '-ar', str(SR), wav], check=True)
with wave.open(wav) as w:
    pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
NF = int(round(DUR * FPS))
pf = max(1, len(pcm) // NF)
rms = np.array([np.sqrt(np.mean(pcm[i*pf:(i+1)*pf]**2) + 1e-12) for i in range(NF)])
env = np.power(rms, 0.5)
lo, hi = np.percentile(env, 5), np.percentile(env, 95)
env = np.clip((env - lo) / max(1e-6, hi - lo), 0, 1)
env = np.convolve(env, np.ones(9) / 9, mode='same')
os.remove(wav)
print('%s | %.0f s | voix active %.0f%% du temps' % (MODE, DUR, 100 * (env > 0.25).mean()))

# ---------------------------------------------------------------- clips en memoire (BGR)
banque = []
for c in CLIPS:
    d = '/tmp/pa_%d' % abs(hash(c))
    subprocess.run(['rm', '-rf', d]); os.makedirs(d)
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-c:v', 'libvpx-vp9', '-i', c,
                    '-vf', 'format=rgba', '%s/f%%04d.png' % d], check=True)
    frames = []
    for f in sorted(os.listdir(d)):
        im = cv2.imread(os.path.join(d, f), cv2.IMREAD_UNCHANGED)
        if im.shape[2] == 4:
            # le personnage est detoure : on le pose sur le DECOR (--fond), ou sur du noir
            # a defaut. C est le fond DERRIERE l avatar, dans sa fenetre — a ne pas
            # confondre avec le background du montage (Matrix), qui reste gere par
            # vf_finalize et n a pas a changer.
            a = im[:, :, 3:4].astype(np.float32) / 255.0
            h_, w_ = im.shape[:2]
            fond = (cv2.resize(DECOR, (w_, h_), interpolation=cv2.INTER_AREA)
                    if DECOR is not None else np.zeros((h_, w_, 3), np.uint8))
            im = (im[:, :, :3].astype(np.float32) * a +
                  fond.astype(np.float32) * (1 - a)).astype(np.uint8)
        frames.append(im)
    subprocess.run(['rm', '-rf', d])
    banque.append(frames)
    print('  %-34s %d images' % (os.path.basename(c), len(frames)))
H, W = banque[0][0].shape[:2]

# ---------------------------------------------------------------- frontieres de segments
# Mode "segments" (consigne Boss) : la pose ne change QU AUX FRONTIERES du montage —
# passage hero<->pip, ou changement de scene. Changer au bout d un clip (mode banque)
# donnait 559 bascules sur 12 min, souvent au milieu d une phrase.
# host_map.json est en temps du MASTER VERT, comme la piste avatar : aucun decalage ici.
# L OFFSET ne concerne que l audio, pris sur le montage final ou la neige decale tout.
FRONTIERES = []
HM = arg('--segments')
if HM and os.path.exists(HM):
    import json
    segs = sorted(json.load(open(HM)), key=lambda s: s['start'])
    # On ne change de pose QUE lorsque le cadrage bascule hero <-> pip (consigne Boss).
    # Prendre toutes les frontieres de segments donnait 90 changements sur MS7, y compris
    # entre deux scenes hero consecutives ou l image de l avatar ne change pas de nature :
    # ca se voyait comme une saute. Ici : 36 changements, un toutes les ~15 s.
    # Les segments 'off' (narrateur absent) sont ignores pour la comparaison : hero -> off
    # -> hero n est pas une bascule, c est la meme situation qui reprend.
    vis = [s for s in segs if s['host'] in ('hero', 'pip')]
    chg = [vis[i] for i in range(1, len(vis)) if vis[i]['host'] != vis[i - 1]['host']]

    # --coupures : les instants ou le MONTAGE coupe deja (detectes par scene detection).
    # Consigne Boss : le narrateur doit simplement parler ; quand on change de pose, on le
    # fait sur une coupure existante — l oeil accepte deja un changement a cet instant,
    # donc une coupe franche y passe inapercue (et le fondu croise devient inutile).
    # ESPACEMENT impose une duree minimale de parole entre deux changements : sans lui, on
    # basculerait a chacune des 78 coupures de la video, soit une toutes les 10 s.
    CP = arg('--coupures')
    if CP and os.path.exists(CP):
        cuts = sorted(float(x) for x in open(CP).read().split() if x.strip())
        esp = float(arg('--espacement', '45'))
        gardees, dernier = [], -1e9
        for t in cuts:
            if 0 < t < DUR and t - dernier >= esp:
                gardees.append(t); dernier = t
        FRONTIERES = sorted({int(round(t * FPS)) for t in gardees})
        print('  %d coupures dans le montage -> %d changements (1 toutes les %.0f s minimum)'
              % (len(cuts), len(FRONTIERES), esp))
        chg = None      # on ignore les bascules hero/pip : les coupures priment

    # RENDRE LE CHANGEMENT INVISIBLE (consigne Boss) : quand un passage 'off' (narrateur
    # absent de l image) tombe pres de la bascule, on y deplace le changement — personne
    # ne peut voir une pose changer si le personnage n est pas a l ecran. Sinon, le fondu
    # croise plus bas s en charge.
    offs = [s for s in segs if s['host'] == 'off']
    FRONTIERES, caches = FRONTIERES, 0
    for s in (chg or []):
        t = s['start']
        if not (0 < t < DUR):
            continue
        proche = [o for o in offs if abs((o['start'] + o['end']) / 2 - t) <= 3.0]
        if proche:
            o = min(proche, key=lambda o: abs((o['start'] + o['end']) / 2 - t))
            t = (o['start'] + o['end']) / 2      # au milieu du trou : marge des deux cotes
            caches += 1
        FRONTIERES.append(int(round(t * FPS)))
    FRONTIERES = sorted(set(FRONTIERES))
    if chg is not None:
        print('  %d segments (%d visibles) -> %d bascules hero<->pip, dont %d cachees dans un "off"'
              % (len(segs), len(vis), len(FRONTIERES), caches))

# ---------------------------------------------------------------- parcours
base = 0.08 + 0.92 * (1 - FORCE)
vit = base + (1.0 - base) * np.power(env, 0.8) * (1 + 0.8 * FORCE)
rng = np.random.default_rng(11)
fset = set(FRONTIERES)

# --mains <motif> : les clips dont le nom contient ce motif montrent les MAINS. Wan les
# reproduit mal, mais dans le pip la fenetre est trop petite pour que ca se voie. On les
# reserve donc au pip et on garde les bustes pour le plein cadre (consigne Boss).
MOTIF_MAINS = arg('--mains', '2021-07-20')
AVEC_MAINS = {i for i, c in enumerate(CLIPS) if MOTIF_MAINS and MOTIF_MAINS in os.path.basename(c)}
SANS_MAINS = [i for i in range(len(banque)) if i not in AVEC_MAINS]
if AVEC_MAINS and SANS_MAINS:
    print('  mains : %d clip(s) reserve(s) au pip | %d clip(s) pour le hero'
          % (len(AVEC_MAINS), len(SANS_MAINS)))

# type d hote image par image (hero / pip / off), pour savoir ce qui est permis a chaque instant
TYPE = ['off'] * NF
if HM and os.path.exists(HM):
    for s in segs:
        a, b = int(round(s['start'] * FPS)), min(NF, int(round(s['end'] * FPS)))
        for k in range(max(0, a), max(0, b)):
            TYPE[k] = s['host']


def permis(t):
    """clips utilisables a cet instant : pas de mains en plein cadre."""
    if t == 'hero' and SANS_MAINS:
        return SANS_MAINS
    return list(range(len(banque)))
# FONDU : duree du croise entre deux poses. Une bascule seche se voit meme au bon moment —
# le personnage saute d une position a l autre. Sur un fondu, comme c est le meme
# personnage dans le meme cadre sur le meme decor, seul le GESTE change : l oeil ne
# raccroche pas. 0,5 s est assez long pour etre doux, assez court pour rester net.
TRANS = int(round(float(arg('--fondu', '0.5')) * FPS))
ci, pos, sens, plan = 0, 0.0, 1, []
sortant = None            # (clip, position, images restantes) pendant un fondu
forces = 0
for i in range(NF):
    # une pose aux mains ne doit jamais passer en plein cadre : si le montage bascule en
    # hero, on change TOUT DE SUITE sans attendre une coupure — la bascule hero/pip est
    # elle-meme une rupture visuelle, le changement y passe.
    if ci in AVEC_MAINS and TYPE[i] == 'hero' and SANS_MAINS:
        sortant = [ci, pos, TRANS] if TRANS > 0 else None
        ci = int(rng.choice(SANS_MAINS))
        pos, sens = 0.0, 1
        forces += 1
    elif MODE == 'segments' and i in fset and len(banque) > 1:
        cand = [k for k in permis(TYPE[i]) if k != ci]
        if cand:
            sortant = [ci, pos, TRANS] if TRANS > 0 else None
            ci = int(rng.choice(cand))
            pos, sens = 0.0, 1
    n = len(banque[ci])
    pos += vit[i] * sens
    if pos >= n - 1 or pos <= 0:
        if MODE == 'banque' and len(banque) > 1:
            sortant = [ci, pos, TRANS] if TRANS > 0 else None
            ci = int(rng.choice([k for k in range(len(banque)) if k != ci]))
            pos = 0.0 if sens > 0 else len(banque[ci]) - 1.0
        else:
            # dans un segment (ou en mode boucle) : aller-retour sur la meme pose
            sens = -sens
            pos = max(0.0, min(n - 1.0, pos))

    if sortant and sortant[2] > 0:
        # l ancienne pose CONTINUE d avancer pendant le fondu : la figer produirait un
        # arret sur image au milieu de la transition, plus visible que la coupe elle-meme
        sortant[1] += vit[i]
        so = len(banque[sortant[0]])
        if sortant[1] >= so - 1:
            sortant[1] = so - 1.0
        alpha = 1.0 - sortant[2] / float(TRANS)          # 0 -> 1 sur la duree du fondu
        plan.append((ci, int(round(pos)), sortant[0], int(round(sortant[1])), alpha))
        sortant[2] -= 1
    else:
        sortant = None
        plan.append((ci, int(round(pos))))

# ---------------------------------------------------------------- ecriture directe
enc = ['h264_nvenc', '-preset', 'p5', '-rc', 'vbr', '-cq', '20', '-b:v', '0']
if subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i',
                   'color=c=red:s=320x180:r=30', '-t', '1', '-c:v', 'h264_nvenc',
                   '/tmp/nvp4.mp4'], capture_output=True).returncode != 0:
    enc = ['libx264', '-preset', 'fast', '-crf', '20']
p = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                      '-s', '%dx%d' % (W, H), '-r', str(FPS), '-i', '-', '-an',
                      '-c:v'] + enc + ['-pix_fmt', 'yuv420p', DST], stdin=subprocess.PIPE)
for e in plan:
    c, f = e[0], e[1]
    img = banque[c][min(f, len(banque[c]) - 1)]
    if len(e) == 5:                       # fondu croise entre l ancienne et la nouvelle pose
        c2, f2, a = e[2], e[3], e[4]
        img = cv2.addWeighted(img, a, banque[c2][min(f2, len(banque[c2]) - 1)], 1.0 - a, 0)
    p.stdin.write(img.tobytes())
p.stdin.close(); p.wait()

chg = sum(1 for i in range(1, len(plan)) if plan[i][0] != plan[i-1][0])
fondus = sum(1 for e in plan if len(e) == 5)
d = float(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
                                   'format=duration', '-of', 'csv=p=0', DST]))
mains_hero = sum(1 for i, e in enumerate(plan) if e[0] in AVEC_MAINS and TYPE[i] == 'hero')
print('%s : %.1f s, %dx%d | %d changements de pose (%d forces par un passage en hero) '
      '| %d images de fondu | %d images distinctes'
      % (os.path.basename(DST), d, W, H, chg, forces, fondus, len(set(e[:2] for e in plan))))
print('  controle : %d image(s) avec les mains en plein cadre (doit etre 0)' % mains_hero)
