#!/usr/bin/env python3
# vf_intro_outro.py <video> <sortie.mp4> [--intro F] [--outro F]
#                   [--intro-dur S] [--outro-dur S] [--fondu S]
#
# Dernier maillon : generique de debut + video + generique de fin.
#
# La VIDEO est la reference : intro et outro sont ramenes a SA definition, SA cadence et
# SON format de pixel. Sans cette normalisation, concat sort une video corrompue ou
# desynchronisee des que les sources different — et ici elles different (intro 1080p,
# outro 848x464).
#
# Le ratio est PRESERVE : mise a l echelle sans deformation puis bandes noires si besoin.
# Un scale brut etirerait l outro, dont le ratio n est pas celui de la video.
#
# L audio est obligatoire sur les trois segments : si une source est muette, on fabrique
# un silence de sa duree. concat exige le meme nombre de flux partout, sinon il echoue.
import os, subprocess, sys


def arg(n, d=None):
    return sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d


SRC = sys.argv[1]
DST = sys.argv[2]
ASSETS = os.path.expanduser('~/videogen/assets')
INTRO = arg('--intro', os.path.join(ASSETS, 'intro.mp4'))
OUTRO = arg('--outro', os.path.join(ASSETS, 'outro.mp4'))
INTRO_DUR = float(arg('--intro-dur', '0') or 0)      # 0 = garder toute l intro
OUTRO_DUR = float(arg('--outro-dur', '0') or 0)
FONDU = float(arg('--fondu', '0') or 0)              # fondu audio aux jointures


def sh(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()


def infos(p):
    w, h = sh(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
               'stream=width,height', '-of', 'csv=p=0', p]).split(',')[:2]
    d = float(sh(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                  '-of', 'csv=p=0', p]))
    fps = sh(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
              'stream=r_frame_rate', '-of', 'csv=p=0', p]).split(',')[0]
    a = bool(sh(['ffprobe', '-v', 'error', '-select_streams', 'a', '-show_entries',
                 'stream=index', '-of', 'csv=p=0', p]))
    return int(w), int(h), d, fps, a


for f in (SRC, INTRO, OUTRO):
    if not os.path.isfile(f):
        sys.exit('introuvable : %s' % f)

W, H, DUR, FPS, _ = infos(SRC)
iw, ih, idur, _, ia = infos(INTRO)
ow, oh, odur, _, oa = infos(OUTRO)
if INTRO_DUR:
    idur = min(idur, INTRO_DUR)
if OUTRO_DUR:
    odur = min(odur, OUTRO_DUR)
print('video  %dx%d  %.1fs  @%s' % (W, H, DUR, FPS))
print('intro  %dx%d  %.1fs%s' % (iw, ih, idur, '' if ia else '  (muette)'))
print('outro  %dx%d  %.1fs%s' % (ow, oh, odur, '' if oa else '  (muette)'))
for nom, w, h in (('intro', iw, ih), ('outro', ow, oh)):
    if w < W or h < H:
        print('  ! %s est plus petit que la video : il sera agrandi (%dx%d -> %dx%d)'
              % (nom, w, h, W, H))

# mise a l echelle sans deformation + bandes noires, cadence et SAR alignes sur la video
def norm(i, dur, audio, coupe):
    v = ('[%d:v]%sscale=%d:%d:force_original_aspect_ratio=decrease,'
         'pad=%d:%d:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=%s[v%d]'
         % (i, ('trim=0:%.3f,setpts=PTS-STARTPTS,' % dur) if coupe else '', W, H, W, H, FPS, i))
    if audio:
        a = ('[%d:a]%saformat=sample_rates=48000:channel_layouts=stereo[a%d]'
             % (i, ('atrim=0:%.3f,asetpts=PTS-STARTPTS,' % dur) if coupe else '', i))
    else:
        # concat exige un flux audio partout : on fabrique le silence manquant
        a = ('anullsrc=r=48000:cl=stereo,atrim=duration=%.3f,asetpts=PTS-STARTPTS[a%d]'
             % (dur, i))
    return v, a


parts = []
for i, (dur, audio, coupe) in enumerate([(idur, ia, bool(INTRO_DUR)),
                                         (DUR, True, False),
                                         (odur, oa, bool(OUTRO_DUR))]):
    parts.extend(norm(i, dur, audio, coupe))

chaine = '[v0][a0][v1][a1][v2][a2]concat=n=3:v=1:a=1[outv][outa0]'
if FONDU > 0:
    # adoucit les deux jointures sans toucher au corps de la video
    chaine += (';[outa0]afade=t=out:st=%.3f:d=%.3f,afade=t=in:st=%.3f:d=%.3f[outa]'
               % (idur - FONDU, FONDU, idur, FONDU))
    sortie_a = '[outa]'
else:
    sortie_a = '[outa0]'

filtre = ';'.join(parts) + ';' + chaine
with open('/tmp/vf_io_filtre.txt', 'w') as f:
    f.write(filtre)

enc = ['-c:v', 'h264_nvenc', '-preset', 'p5', '-rc', 'vbr', '-cq', '20', '-b:v', '0']
if subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i',
                   'color=c=red:s=320x180:r=30', '-t', '1', '-c:v', 'h264_nvenc',
                   '/tmp/nv_io.mp4'], capture_output=True).returncode != 0:
    enc = ['-c:v', 'libx264', '-preset', 'medium', '-crf', '18']

subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', INTRO, '-i', SRC, '-i', OUTRO,
                '-filter_complex_script', '/tmp/vf_io_filtre.txt',
                '-map', '[outv]', '-map', sortie_a] + enc +
               ['-c:a', 'aac', '-b:a', '192k', '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart', DST], check=True)

# la duree finale doit valoir la somme des trois : c est ce qui prouve qu aucun segment
# n a ete tronque par un flux plus court que les autres
attendu = idur + DUR + odur
reel = infos(DST)[2]
print('\n%s : %.1fs (attendu %.1fs, ecart %+.2fs)' % (DST, reel, attendu, reel - attendu))
if abs(reel - attendu) > 0.5:
    print('  ! ECART SUSPECT — un segment a probablement ete coupe')
else:
    print('  intro 0-%.1fs | video %.1f-%.1fs | outro %.1f-%.1fs'
          % (idur, idur, idur + DUR, idur + DUR, attendu))
