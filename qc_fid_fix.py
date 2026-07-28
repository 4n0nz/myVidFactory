#!/usr/bin/env python3
# qc_fid_fix.py <workdir> — applique les corrections du juge de fidelite.
#
# TROP-GRAND / SOUS-COUVERTURE -> la box de la scene devient la CARTE MESUREE dans la
# source, posee en autorite (`patched_keep`, deja respecte par pin_render). Comme la
# cible vient de la SOURCE (invariante), il n y a pas d oscillation possible : c est
# ce qui autorise enfin une correction NON monotone, donc le retrecissement.
#
# PAS-DE-CARTE : PAS de correction automatique ici. Le vert ne repose sur aucune carte
# (contenu, ou hero rate type XzEg) — c est une decision de classification (hero/off),
# pas de geometrie. Le flag reste dans qc_fid.json et remonte au rapport pour qu il
# soit traite en amont (pinpoint3 / decision hero), jamais rustine par scene.
import json, os, sys

wd = sys.argv[1]
fails = json.load(open(os.path.join(wd, 'qc_fid.json')))
pinf = os.path.join(wd, 'host_map_pin.json')
pin = json.load(open(pinf))

n = 0
skipped = 0
for f in fails:
    if f['type'] == 'PAS-DE-CARTE' or not f.get('card'):
        skipped += 1
        continue
    t = (f['t0'] + f['t1']) / 2.0
    for s in pin:
        if s.get('region') == 'hero': continue
        if s['start'] <= t <= s['end']:
            s['box'] = list(f['card'])
            s['patched'] = True
            s['patched_keep'] = True   # autorite : le consensus ne l ecrase pas
            s['fid'] = True
            n += 1
            break

json.dump(pin, open(pinf, 'w'), indent=1)
print('qc_fid_fix : %d scene(s) alignee(s) sur la carte mesuree, %d PAS-DE-CARTE laissee(s) au rapport' % (n, skipped))
