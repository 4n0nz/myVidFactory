#!/usr/bin/env python3
# qc_fid_fix.py <workdir> — applique TOUS les verdicts du juge de fidelite.
#
# Boss 2026-07-28 : « arrange pour qu il le corrige ». Les quatre verdicts sont donc
# tous actionnables, plus rien ne remonte a l oeil :
#   TROP-GRAND / SOUS-COUVERTURE -> box = carte mesuree, posee en autorite
#   HERO-RATE                    -> la scene devient HERO plein cadre (region=hero)
#   FAUX-PIP / PAS-NARRATEUR     -> la scene est RETIREE : pin_render la rendra OFF,
#                                   l image reste intacte (doctrine Boss du 27/07)
#
# Pas d oscillation possible : toutes les cibles viennent de la SOURCE, qui ne change
# jamais d un tour a l autre.
import json, os, sys

wd = sys.argv[1]
fails = json.load(open(os.path.join(wd, 'qc_fid.json')))
pinf = os.path.join(wd, 'host_map_pin.json')
pin = json.load(open(pinf))

n_box = n_hero = 0
drop = []
for f in fails:
    t = (f['t0'] + f['t1']) / 2.0
    for i, s in enumerate(pin):
        if s['start'] <= t <= s['end']:
            if f['type'] in ('TROP-GRAND', 'SOUS-COUVERTURE') and f.get('card'):
                if s.get('region') == 'hero': break
                s['box'] = list(f['card'])
                s['patched'] = True
                s['patched_keep'] = True    # autorite : le consensus ne l ecrase pas
                s['fid'] = True
                n_box += 1
            elif f['type'] == 'HERO-RATE':
                s['region'] = 'hero'
                s['fid'] = True
                n_hero += 1
            elif f['type'] in ('FAUX-PIP', 'PAS-NARRATEUR'):
                drop.append(i)
            break

# retrait des scenes a ne pas toucher (trou dans host_map_pin = OFF chez pin_render)
for i in sorted(set(drop), reverse=True):
    pin.pop(i)

json.dump(pin, open(pinf, 'w'), indent=1)
print('qc_fid_fix : %d box alignees sur la carte, %d scene(s) -> HERO, %d scene(s) -> OFF'
      % (n_box, n_hero, len(set(drop))))
