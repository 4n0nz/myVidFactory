# myAvatarFactory

Génération du narrateur masqué de [myVidFactory](https://github.com/4n0nz/myVidFactory) :
une photo fond vert en entrée, une piste avatar animée et synchronisée sur la voix en sortie.

## Chaîne

1. **`bande_perso.py`** — une photo fond vert → clip animé (Wan 2.2 image-to-video) → chroma key → WebM alpha.
   Prompt par image : registre de narration pour les bustes, mouvement adapté pour celles qui montrent les mains.
2. **`piste_avatar.py`** — assemble les clips en une piste à la durée exacte d'une vidéo, pilotée par
   l'enveloppe de la voix. Trois modes : `segments` (change de pose aux frontières de `host_map.json`),
   `banque` (change au bout de chaque clip), `boucle` (une seule pose).
3. **`boucle60.py`** — boucle longue depuis un clip court, raccord exact sans fondu.
4. **`sync_alpha.py`** — montage synchronisé voix, sortie fond transparent.

## Réglages mesurés, à ne pas redécouvrir

- **Chroma key** : le fond des photos est du vert **pur** (`0x00FF00`), pas le vert normalisé
  `0x00B140`. Avec la mauvaise cible, le détourage efface **tout, sujet compris**.
  Similarité `0.18` → sujet 21,5 %, 100 % de pixels francs.
- **VP9 alpha** : sans `-auto-alt-ref 0 -lag-in-frames 0`, l'encodage réussit sans erreur mais
  sort en `yuv420p` — l'alpha est perdu **en silence**. Vérifier avec `alphaextract`.
- **Wan 2.2 sur 16 Go de VRAM** : `enable_model_cpu_offload()` **et** `vae.enable_tiling()` +
  `enable_slicing()`, sinon plantage mémoire au décodage final, après le calcul.
  Dépendances non tirées par diffusers : `ftfy`, `imageio-ffmpeg`.
- **Interpolation 24→30 im/s** : `minterpolate=fps=30:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1`.

## Approches écartées (et pourquoi)

- **`anim_rigid.py`** — transformation de l'image entière : mouvement de caméra, pas d'animation.
- **SVD** — anime le sujet mais **déforme le masque** (modèle de 2023, aucune notion de rigidité).
- **`head_turn.py`** — homographie sur la tête découpée : effet de drapeau, une tête n'est pas un plan.
- **`head_turn2.py`** — similitude stricte : rigide, mais déplace une découpe au lieu d'animer.
- **`LivePortrait`** — vraie pose 3D, fond immobile, rapide ; le geste vient d'une vidéo de conduite.
- **Modèles audio-driven** (SadTalker, Hallo, Sonic) — régénèrent le visage, donc déforment le
  masque ; leur lip sync est inutile ici puisque la bouche est couverte.
