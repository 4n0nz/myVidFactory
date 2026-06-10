# Architecture

## Vue d'ensemble

myVidFactory transforme un script JSON en vidéo, par couches indépendantes assemblées avec ffmpeg. Le principe : **tout ce qu'un tuto doit montrer (texte, code, schémas) se rend en code** — pas de génération d'images par diffusion.

```
script.json ──► build_audio.py ──► public/audio/*.mp3 + render-manifest.json
            └─► gen_captions.py ──► captions.json
                                        │
render-manifest.json ──► Remotion "Tutorial"  ──► out/master.mp4   (visuels + voix)
captions.json        ──► Remotion "SubsTerminal" ─► out/term.mov   (PiP alpha)
                                        │
                              ffmpeg overlay ──► out/master_final.mp4
```

## Composants

### `build_audio.py`
Lit `script.json`. Pour chaque scène : génère la voix off avec edge-tts (`meta.voice`, `--rate`, `--pitch`), mesure la durée avec ffprobe, et calcule `durationInFrames` (= durée audio + 0,7 s de silence). Écrit `render-manifest.json` (scènes + durées + chemins audio). **La durée de chaque scène = la durée de sa voix off → synchro automatique.**

Override de voix : `python build_audio.py script.json fr-CA-SylvieNeural`.

### `gen_captions.py`
Sous-titres **texte propre** : reprend le texte EXACT du script (zéro coquille), le découpe en lignes ≤ 52 caractères, et le time scène par scène (réparti proportionnellement au nombre de caractères dans la fenêtre de parole). Écrit `captions.json` = `[{start, end, text}]`. Alternative : `gen_srt_whisper.sh` transcrit l'audio réel avec whisper.cpp (synchro au mot près, mais coquilles phonétiques).

### Remotion — composition `Tutorial`
`src/Root.tsx` enregistre la composition. `src/Video.tsx` mappe chaque scène du manifest vers son composant (`src/scenes/*.tsx`) via une `<Series>`, et intègre l'audio par scène avec `<Audio>`. `src/Background.tsx` = grille + halo de fond. `src/theme.ts` = couleurs.

Types de scènes : `Title`, `Chapter`, `Bullets`, `Architecture`, `FileTree`, `Code`, `Terminal`, `Stat`.

### Remotion — composition `SubsTerminal`
Composition **séparée**, fond transparent, qui lit `captions.json` et affiche les sous-titres dans une fenêtre de terminal : viewport fixe de 4 lignes, défilement, curseur clignotant. Rendue en **ProRes 4444** (canal alpha) → overlay ffmpeg par-dessus n'importe quelle base. La durée de la composition est calculée depuis la dernière caption.

### Assemblage ffmpeg
```
ffmpeg -i master.mp4 -i term.mov -filter_complex '[0][1]overlay=W-w-48:H-h-48' \
       -c:a copy -c:v libx264 -crf 20 master_final.mp4
```

## Pourquoi des couches séparées

- **Multilingue** : la même base visuelle peut recevoir une voix + un terminal-PiP FR ou EN. (Limite actuelle : les durées de scènes dépendent de la longueur de la voix ; pour un vrai swap sans re-rendre la base, il faut passer à des durées fixes.)
- **Sous-titres modifiables** : tout part de la narration du script → corriger une expression met à jour voix ET sous-titres, sans désynchro.

## Pistes (non implémentées)

- Durées de scènes fixes → swap FR/EN sans re-rendre la base.
- Clips verticaux 9:16 taillés dans le master pour les réseaux.
- Orchestration n8n : URL → API LLM (script) → `render_master.sh` → dépôt.
