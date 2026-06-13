# myVidFactory

**GitHub repo → tutoriel vidéo automatisé.** Donne une URL de dépôt, récupère une vidéo éducative 1080p avec voix off, sous-titres et motion graphics — le tout généré en code, **sur CPU, sans GPU**.

Pensé pour produire du contenu en volume (grille TV 24/7, chaîne YouTube). Pas de diffusion d'images IA : pour du contenu factuel (code, schémas, texte), les motion graphics programmés donnent une qualité publiable et reproductible, là où la génération par diffusion échoue (texte illisible, détails déformés).

---

## Comment ça marche

```
URL GitHub
  → analyse du repo (README, arborescence, fichiers clés)
  → script.json  (scènes : narration + type de visuel + contenu)
  → voix off     (edge-tts, gratuit — FR/EN)
  → visuels      (Remotion : titres, listes, schémas, code, terminal, stats…)
  → sous-titres  (fenêtre terminal en PiP, texte propre synchronisé)
  → assemblage   (ffmpeg) → master.mp4
```

### Architecture en couches

Les visuels, la voix et les sous-titres sont des **couches séparées** assemblées à la fin :

| Couche | Outil | Sortie |
|---|---|---|
| Visuels (muets) | Remotion (React → vidéo) | `master.mp4` |
| Voix off | edge-tts | audio par scène, intégré au rendu |
| Sous-titres PiP | Remotion (comp alpha séparée) | `term.mov` (fond transparent) |
| Assemblage | ffmpeg overlay | `master_final.mp4` |

Les sous-titres s'affichent dans une **fenêtre de terminal** (PiP coin bas-droite) : le texte se tape et défile, look « live transcript ».

---

## Stack

- **Node 20+** + **Remotion 4** (`@remotion/cli`) — rendu des visuels (Chrome headless)
- **Python 3.10+** + **edge-tts** — voix off (gratuit, voix neurales Microsoft)
- **ffmpeg** — overlay du PiP, encodage
- **whisper.cpp** *(optionnel)* — sous-titres par transcription audio (sinon : texte propre du script)

Tout tourne sur Linux x86 (testé sur Ubuntu 24.04, 4 cœurs / 8 Go RAM). Pas de GPU.

---

## Installation

```bash
git clone https://github.com/4n0nz/myVidFactory.git
cd myVidFactory
./install.sh          # deps système + venv edge-tts + npm + navigateur Remotion
```

`install.sh` installe les bibliothèques système requises par Chrome headless, crée un venv Python avec edge-tts, fait `npm install` et télécharge le navigateur de Remotion. whisper.cpp est optionnel (flag `--whisper`).

Voir [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) pour le détail des composants.

---

## Utilisation

### 1. Écrire un script

Un `script.json` décrit la vidéo scène par scène (voir `script_qc.json` comme exemple complet) :

```json
{
  "meta": { "voice": "fr-CA-ThierryNeural", "rate": "+15%", "pitch": "-16Hz",
            "fps": 30, "width": 1920, "height": 1080 },
  "scenes": [
    { "id": "intro", "type": "title",
      "narration": "Texte lu par la voix off…",
      "props": { "title": "MON REPO", "subtitle": "…", "badge": "★ 9.2k" } },
    { "id": "ch1", "type": "chapter", "narration": "Chapitre un.",
      "props": { "num": "01", "title": "C'est quoi" } }
  ]
}
```

### 2. Générer la vidéo

```bash
# tout-en-un (voix + visuels + terminal-PiP + assemblage)
bash render_master.sh

# ou étape par étape :
. .venv/bin/activate
python build_audio.py script_qc.json            # voix off + manifest synchronisé
python gen_captions.py script_qc.json           # sous-titres texte propre
npx remotion render src/index.ts Tutorial out/master.mp4 --concurrency=3
npx remotion render src/index.ts SubsTerminal out/term.mov --codec=prores --prores-profile=4444
ffmpeg -i out/master.mp4 -i out/term.mov -filter_complex '[0][1]overlay=W-w-48:H-h-48' \
       -c:a copy -c:v libx264 -crf 20 out/master_final.mp4
```

Sortie : `out/master_final.mp4`.

### Types de scènes

| Type | Visuel |
|---|---|
| `title` | Carte titre animée (titre, sous-titre, badge) |
| `chapter` | Intercalaire de section (numéro + titre) |
| `bullets` | Liste à puces animée |
| `architecture` | Schéma de boîtes reliées par des flèches |
| `filetree` | Arborescence de fichiers qui se déploie |
| `code` | Fenêtre de code avec coloration + lignes surlignées |
| `terminal` | Terminal simulé qui tape des commandes |
| `stat` | Gros chiffres animés (count-up) |

---

## Personnalisation

- **Voix / ton** : `meta.voice` (`edge-tts --list-voices`), `meta.rate` (vitesse, ex `+15%`), `meta.pitch` (hauteur, ex `-16Hz`). Override voix : `python build_audio.py script.json fr-CA-SylvieNeural`.
- **Couleurs** : `src/theme.ts` (`accent`, fond, etc.).
- **Taille/position du terminal-PiP** : dimensions de la composition `SubsTerminal` dans `src/Root.tsx`, position dans le filtre `overlay` ffmpeg.

---

## Performance

Master 1080p de ~6 min : ~35 min de rendu (base) + ~30 min (terminal-PiP alpha) sur 4 cœurs CPU. Pensé pour du batch de nuit, pas du temps réel.

---

## Licence

Usage privé. Les dépôts analysés gardent leur propre licence.

---

## v6 — host avatar, B-roll, voix XTTS, cerveau agent

- **Host avatar** () : un présentateur masqué (vidéo  via OffthreadVideo) + le terminal de sous-titres, en un bloc qui passe de **plein écran (hero)** aux **chapitres** à **PiP** pendant le contenu. Piloté par le champ `host` (hero/pip/off) de chaque scène. Plus d'overlay alpha séparé.
- **B-roll** (`src/scenes/Broll.tsx` + `add_broll.py`) : clips d'habillage insérés entre les sections avec un effet **glitch / coupure de signal**. Clips préparés via `prep_broll.ps1`.
- **Scènes UI** : `BrowserSearch` (recherche simulée), `Action` (curseur synchronisé sur la voix via whisper + `gen_cues.py`/`build_cues.py`), `Install` (terminal d'install), `MatrixRain`, `Stat`, `Chapter`.
- **Voix** : `build_audio_xtts.py` (XTTS local sur GPU, voix clonables) en alternative à edge-tts. Driver : `render_master_xtts.sh`.
- **Cerveau agent** : `SPEC.md` = les consignes complètes données à un agent Claude Code headless pour transformer un repo en `script.json` (structure narrative en 5 temps, types de scènes, tags host). C'est ce qui rend la génération 100% autonome depuis une URL.

> Note : `public/` (avatar, B-roll, audio générés) n'est pas versionné — fournis tes propres assets.
