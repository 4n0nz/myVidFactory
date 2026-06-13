# myVidFactory

**Une URL de dépôt GitHub → un tutoriel vidéo complet, narré, animé.** Un agent Claude lit le repo, écrit le script, et un pipeline de rendu code-based produit la vidéo. **CPU + GPU local, sans génération d'images par diffusion** — donc du texte, du code et des schémas nets, en qualité publiable.

Pensé pour produire du contenu en volume (chaîne YouTube, grille TV 24/7).

---

## Le principe

```
URL GitHub
  → [agent Claude Code headless]  lit le repo, suit SPEC.md, écrit script.json
  → voix off          (XTTS local GPU  ou  edge-tts gratuit)
  → cues + captions   (whisper.cpp : timing des actions + sous-titres propres)
  → rendu Remotion    (scènes + avatar host + terminal sous-titres, un seul pass)
  → B-roll glitch      inséré entre les chapitres
  → master.mp4
```

Le seul input humain, c'est l'**URL**. Tout le reste est généré.

---

## Le cerveau : `SPEC.md`

`SPEC.md` est le « system prompt » donné à un agent **Claude Code headless** (`claude -p … --dangerously-skip-permissions`). Il décrit : comment explorer un repo, le format `script.json`, tous les types de scènes et leurs props, les tags `host`, et surtout la **structure narrative en 5 temps** (accroche → contexte → problème → proposition → transformation). L'agent clone le repo, lit le vrai code, et produit le script — il s'adapte au type de repo (app, liste, lib…).

```bash
cd un_dossier_avec_SPEC.md
claude -p "Lis SPEC.md et suis-le. Clone <URL> dans work/repo, analyse, écris script.json." --dangerously-skip-permissions
```

---

## Le pipeline de rendu

| Étape | Script | Rôle |
|---|---|---|
| 1. Voix off | `build_audio_xtts.py` | TTS XTTS par scène → `public/audio/*.mp3` + `render-manifest.json` (durées synchro, tags host) |
| 2. Cues | `build_cues.py` | whisper word-timing → pilote le curseur des scènes `action` |
| 3. Sous-titres | `gen_captions.py` | texte EXACT du script → captions propres pour le terminal |
| 4. Rendu | `npx remotion render … Tutorial` | visuels + avatar host + terminal, **un seul rendu** |
| 5. B-roll | `add_broll.py` + `prep_broll.ps1` | insère des clips d'habillage glitch entre les chapitres |

**Tout-en-un** : `render_master_xtts.sh`.

---

## La voix : XTTS (local GPU)

`build_audio_xtts.py` utilise **XTTS** en local sur le GPU : 58 voix intégrées + clonage de voix depuis un clip de référence (et émotion). Voix par défaut actuelle : `Zacharie Aimilios`. ⚠️ Le modèle XTTS v2 est sous **license non-commerciale** (pour une chaîne monétisée, prévoir une voix à license commerciale).

L'étape voix est **agnostique** : le pipeline lit juste des `.mp3` par scène, donc n'importe quelle source se branche pareil. Le repo contient aussi un chemin **edge-tts** (gratuit, en ligne, `build_audio.py` / `render_master.sh`) comme alternative.

---

## L'avatar host

`src/HostModule.tsx` : un **présentateur masqué** (vidéo `public/avatar.mp4` jouée via `OffthreadVideo`, muette) + le **terminal de sous-titres**, en **un seul bloc**. Il passe en **plein écran (hero)** aux chapitres, puis se réduit en **PiP** dans le coin pendant le contenu technique. Piloté par le champ `host` (`hero` | `pip` | `off`) de chaque scène.

---

## Types de scènes

| Type | Visuel |
|---|---|
| `title` | Carte titre animée (intro/outro) |
| `chapter` | Intercalaire de section (numéro + titre) |
| `bullets` | Liste à puces animée |
| `architecture` | Schéma de boîtes reliées |
| `filetree` | Arborescence du repo |
| `code` | Fenêtre de code (vrai code du repo) + surlignage |
| `terminal` | Terminal qui tape des commandes |
| `install` | Tuto d'installation (commandes + sortie) |
| `stat` | Gros chiffres animés (count-up) |
| `browser` | Recherche web simulée |
| `action` | Démo UI où le curseur clique **synchronisé sur la voix** (whisper) |
| `broll` | Clip d'habillage glitch entre les sections |

---

## Stack

- **Node 20+** + **Remotion 4** (`@remotion/cli`) — rendu (Chrome headless)
- **Python 3.10+** — `coqui-tts` (XTTS, GPU) ou `edge-tts`
- **ffmpeg** — encodage / B-roll
- **whisper.cpp** — timing des cues + sous-titres
- **Claude Code CLI** — l'agent qui écrit les scripts

Testé sur Ubuntu 24.04, 4 cœurs CPU + GTX 1060 6GB (pour XTTS).

---

## Installation

```bash
git clone https://github.com/4n0nz/myVidFactory.git
cd myVidFactory
./install.sh            # deps système + venv + npm + navigateur Remotion
# XTTS (optionnel, GPU) : venv séparé + coqui-tts (voir docs)
```

> `public/` (avatar, B-roll, audio générés) n'est pas versionné — fournis tes propres assets : `public/avatar.mp4` et `public/broll/*.mp4`.

---

## Utilisation

```bash
# 1. L'agent écrit le script depuis un repo
claude -p "Lis SPEC.md et suis-le. Clone <URL>, écris script.json." --dangerously-skip-permissions

# 2. Rendu complet (voix XTTS + avatar host + sous-titres + B-roll)
bash render_master_xtts.sh
# → out/master.mp4
```

Ou écris ton propre `script.json` à la main (voir le format dans `SPEC.md`).

---

## Performance

Master 1080p de ~8 min : voix XTTS sur GPU (~10 min) + rendu Remotion ~50 min sur 4 cœurs. Pensé pour du **batch de nuit**, pas du temps réel.

---

## Licence

Usage privé. Le modèle XTTS v2 est non-commercial — pour une chaîne monétisée, utiliser edge-tts ou une voix à license permissive/commerciale. Les dépôts analysés gardent leur propre licence.
