# CLAUDE.md — sam3_licencePlate_processing

## Vue d'ensemble du projet

Pipeline ALPR (Automatic License Plate Recognition) complet pour la lecture de plaques d'immatriculation françaises à partir de flux vidéo.

**Stack principale :**
- **SAM3** (Segment Anything Model 3) pour la détection des plaques — tourne dans un container Docker sur GPU
- **VLM OCR** (GLM-OCR par défaut, compatible OpenAI API) servi par **vLLM** — tourne dans un second container Docker
- **Python 3.10+** pour l'orchestration côté hôte
- Déploiement serverless possible via **Modal** (`modal_app.py`)

**Machine cible locale :** 64 GB RAM, GPU NVIDIA RTX 5090 (32 GB VRAM)

---

## Architecture du pipeline

```
Vidéo (.avi / .mp4 / .265)
    ↓
video_processor.py   — SAM3 extrait les crops de plaques (Docker, GPU)
    ↓
auto_sort_plates.py  — Filtre les crops (ratio ≤ 7.0, taille ≥ 80×20px, luminosité ≥ 25)
    ↓
homography.py        — Correction de perspective des plaques inclinées
    ↓
OCR via vLLM         — Lecture du texte (GLM-OCR recommandé, ~2 GB VRAM)
    ↓
regex_augmente.py    — Reconnaissance et correction SIV / FNI + score confiance
    ↓
dedup_plates.py      — Déduplication temporelle (fenêtre configurable, similarité substring)
    ↓
results/batch_*.csv  — CSV final : Poste, Date, Timestamp, Categorie, Plaque
```

L'orchestrateur principal est `batch_processor.py` : il gère le cycle de vie des containers Docker (SAM3 et GLM-OCR alternent sur la VRAM), la conversion H.265 → MP4, et l'ETA multi-postes.

---

## Structure des fichiers importants

| Fichier | Rôle |
|---------|------|
| `batch_processor.py` | Orchestrateur principal multi-postes/multi-vidéos |
| `pipeline.py` | Pipeline single-vidéo interactif |
| `video_processor.py` | Extraction SAM3 (tourne **dans** le container Docker) |
| `auto_sort_plates.py` | Filtrage des crops (ratio, taille, luminosité) |
| `homography.py` | Correction de perspective |
| `regex_augmente.py` | Reconnaissance SIV/FNI + corrections OCR (O→0, I→1, S→5…) |
| `dedup_plates.py` | Déduplication par sous-chaîne et caractères communs |
| `mask_picker.py` | Interface web (port 8888) pour dessiner les zones à masquer |
| `extract_vehicles.py` | Détection de catégories véhicules (VL, PL, MOTO, BUS, VUL) |
| `evaluate_models.py` | Benchmark comparatif des modèles OCR (CER, WER, précision) |
| `label_gen_qwen.py` | Génération de ground truth via VLM |
| `modal_app.py` | Déploiement serverless Modal (GPU L40S, vidéos en parallèle) |

---

## Données

```
data/
├── videos/{POSTE}/          ← Vidéos source + mask.json (optionnel)
├── crop/                    ← Crops bruts (sortie SAM3)
├── clean_plates/            ← Après auto-sort
├── corrected_plates/        ← Après homographie
├── rejected_plates/         ← Rejetés (avec sous-dossiers par raison)
├── crops_per_video/         ← Crops isolés par vidéo
└── frames_per_poste/        ← Frames entières avec véhicule + plaque

results/
├── batch_{POSTE}_{DATE}.csv ← Résultat final par poste
└── batch_progress.log       ← Progression en temps réel
```

---

## Commandes fréquentes

```bash
# Traitement batch (mode recommandé)
python3 batch_processor.py
python3 batch_processor.py --postes CA1
python3 batch_processor.py --postes CA1 --max-videos 2 --frame-skip 5
python3 batch_processor.py --dry-run          # Aperçu sans traitement

# Single vidéo
python3 pipeline.py --video data/videos/CA1/video.avi --start-time 12:30

# Interface de masquage (à faire AVANT le premier lancement d'un poste)
python3 mask_picker.py data/videos/CA1        # → http://localhost:8888

# Déploiement Modal (serverless, vidéos en parallèle)
modal run modal_app.py --poste CA1
modal run modal_app.py --poste CA1 --max-videos 1   # Test

# Suivi progression
tail -f results/batch_progress.log
```

---

## Formats de plaques supportés

- **SIV** (France, 2009+) : `AA-123-AA` — lettres sauf I, O, U
- **FNI** (France, avant 2009) : `1234 AB 57`

Le module `regex_augmente.py` applique des corrections OCR courantes : `O→0`, `Q→0`, `I→1`, `S→5`, `B→8`, `G→6`, `T→7`, `A→4` (et inversement selon le contexte lettre/chiffre).

---

## Configuration Docker

- `Dockerfile` — Image SAM3 (base : `nvcr.io/nvidia/pytorch:25.10-py3`)
- `Dockerfile.glm-ocr` — Image GLM-OCR (base : `vllm/vllm-openai:nightly`)
- `Makefile` — Cibles : `build`, `run`, `start-glm-ocr`, `stop-vlm`, `eval-*`, `pipeline`

Le `batch_processor.py` gère l'**alternance VRAM** automatique : SAM3 et GLM-OCR ne tournent pas simultanément.

---

## Modèles OCR disponibles (via Makefile)

| Modèle | VRAM approx. | Commande |
|--------|-------------|----------|
| GLM-OCR (0.9B) — **recommandé** | ~2 GB | `make start-glm-ocr` |
| Qwen2-VL-2B-OCR | ~5 GB | `make start-qwen-2b-ocr` |
| Qwen2.5-VL-7B | ~16 GB | `make start-qwen-7b` |
| RolmOCR (7B) | ~16 GB | `make start-rolm-ocr` |

---

## Conventions et points d'attention

- `video_processor.py` s'exécute **à l'intérieur** du container Docker SAM3 — il lit ses paramètres depuis les variables d'environnement (`INPUT_VIDEO_DIR`, `WEIGHTS_PATH`, `FRAME_SKIP`, `MASK_ZONES`).
- Les zones de masquage sont stockées dans `data/videos/{POSTE}/mask.json` en format polygone `{points: [{x, y}]}` (ou rectangle legacy `{x1, y1, x2, y2}`).
- Le nom des fichiers vidéo encode la date/heure : `2_YYYYMMDD_HHMMSS_0025d5.avi` ou `U20260310-065844_19N100.265`.
- Le CSV de sortie contient les colonnes : `Poste`, `Date`, `Timestamp`, `Categorie`, `Plaque`.
- Le serveur OCR vLLM écoute sur `http://localhost:8000/v1` (compatible API OpenAI).
- Les poids SAM3 (`sam3_weights.pt`, ~3.4 GB) doivent être à la racine du projet.
- Sur Modal, les poids sont dans un Volume persistant `alpr-weights` et les vidéos/résultats transitent par **Nextcloud** (`cloud.itecmobility.com`).
