# 🚗 ALPR Video Detection — Lecture Automatique de Plaques d'Immatriculation

Pipeline complet de détection et lecture de plaques d'immatriculation à partir de flux vidéo, utilisant **SAM3** (Segment Anything Model 3) pour la détection et des **modèles VLM** (Vision Language Models) pour l'OCR.

> **Machine cible** : 64 GB RAM, GPU NVIDIA RTX 5090 (32 GB VRAM)

---

## 📋 Table des matières

- [Architecture du pipeline](#-architecture-du-pipeline)
- [Structure du projet](#-structure-du-projet)
- [Prérequis](#-prérequis)
- [Installation](#-installation)
- [Utilisation rapide](#-utilisation-rapide)
- [Scripts détaillés](#-scripts-détaillés)
- [Modèles OCR disponibles](#-modèles-ocr-disponibles)
- [Données et résultats](#-données-et-résultats)

---

## ⚙️ Architecture du pipeline

Chaque vidéo passe par les étapes suivantes :

```
Vidéo (.avi / .mp4 / .265)
    │
    ▼
[1] SAM3 (video_processor.py)       → Détection des plaques dans les frames (GPU)
    │
    ▼
[2] Auto-sort (auto_sort_plates.py) → Filtrage des faux positifs (ratio, luminosité, taille)
    │
    ▼
[3] Homographie (homography.py)     → Correction de perspective des plaques inclinées
    │
    ▼
[4] OCR VLM (GLM-OCR / Qwen / …)   → Lecture du texte via modèle vision-langage (GPU)
    │
    ▼
[5] Regex (regex_augmente.py)       → Correction et validation du format (SIV / FNI)
    │
    ▼
[6] Dédup (dedup_plates.py)         → Déduplication temporelle des lectures multiples
    │
    ▼
   CSV final (results/)
```

---

## 📁 Structure du projet

### Scripts principaux

| Fichier | Description |
|---------|-------------|
| `batch_processor.py` | **Orchestrateur principal** — traitement multi-postes / multi-vidéos avec progression |
| `pipeline.py` | Pipeline single-vidéo (mode interactif, avec heure de départ) |
| `video_processor.py` | Extraction des plaques depuis les frames vidéo via SAM3 (Docker) |

### Scripts du pipeline

| Fichier | Description |
|---------|-------------|
| `auto_sort_plates.py` | Tri automatique : filtre les crops par ratio, luminosité, surface |
| `homography.py` | Correction de perspective (redresse les plaques inclinées) |
| `regex_augmente.py` | Reconnaissance et correction de format (SIV `AA-123-AA`, FNI `1234 AB 57`) |
| `dedup_plates.py` | Déduplication intelligente par fenêtre temporelle et similarité |

### Outils

| Fichier | Description |
|---------|-------------|
| `mask_picker.py` | Interface web pour définir les zones à masquer (faux positifs fixes) |
| `extract_vehicles.py` | Extraction de frames contenant des véhicules (détection par catégorie) |
| `label_gen_qwen.py` | Labélisation automatique des plaques via VLM (génération de ground truth) |
| `evaluate_models.py` | Benchmark comparatif des modèles OCR (CER, WER, précision) |
| `generate_ground_truth.py` | Génération du ground truth depuis les noms de fichiers |
| `rename_images.py` | Renommage des images avec le texte de plaque détecté |

### Configuration Docker

| Fichier | Description |
|---------|-------------|
| `Dockerfile` | Image SAM3 (basée sur `nvcr.io/nvidia/pytorch:25.10-py3`) |
| `Dockerfile.glm-ocr` | Image GLM-OCR (basée sur `vllm/vllm-openai:nightly`) |
| `Makefile` | Commandes Make pour build, run, évaluation et pipeline |

### Répertoires de données

```
data/
├── videos/              ← Vidéos source, organisées par poste (CA1/, CA3/, …)
│   └── {POSTE}/
│       ├── mask.json    ← Zones de masquage (optionnel)
│       └── *.avi / *.265
├── crop/                ← Crops bruts des plaques (sortie SAM3)
├── clean_plates/        ← Plaques filtrées (après auto-sort)
├── corrected_plates/    ← Plaques redressées (après homographie)
├── rejected_plates/     ← Plaques rejetées (ratio, taille, luminosité)
├── crops_per_video/     ← Crops isolés par vidéo
├── frames_per_poste/    ← Frames entières par poste (avec plaque + véhicule)
├── vehicle_frames/      ← Frames contenant des véhicules détectés
├── images/              ← Photos entières (entrée SAM3)
├── resultat_ocr/        ← Résultats OCR bruts
└── train_images/        ← Images d'entraînement

dataset/
├── train/               ← Dataset d'entraînement
├── validation/          ← Dataset de validation
└── test/                ← Dataset de test (avec ground truth)

results/
├── batch_{POSTE}_{DATE}.csv   ← CSV final par poste
├── batch_progress.log         ← Log de progression en temps réel
└── {model_name}/              ← Résultats par modèle OCR testé
```

---

## 🔧 Prérequis

- **Docker** avec support GPU (NVIDIA Container Toolkit)
- **Python 3.10+**
- **GPU NVIDIA** avec CUDA (≥ 16 GB VRAM recommandé)
- `nvidia-smi` fonctionnel

---

## � Bibliothèques et dépendances

### Côté hôte (scripts Python)

| Bibliothèque | Usage |
|--------------|-------|
| `opencv-python` | Lecture vidéo, traitement d'images, correction de perspective |
| `numpy` | Manipulation de tableaux, calculs image |
| `Pillow` | Conversion d'images pour SAM3 |
| `openai` | Client API pour interroger les modèles VLM via vLLM |
| `requests` | Requêtes HTTP (health checks serveur OCR) |
| `tqdm` | Barres de progression |
| `torch` | Inférence SAM3 (GPU) |

### Conteneur Docker SAM3 (`requirements.txt`)

| Bibliothèque | Usage |
|--------------|-------|
| `numpy`, `Pillow`, `opencv-python-headless` | Traitement d'images |
| `torch`, `torchvision` | Inférence deep learning (pré-installés dans l'image NVIDIA) |
| `sam3` | Segment Anything Model 3 (cloné depuis GitHub) |
| `huggingface_hub` | Gestion des modèles HuggingFace |
| `einops`, `timm` | Opérations tensorielles et architectures vision |
| `decord`, `moviepy`, `av` | Décodage vidéo |
| `hydra-core`, `omegaconf` | Configuration SAM3 |
| `matplotlib`, `pycocotools` | Visualisation et évaluation COCO |
| `psutil` | Monitoring ressources système |
| `tqdm` | Barres de progression |
| `redis`, `rq` | File de tâches (optionnel) |
| `boto3`, `urllib3` | Stockage cloud (optionnel) |
| `colorama` | Couleurs terminal |

### Conteneur Docker OCR (vLLM)

| Bibliothèque | Usage |
|--------------|-------|
| `vllm` | Serveur d'inférence LLM/VLM (API compatible OpenAI) |
| `transformers` | Nécessaire pour certains modèles (GLM-OCR) |

### Bibliothèques standard Python (sans installation)

`os`, `sys`, `re`, `csv`, `json`, `base64`, `argparse`, `subprocess`, `shutil`, `time`, `http.server`, `collections`, `dataclasses`, `typing`, `pathlib`, `datetime`

---

## �📦 Installation

```bash
# 1. Cloner le projet
git clone <repo-url>
cd sam3_licencePlate_processing

# 2. Installer les dépendances Python (pour le batch processor)
pip install requests opencv-python openai

# 3. Construire les images Docker
make build          # Image SAM3
make build-glm-ocr  # Image GLM-OCR

# 4. Placer les poids SAM3
# Le fichier sam3_weights.pt (~3.4 GB) doit être à la racine du projet
```

---

## 🚀 Utilisation rapide

### Mode batch (recommandé)

```bash
# Voir les postes et vidéos détectés
python3 batch_processor.py --dry-run

# Traiter tous les postes
python3 batch_processor.py

# Traiter un seul poste
python3 batch_processor.py --postes CA1

# Limiter le nombre de vidéos (pour tester)
python3 batch_processor.py --postes CA1 --max-videos 2 --frame-skip 5
```

### Mode single vidéo

```bash
# Via le Makefile (la vidéo doit être dans data/videos/)
make pipeline START_TIME=12:30 TIME_WINDOW=3 FRAME_SKIP=1
```

### Masquage de zones

```bash
# Ouvrir l'interface web pour dessiner les zones à masquer
python3 mask_picker.py data/videos/CA1
# → http://localhost:8888
```

> Voir [README_BATCH.md](README_BATCH.md) pour la documentation complète du batch processor.

---

## 📊 Scripts détaillés

### `batch_processor.py` — Orchestrateur

Traitement automatique multi-postes. Gère le cycle de vie des containers Docker (SAM3 et GLM-OCR), l'alternance GPU, la conversion H.265 → MP4, et le suivi de progression avec ETA.

| Option | Défaut | Description |
|--------|--------|-------------|
| `--dry-run` | — | Liste les postes/vidéos sans traitement |
| `--postes CA1 CA3` | tous | Postes à traiter |
| `--max-videos N` | 0 (tous) | Nombre max de vidéos par poste |
| `--frame-skip N` | 3 | Traite 1 frame sur N |
| `--time-window N` | 3 | Fenêtre de déduplication en secondes |

### `regex_augmente.py` — Correction de format

Reconnaît et corrige les plaques aux formats :
- **SIV** (France, depuis 2009) : `AA-123-AA`
- **FNI** (France, avant 2009) : `1234 AB 57`

Applique des corrections OCR courantes (O→Q, I→J, 0→D, etc.) et calcule un score de confiance.

### `auto_sort_plates.py` — Filtrage

Filtre les crops selon :
- **Ratio** largeur/hauteur (max 7.0)
- **Dimensions minimales** (80×20 px, surface ≥ 2000 px²)
- **Luminosité** (seuil = 25/255, rejette les images noires)

### `extract_vehicles.py` — Détection de véhicules

Utilise SAM3 en mode open-vocabulary pour détecter les types de véhicules (VL, PL, MOTO, BUS, VAN) et sauvegarder les frames correspondantes.

---

## 🤖 Modèles OCR disponibles

Tous les modèles sont servis via **vLLM** (API compatible OpenAI) dans des containers Docker.

| Modèle | VRAM | Commande Make |
|--------|------|---------------|
| **GLM-OCR** (0.9B) — Recommandé | ~2 GB | `make start-glm-ocr` |
| Qwen2-VL-2B-OCR | ~5 GB | `make start-qwen-2b-ocr` |
| Qwen2.5-VL-7B | ~16 GB | `make start-qwen-7b` |
| RolmOCR (7B) | ~16 GB | `make start-rolm-ocr` |
| Chandra | ~16 GB | `make start-chandra` |
| LightOnOCR-2-1B | ~3 GB | `make start-lighton-ocr` |
| Dots.OCR | variable | `make start-dots-ocr` |
| PaddleOCR-VL | variable | `make start-paddle-ocr` |
| DeepSeek-OCR-2 (3B) | variable | `make start-deepseek-ocr2` |

```bash
# Évaluer un modèle sur le dataset de test
make eval-glm-ocr      # ou eval-7b, eval-rolm-ocr, etc.

# Arrêter le serveur VLM actif
make stop-vlm
```

> Voir [LABELING_QWEN.md](LABELING_QWEN.md) pour la documentation de labélisation avec Qwen.

---

## 📄 Sortie CSV

Un CSV est généré par poste dans `results/` :

| Colonne | Description | Exemple |
|---------|-------------|---------|
| **Poste** | Nom de la caméra | CA1 |
| **Date** | Date de passage | 2026-02-28 |
| **Timestamp** | Heure calculée | 09:42:15 |
| **Categorie** | Type de véhicule | VL |
| **Plaque** | Numéro détecté | GK-081-RH |

---

## 💡 Conseils

- **Premier test** : `python3 batch_processor.py --postes CA1 --max-videos 1`
- **Frame skip** : `3` est un bon compromis vitesse/précision
- **Masque** : toujours le définir **avant** le premier lancement d'un poste
- **Suivi temps réel** : `tail -f results/batch_progress.log`
- **VRAM** : SAM3 et GLM-OCR partagent la VRAM, le batch processor gère l'alternance automatiquement
