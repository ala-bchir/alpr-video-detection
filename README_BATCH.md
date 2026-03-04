# 🎬 Batch Processor — Traitement Multi-Postes

Script de traitement automatique de vidéos pour la détection de plaques d'immatriculation, organisé par **poste** (caméra).

---

## 📁 Structure attendue

```
data/videos/
├── CA1/                          ← Poste 1
│   ├── mask.json                 ← (optionnel) zones à masquer
│   ├── 2_20260228_070735_0025d5.avi
│   ├── 2_20260228_073800_0025d5.avi
│   └── ...
├── CA3/                          ← Poste 3
│   ├── mask.json
│   └── ...
└── CA4/                          ← Poste 4
    └── ...
```

**Format du nom de vidéo** : `{id}_{AAAAMMJJ}_{HHMMSS}_{hex}.avi`

- `AAAAMMJJ` → date (ex: 20260228 = 28 février 2026)
- `HHMMSS` → heure de début de la vidéo (ex: 070735 = 07h07m35s)

---

## 🚀 Utilisation

### Pré-requis

- Docker installé avec GPU (NVIDIA)
- Image SAM3 construite : `make build`
- Image GLM-OCR construite : `make build-glm-ocr`
- Dépendances Python : `pip install requests opencv-python openai`

### Commandes

```bash
# 1. Voir les postes et vidéos détectés (sans rien traiter)
python3 batch_processor.py --dry-run

# 2. Lancer le traitement complet (tous les postes, toutes les vidéos)
python3 batch_processor.py

# 3. Traiter un seul poste
python3 batch_processor.py --postes CA1

# 4. Traiter plusieurs postes spécifiques
python3 batch_processor.py --postes CA1 CA3

# 5. Limiter le nombre de vidéos par poste (pour tester)
python3 batch_processor.py --max-videos 2

# 6. Combiner les options
python3 batch_processor.py --postes CA1 --max-videos 3 --frame-skip 5
```

### Options

| Option | Défaut | Description |
|--------|--------|-------------|
| `--dry-run` | — | Liste les postes/vidéos sans traitement |
| `--postes CA1 CA3` | tous | Postes à traiter |
| `--max-videos N` | 0 (tous) | Nombre max de vidéos par poste |
| `--frame-skip N` | 3 | Traite 1 frame sur N (↑ = plus rapide, ↓ = plus précis) |
| `--time-window N` | 3 | Fenêtre de déduplication en secondes |

---

## 🎭 Masquage de zones (faux positifs)

Si une caméra capte des objets fixes (plastiques, poteaux réfléchissants) que SAM3 confond avec des plaques, vous pouvez les masquer :

### Étape 1 : Définir le masque (interface web)

```bash
python3 mask_picker.py data/videos/CA1
# → Ouvrir http://localhost:8888 dans le navigateur
```

| Action | Effet |
|--------|-------|
| Clic + glisser | Dessiner un rectangle rouge (zone à masquer) |
| ↩ Annuler | Supprimer le dernier rectangle |
| 🔄 Reset | Tout effacer |
| 💾 Sauvegarder | Écrit `mask.json` dans le dossier du poste |

Options : `--port 9090` pour changer le port, `--video nom.avi` pour choisir la vidéo de référence.

### Étape 2 : Relancer le traitement

Le masque est chargé **automatiquement** par `batch_processor.py` — rien à configurer :

```bash
python3 batch_processor.py --postes CA1
# → "🎭 Mask chargé: 2 zone(s) à masquer"
```

---

## 📊 Suivi de progression

Le script affiche un suivi détaillé pendant l'exécution :

```
[████████░░░░░░░░░░░░] 40% — 12/30 vidéos — ETA: 1h23m
🎬 Vidéo 13/30 du poste CA1: 2_20260228_110800_0025d5.avi
   📦 Taille: 1021 MB | Écoulé: 45m12s
   🔍 [1/6] SAM3 (frame_skip=3)...
   ⏱️  SAM3 terminé en 8m32s
   🤖 [4/6] OCR GLM...
   ⏱️  OCR terminé en 1m15s
   📊 [6/6] Résultat: 12 plaques en 10m03s
```

### Suivi depuis un autre terminal SSH

```bash
tail -f results/batch_progress.log
```

---

## 📄 Sortie CSV

Un fichier CSV est généré **par poste** dans le dossier `results/` :

```
results/
├── batch_CA1_20260228_143500.csv
├── batch_CA3_20260228_150200.csv
├── batch_CA4_20260228_152800.csv
└── batch_progress.log
```

### Colonnes

| Colonne | Description | Exemple |
|---------|-------------|---------|
| **Poste** | Nom du dossier caméra | CA1 |
| **Date** | Date de passage (depuis le nom vidéo) | 2026-02-28 |
| **Timestamp** | Heure de passage calculée | 09:42:15 |
| **Categorie** | Type de véhicule | VL |
| **Plaque** | Numéro de plaque détecté | GK-081-RH |

### Exemple de CSV

```csv
Poste,Date,Timestamp,Categorie,Plaque
CA1,2026-02-28,07:08:12,VL,GM-870-VD
CA1,2026-02-28,07:08:45,VL,DD-526-RG
CA1,2026-02-28,09:42:15,VL,GK-081-RH
CA1,2026-02-28,14:23:07,VL,HC-620-BR
```

---

## ⚙️ Pipeline de traitement

Pour chaque vidéo, le script exécute :

```
[1/6] SAM3        → Détection des plaques dans les frames vidéo (GPU)
[2/6] Auto-sort   → Filtrage des faux positifs (ratio, luminosité, taille)
[3/6] Homographie → Correction de perspective des plaques inclinées
[4/6] GLM-OCR     → Lecture du texte des plaques (GPU)
[5/6] Filtres     → Regex + dédup + anti-hallucination
[6/6] Export      → Ajout au CSV du poste
```

> **Note** : SAM3 et GLM-OCR partagent la VRAM. Le script gère automatiquement l'alternance.

---

## 💡 Conseils

- **Premier test** : `python3 batch_processor.py --postes CA1 --max-videos 1`
- **Frame skip** : `3` est un bon compromis. Monter à `5-10` pour les vidéos longues
- **Masque** : toujours définir le masque **avant** le premier lancement d'un poste
- **Vidéos de nuit** : les petites vidéos (<10 MB) sont traitées rapidement, pas besoin de les exclure
- **Suivi** : ouvrir un 2e terminal SSH avec `tail -f results/batch_progress.log`
