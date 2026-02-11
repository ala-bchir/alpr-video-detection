# Labélisation des Plaques avec Qwen2.5-VL-7B

Ce guide décrit le processus de labélisation automatique des plaques d'immatriculation avec le modèle **Qwen2.5-VL-7B** (Vision-Language Model).

## Prérequis

- Docker avec support GPU (NVIDIA)
- GPU avec au moins **16 GB VRAM** (testé sur RTX 5090 32GB)
- ~15 GB d'espace disque pour le modèle

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Machine Hôte                         │
│  ┌────────────────────────────────────────────────────┐ │
│  │         Container vLLM (qwen-labeler)              │ │
│  │  ┌──────────────────────────────────────────────┐  │ │
│  │  │  Qwen2.5-VL-7B-Instruct                      │  │ │
│  │  │  API OpenAI compatible sur :8000             │  │ │
│  │  └──────────────────────────────────────────────┘  │ │
│  └───────────────────────────────────────▲────────────┘ │
│                                          │              │
│  ┌────────────────────────┐              │              │
│  │   label_gen_qwen.py    │◄─────────────┘              │
│  │   (script Python)      │   Requêtes HTTP             │
│  └───────────────┬────────┘                             │
│                  │                                      │
│                  ▼                                      │
│  ┌────────────────────────┐                             │
│  │  data/ground_truth.csv │                             │
│  └────────────────────────┘                             │
└─────────────────────────────────────────────────────────┘
```

## Commandes

### Démarrer le serveur Qwen

```bash
make start-qwen
```

Cette commande :
1. Télécharge l'image vLLM (~10 GB, première fois seulement)
2. Télécharge le modèle Qwen (~15 GB, première fois seulement)
3. Lance le serveur API sur le port 8000

> **Note** : Le premier démarrage peut prendre 5-10 minutes. Surveillez avec `docker logs -f qwen-labeler`

### Lancer la labélisation

Une fois le serveur prêt :

```bash
make label-qwen
```

Cette commande :
1. Lit toutes les images dans `data/clean_plates/`
2. Envoie chaque image au modèle Qwen
3. Génère `data/ground_truth.csv`

### Arrêter le serveur

```bash
make stop-qwen
```

## Format de sortie

Le fichier `ground_truth.csv` contient :

| Colonne | Description |
|---------|-------------|
| `filename` | Nom du fichier image |
| `plate_text` | Texte de la plaque (nettoyé, majuscules) |
| `confidence` | Toujours 1.0 (Qwen ne fournit pas de score) |


## Dépannage

### Le serveur ne démarre pas

```bash
# Vérifier les logs
docker logs qwen-labeler

# Vérifier la mémoire GPU
nvidia-smi
```

### Erreur "Connection refused"

Le serveur n'est pas encore prêt. Attendez que les logs affichent :
```
INFO: Application startup complete.
```

### Mémoire GPU insuffisante

Réduisez `--max-model-len` dans le Makefile (2048 → 1024).
