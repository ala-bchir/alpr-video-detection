# Comparatif des Modèles OCR pour Plaques d'Immatriculation

**Dataset :** 2000 images de test  
**Date :** 2026-02-06

## Résultats

| Modèle | Accuracy | CER | WER | Temps/img |
|--------|----------|-----|-----|-----------|
| **GLM-OCR (0.9B)** ⭐ | **89.10%** | **2.14%** | 10.90% | **41ms** |
| Chandra | 87.05% | 2.31% | 12.95% | 117ms |
| Qwen2.5-VL-7B | 87.10% | 2.41% | 12.90% | 95ms |
| RolmOCR (7B) | 86.10% | 2.58% | 13.90% | 98ms |
| Qwen2-VL-2B-OCR | 84.15% | 2.91% | 15.85% | 46ms |
| PaddleOCR-VL | 69.90% | 8.53% | 30.10% | 43ms |
| LightOnOCR-2-1B | 62.00% | - | 38.00% | 256ms |
| DeepSeek-OCR-2 | 50.20% | 24.64% | 49.80% | 52ms |
| Dots.OCR | 49.25% | 72.47% | 50.75% | 61ms |

## Métriques

- **Accuracy** : Pourcentage de plaques correctement lues (exact match)
- **CER** : Character Error Rate - Taux d'erreur par caractère
- **WER** : Word Error Rate - Taux d'erreur par mot (plaque)
- **Temps/img** : Temps moyen de traitement par image

## Conclusion

🏆 **GLM-OCR** est le meilleur modèle pour cette tâche :
- Meilleure accuracy (89.10%)
- Plus rapide (41ms/image)
- Le plus léger (0.9B paramètres)
- Excellent ratio performance/ressources
