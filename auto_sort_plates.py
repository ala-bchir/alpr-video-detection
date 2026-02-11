#!/usr/bin/env python3
"""
Script de tri AUTOMATIQUE des plaques d'immatriculation.
Filtres:
- Ratio haut (trop allongé) - on garde les ratios bas pour les motos
- Images noires/trop sombres
- Dimensions trop bizarres (trop petit ou très déformé)
"""

import os
import cv2
import shutil
import numpy as np
from pathlib import Path

# ============== CONFIGURATION ==============
CROP_DIR = "data/crop"
CLEAN_DIR = "data/clean_plates"
REJECTED_BASE_DIR = "data/rejected_plates"

# Critères de filtrage
MAX_RATIO = 7.0             # Ratio max (largeur/hauteur) - plaque trop allongée
MIN_WIDTH = 80              # Largeur minimale en pixels (élimine les demi-plaques)
MIN_HEIGHT = 20             # Hauteur minimale en pixels
MIN_AREA = 2000             # Surface minimale en pixels² (largeur x hauteur)
BRIGHTNESS_THRESHOLD = 25   # Luminosité moyenne minimale (0-255) - en dessous = image noire
# ==========================================


def analyze_plate(img_path):
    """Analyse une image de plaque et retourne les résultats."""
    img = cv2.imread(str(img_path))
    if img is None:
        return {"valid": False, "reason": "lecture_impossible"}
    
    h, w = img.shape[:2]
    ratio = w / h if h > 0 else 0
    
    # Critère 1 : Image trop petite (dimensions bizarres)
    if w < MIN_WIDTH or h < MIN_HEIGHT:
        return {"valid": False, "reason": "trop_petit", "ratio": ratio, "w": w, "h": h}
    
    # Critère 2 : Surface trop petite (demi-plaque)
    area = w * h
    if area < MIN_AREA:
        return {"valid": False, "reason": "trop_petit", "ratio": ratio, "w": w, "h": h}
    
    # Critère 2 : Ratio trop haut (trop allongé)
    if ratio > MAX_RATIO:
        return {"valid": False, "reason": "ratio_haut", "ratio": ratio}
    
    # Critère 3 : Image noire / trop sombre
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    if brightness < BRIGHTNESS_THRESHOLD:
        return {"valid": False, "reason": "image_noire", "ratio": ratio, "brightness": brightness}
    
    return {"valid": True, "ratio": ratio, "brightness": brightness}


def main():
    # Nettoyer les dossiers de sortie au début
    for dir_path in [CLEAN_DIR, REJECTED_BASE_DIR]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
    
    os.makedirs(CLEAN_DIR, exist_ok=True)
    
    # Sous-dossiers pour les rejets
    rejection_folders = {
        "ratio_haut": "ratio_haut",
        "trop_petit": "trop_petit",
        "image_noire": "image_noire",
        "lecture_impossible": "erreur_lecture"
    }
    
    for folder in rejection_folders.values():
        os.makedirs(os.path.join(REJECTED_BASE_DIR, folder), exist_ok=True)
    
    # Récupérer les images
    crop_path = Path(CROP_DIR)
    images = sorted([f for f in crop_path.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    
    if not images:
        print("❌ Aucune image trouvée dans", CROP_DIR)
        return
    
    print(f"🔍 Analyse de {len(images)} images...")
    print(f"📋 Filtres:")
    print(f"   - Ratio max: {MAX_RATIO}")
    print(f"   - Dimensions min: {MIN_WIDTH}x{MIN_HEIGHT}px")
    print(f"   - Luminosité min: {BRIGHTNESS_THRESHOLD}/255")
    print(f"   - Ratio bas: DÉSACTIVÉ (motos autorisées)")
    print("-" * 50)
    
    stats = {"accepted": 0, "rejected": 0, "ratio_haut": 0, "trop_petit": 0, "image_noire": 0, "lecture_impossible": 0}
    
    for img_path in images:
        result = analyze_plate(img_path)
        
        if result["valid"]:
            shutil.copy2(img_path, Path(CLEAN_DIR) / img_path.name)
            stats["accepted"] += 1
        else:
            reason = result["reason"]
            dest_folder = rejection_folders.get(reason, "erreur_lecture")
            shutil.copy2(img_path, Path(REJECTED_BASE_DIR) / dest_folder / img_path.name)
            stats["rejected"] += 1
            stats[reason] = stats.get(reason, 0) + 1
    
    # Résumé
    print("\n" + "=" * 50)
    print("📊 RÉSUMÉ DU TRI AUTOMATIQUE")
    print("=" * 50)
    print(f"✅ Acceptées:  {stats['accepted']} → {CLEAN_DIR}/")
    print(f"❌ Rejetées:   {stats['rejected']} → {REJECTED_BASE_DIR}/")
    print(f"\n📁 Détail rejets:")
    print(f"   - ratio_haut (> {MAX_RATIO}): {stats['ratio_haut']}")
    print(f"   - trop_petit (<{MIN_WIDTH}x{MIN_HEIGHT}): {stats['trop_petit']}")
    print(f"   - image_noire (luminosité <{BRIGHTNESS_THRESHOLD}): {stats['image_noire']}")
    print(f"   - erreur_lecture: {stats['lecture_impossible']}")

if __name__ == "__main__":
    main()
