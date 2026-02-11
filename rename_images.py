#!/usr/bin/env python3
"""
Script temporaire pour renommer les images avec le texte de la plaque.
Supprime les doublons (garde une seule image par plaque).
"""
import os
import csv

# Configuration
CSV_FILE = "./results/qwen2_vl_2b_ocr/ground_truth.csv"
IMAGE_DIR = "./data/clean_plates"

def main():
    print("=" * 60)
    print("RENOMMAGE DES IMAGES (avec suppression des doublons)")
    print("=" * 60)
    
    if not os.path.exists(CSV_FILE):
        print(f"❌ Fichier {CSV_FILE} non trouvé")
        return
    
    renamed = 0
    deleted = 0
    errors = 0
    seen_plates = set()
    
    with open(CSV_FILE, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            old_name = row['filename']
            plate_text = row['predicted_text']
            
            old_path = os.path.join(IMAGE_DIR, old_name)
            
            if not os.path.exists(old_path):
                errors += 1
                continue
            
            # Si la plaque a déjà été vue, supprimer le doublon
            if plate_text in seen_plates:
                os.remove(old_path)
                deleted += 1
                continue
            
            seen_plates.add(plate_text)
            
            # Renommer avec le texte de la plaque
            ext = os.path.splitext(old_name)[1]
            new_name = f"{plate_text}{ext}"
            new_path = os.path.join(IMAGE_DIR, new_name)
            
            os.rename(old_path, new_path)
            renamed += 1
    
    print(f"\n✅ {renamed} images renommées (plaques uniques)")
    print(f"🗑️  {deleted} doublons supprimés")
    print(f"❌ {errors} erreurs")

if __name__ == "__main__":
    main()
