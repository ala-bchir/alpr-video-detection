#!/usr/bin/env python3
"""
Génère le ground truth à partir des noms de fichiers des images de test.
Les images sont nommées avec le texte de la plaque (ex: AB123CD.jpg, AB123CD_1.jpg).

Usage:
    python3 generate_ground_truth.py
"""
import os
import re
import csv
from datetime import datetime

# Configuration
TEST_DIR = "./dataset/test"
OUTPUT_DIR = "./dataset"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "ground_truth.csv")


def extract_plate_from_filename(filename):
    """
    Extrait le texte de la plaque à partir du nom de fichier.
    Gère les doublons avec suffixe _1, _2, etc.
    
    Exemples:
        AB123CD.jpg -> AB123CD
        AB123CD_1.jpg -> AB123CD
        1474DPG_2.jpg -> 1474DPG
    """
    # Retirer l'extension
    name = os.path.splitext(filename)[0]
    
    # Retirer le suffixe de doublon (_1, _2, _3, etc.)
    name = re.sub(r'_\d+$', '', name)
    
    # Nettoyer (garder uniquement lettres et chiffres)
    plate = re.sub(r'[^A-Z0-9]', '', name.upper())
    
    return plate


def main():
    print("=" * 60)
    print("GÉNÉRATION DU GROUND TRUTH")
    print("=" * 60)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Vérifier que le dossier existe
    if not os.path.exists(TEST_DIR):
        print(f"❌ Erreur: Le dossier {TEST_DIR} n'existe pas")
        return
    
    # Créer le dossier de sortie
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Lister les images
    files = sorted([f for f in os.listdir(TEST_DIR) 
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    total_files = len(files)
    
    print(f"📁 Dossier source: {TEST_DIR}")
    print(f"📊 {total_files} images trouvées")
    print()
    
    # Statistiques
    unique_plates = set()
    
    # Générer le CSV
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "plate_text"])
        
        for filename in files:
            plate = extract_plate_from_filename(filename)
            unique_plates.add(plate)
            writer.writerow([filename, plate])
    
    print("=" * 60)
    print("RÉSUMÉ")
    print("=" * 60)
    print(f"""
📁 Images traitées: {total_files}
🔤 Plaques uniques: {len(unique_plates)}
🔄 Doublons: {total_files - len(unique_plates)}

📄 Fichier généré: {OUTPUT_CSV}
""")
    
    # Afficher quelques exemples
    print("Exemples:")
    print("-" * 40)
    with open(OUTPUT_CSV, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for i, row in enumerate(reader):
            if i >= 5:
                break
            print(f"  {row[0]} -> {row[1]}")
    print()


if __name__ == "__main__":
    main()
