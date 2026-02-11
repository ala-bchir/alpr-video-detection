#!/usr/bin/env python3
"""
Script de test OCR pour plaques d'immatriculation.
Compare les résultats des modèles VLM avec le ground truth.

Usage:
    python3 label_gen_qwen.py                        # Modèle par défaut (2B-OCR)
    python3 label_gen_qwen.py --model qwen25-vl-7b   # Modèle 7B
    python3 label_gen_qwen.py --model qwen2-vl-2b-ocr
"""
import os
import csv
import base64
import re
import time
import json
import argparse
from datetime import datetime
from openai import OpenAI

# Configuration des modèles disponibles (testés et validés)
MODELS = {
    "qwen25-vl-7b": {
        "full_name": "Qwen/Qwen2.5-VL-7B-Instruct",
        "short_name": "qwen25_vl_7b",
        "description": "Qwen2.5-VL-7B (haute précision)"
    },
    "qwen2-vl-2b-ocr": {
        "full_name": "prithivMLmods/Qwen2-VL-OCR-2B-Instruct",
        "short_name": "qwen2_vl_2b_ocr",
        "description": "Qwen2-VL-2B-OCR (rapide - recommandé)"
    }
}

# Configuration par défaut
DEFAULT_MODEL = "qwen2-vl-2b-ocr"
IMAGE_DIR = "./data/clean_plates"
RESULTS_DIR = "./results"
VLLM_URL = "http://localhost:8000/v1"

# Prompt optimisé pour obtenir texte + confiance
PROMPT = """Analyze this license plate image. 
Return ONLY a JSON object with exactly these fields:
- "plate": the plate text in uppercase (letters and numbers only, no spaces or dashes)
- "confidence": your confidence level from 0.0 to 1.0

Example response: {"plate": "AB123CD", "confidence": 0.95}

If you cannot read the plate clearly, use a lower confidence score.
Respond with ONLY the JSON, no other text."""


def encode_image(path):
    """Encode une image en base64."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def clean_plate_text(text):
    """Nettoie le texte de la plaque."""
    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
    return cleaned


def parse_response(response_text):
    """Parse la réponse JSON du modèle."""
    try:
        data = json.loads(response_text)
        plate = clean_plate_text(data.get("plate", ""))
        confidence = float(data.get("confidence", 0.5))
        return plate, min(max(confidence, 0.0), 1.0)
    except json.JSONDecodeError:
        plate = clean_plate_text(response_text)
        return plate, 0.5


def format_time(seconds):
    """Formate le temps en format lisible."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"


def main():
    # Parser les arguments
    parser = argparse.ArgumentParser(description="Test OCR des plaques avec VLM")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL,
                        choices=list(MODELS.keys()),
                        help=f"Modèle à utiliser (défaut: {DEFAULT_MODEL})")
    args = parser.parse_args()
    
    # Récupérer la config du modèle
    model_config = MODELS[args.model]
    model_full_name = model_config["full_name"]
    model_short_name = model_config["short_name"]
    
    # Créer le dossier de résultats pour ce modèle
    model_results_dir = os.path.join(RESULTS_DIR, model_short_name)
    os.makedirs(model_results_dir, exist_ok=True)
    
    # Chemins des fichiers de sortie
    output_csv = os.path.join(model_results_dir, "ground_truth.csv")
    report_file = os.path.join(model_results_dir, "performance.txt")
    
    start_time = time.time()
    
    print("=" * 70)
    print(f"TEST OCR AVEC {model_config['description'].upper()}")
    print("=" * 70)
    print(f"📅 Démarrage: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🤖 Modèle: {model_full_name}")
    print()
    
    # Vérifier que le dossier source existe
    if not os.path.exists(IMAGE_DIR):
        print(f"❌ Erreur: Le dossier {IMAGE_DIR} n'existe pas")
        return
    
    # Lister les images
    files = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    total_files = len(files)
    print(f"📁 {total_files} images trouvées")
    print(f"📂 Résultats: {model_results_dir}")
    print(f"🌐 Connexion à vLLM: {VLLM_URL}")
    print()
    
    # Initialiser le client
    client = OpenAI(api_key="EMPTY", base_url=VLLM_URL)
    
    # Statistiques
    stats = {
        "success": 0,
        "high_confidence": 0,
        "medium_confidence": 0,
        "low_confidence": 0,
        "errors": 0
    }
    
    # Préparation du CSV
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "predicted_text", "confidence", "processing_time_ms"])
        
        for i, filename in enumerate(files):
            img_start_time = time.time()
            img_path = os.path.join(IMAGE_DIR, filename)
            
            try:
                base64_image = encode_image(img_path)
                
                response = client.chat.completions.create(
                    model=model_full_name,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": PROMPT},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }],
                    max_tokens=100
                )
                
                raw_answer = response.choices[0].message.content.strip()
                plate_text, confidence = parse_response(raw_answer)
                processing_time = (time.time() - img_start_time) * 1000
                
                if plate_text:
                    stats["success"] += 1
                    
                    if confidence >= 0.8:
                        stats["high_confidence"] += 1
                    elif confidence >= 0.5:
                        stats["medium_confidence"] += 1
                    else:
                        stats["low_confidence"] += 1
                    
                    writer.writerow([filename, plate_text, round(confidence, 4), round(processing_time, 1)])
                else:
                    stats["errors"] += 1
                    writer.writerow([filename, "ERROR", 0.0, round(processing_time, 1)])
                    
            except Exception as e:
                processing_time = (time.time() - img_start_time) * 1000
                print(f"❌ {filename}: Erreur - {e}")
                stats["errors"] += 1
                writer.writerow([filename, "ERROR", 0.0, round(processing_time, 1)])
            
            if (i + 1) % 100 == 0 or (i + 1) == total_files:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (total_files - i - 1) / rate if rate > 0 else 0
                print(f"📊 Progression: {i + 1}/{total_files} ({(i+1)/total_files*100:.1f}%) | "
                      f"Temps: {format_time(elapsed)} | ETA: {format_time(eta)}")
    
    total_time = time.time() - start_time
    avg_time_per_image = total_time / total_files if total_files > 0 else 0
    
    # Générer le rapport de performances
    report_content = f"""================================================================================
RAPPORT DE PERFORMANCES - {model_config['description'].upper()}
================================================================================

📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🤖 Modèle: {model_full_name}

================================================================================
STATISTIQUES
================================================================================

📁 Images traitées: {total_files}
✅ Succès: {stats['success']} ({stats['success']/total_files*100:.1f}%)
   🟢 Haute confiance (≥0.8): {stats['high_confidence']} ({stats['high_confidence']/total_files*100:.1f}%)
   🟡 Moyenne confiance (0.5-0.8): {stats['medium_confidence']} ({stats['medium_confidence']/total_files*100:.1f}%)
   🔴 Faible confiance (<0.5): {stats['low_confidence']} ({stats['low_confidence']/total_files*100:.1f}%)
❌ Erreurs: {stats['errors']} ({stats['errors']/total_files*100:.1f}%)

================================================================================
TEMPS DE TRAITEMENT
================================================================================

⏱️  Temps total: {format_time(total_time)}
⏱️  Moyenne par image: {avg_time_per_image*1000:.0f} ms
⏱️  Images par minute: {60/avg_time_per_image:.1f}
⏱️  Images par heure: {3600/avg_time_per_image:.0f}

================================================================================
FICHIERS GÉNÉRÉS
================================================================================

📄 Ground Truth CSV: {output_csv}
📊 Rapport performances: {report_file}

================================================================================
"""
    
    # Sauvegarder le rapport
    with open(report_file, 'w') as f:
        f.write(report_content)
    
    # Afficher le résumé
    print()
    print("=" * 70)
    print("RÉSUMÉ")
    print("=" * 70)
    print(f"""
🤖 Modèle: {model_config['description']}
📁 Images traitées: {total_files}
✅ Succès: {stats['success']} ({stats['success']/total_files*100:.1f}%)
   🟢 Haute confiance (≥0.8): {stats['high_confidence']}
   🟡 Moyenne confiance (0.5-0.8): {stats['medium_confidence']}
   🔴 Faible confiance (<0.5): {stats['low_confidence']}
❌ Erreurs: {stats['errors']}

⏱️  TEMPS DE TRAITEMENT:
   Total: {format_time(total_time)}
   Moyenne par image: {avg_time_per_image*1000:.0f}ms
   Images par minute: {60/avg_time_per_image:.1f}

📄 Fichier CSV: {output_csv}
📊 Rapport: {report_file}
""")


if __name__ == "__main__":
    main()
