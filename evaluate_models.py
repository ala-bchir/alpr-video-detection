#!/usr/bin/env python3
"""
Script d'évaluation des modèles OCR sur le dataset de test.
Compare les prédictions avec le ground truth et calcule les métriques.

Usage:
    python3 evaluate_model.py --model qwen2-vl-2b-ocr
    python3 evaluate_model.py --model qwen25-vl-7b
"""
import os
import csv
import base64
import re
import time
import argparse
from datetime import datetime
from openai import OpenAI

# Configuration des modèles
MODELS = {
    "qwen25-vl-7b": {
        "full_name": "Qwen/Qwen2.5-VL-7B-Instruct",
        "short_name": "qwen25_vl_7b",
        "description": "Qwen2.5-VL-7B"
    },
    "qwen2-vl-2b-ocr": {
        "full_name": "prithivMLmods/Qwen2-VL-OCR-2B-Instruct",
        "short_name": "qwen2_vl_2b_ocr",
        "description": "Qwen2-VL-2B-OCR"
    },
    "glm-ocr": {
        "full_name": "zai-org/GLM-OCR",
        "short_name": "glm_ocr",
        "description": "GLM-OCR (0.9B)"
    },
    "rolm-ocr": {
        "full_name": "reducto/RolmOCR",
        "short_name": "rolm_ocr",
        "description": "RolmOCR"
    },
    "chandra": {
        "full_name": "datalab-to/chandra",
        "short_name": "chandra",
        "description": "Chandra"
    },
    "lighton-ocr": {
        "full_name": "lightonai/LightOnOCR-2-1B",
        "short_name": "lighton_ocr",
        "description": "LightOnOCR-2-1B",
        "prompt": "",
        "image_only": True
    },
    "dots-ocr": {
        "full_name": "rednote-hilab/dots.ocr",
        "short_name": "dots_ocr",
        "description": "Dots.OCR"
    },
    "paddle-ocr": {
        "full_name": "PaddlePaddle/PaddleOCR-VL",
        "short_name": "paddle_ocr",
        "description": "PaddleOCR-VL",
        "prompt": "OCR:"
    },
    "deepseek-ocr2": {
        "full_name": "deepseek-ai/DeepSeek-OCR-2",
        "short_name": "deepseek_ocr2",
        "description": "DeepSeek-OCR-2",
        "prompt": "<image>\nFree OCR."
    }
}

# Configuration
DEFAULT_MODEL = "qwen2-vl-2b-ocr"
TEST_DIR = "./dataset/test"
GROUND_TRUTH_FILE = "./dataset/test/test_gt.csv"
RESULTS_DIR = "./results"
VLLM_URL = "http://localhost:8000/v1"

# Prompt simple - juste demander le texte
PROMPT = """Read the license plate in this image.
Output ONLY the plate text in uppercase (letters and numbers only, no spaces or dashes).
No explanation, no formatting, just the characters on the plate."""


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def clean_plate_text(text):
    """Nettoie le texte de la plaque."""
    return re.sub(r'[^A-Z0-9]', '', text.upper())


def load_ground_truth():
    """Charge le ground truth depuis le fichier CSV."""
    gt = {}
    with open(GROUND_TRUTH_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt[row['filename']] = row['plate_text']
    return gt


def levenshtein_distance(s1, s2):
    """Calcule la distance de Levenshtein entre deux chaînes."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def calculate_cer(predicted, expected):
    """
    Calcule le Character Error Rate (CER).
    CER = (Substitutions + Insertions + Deletions) / Nombre de caractères de référence
    """
    if not expected:
        return 1.0 if predicted else 0.0
    
    distance = levenshtein_distance(predicted, expected)
    return distance / len(expected)


def calculate_wer(predicted, expected):
    """
    Calcule le Word Error Rate (WER).
    Pour les plaques, on considère la plaque entière comme un "mot".
    WER = 0 si exact match, 1 sinon.
    """
    return 0.0 if predicted == expected else 1.0


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"


def main():
    parser = argparse.ArgumentParser(description="Évaluation OCR sur dataset de test")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL,
                        choices=list(MODELS.keys()),
                        help=f"Modèle à évaluer (défaut: {DEFAULT_MODEL})")
    args = parser.parse_args()
    
    model_config = MODELS[args.model]
    model_full_name = model_config["full_name"]
    model_short_name = model_config["short_name"]
    # Utiliser le prompt personnalisé si défini, sinon le prompt par défaut
    model_prompt = model_config.get("prompt", PROMPT)
    
    # Créer le dossier de résultats
    model_results_dir = os.path.join(RESULTS_DIR, model_short_name)
    os.makedirs(model_results_dir, exist_ok=True)
    
    output_csv = os.path.join(model_results_dir, "evaluation.csv")
    report_file = os.path.join(model_results_dir, "evaluation_report.txt")
    
    print("=" * 70)
    print(f"ÉVALUATION: {model_config['description'].upper()}")
    print("=" * 70)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🤖 Modèle: {model_full_name}")
    print()
    
    # Charger le ground truth
    ground_truth = load_ground_truth()
    print(f"📄 Ground truth: {len(ground_truth)} entrées")
    
    # Lister les images
    files = sorted([f for f in os.listdir(TEST_DIR) 
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    total_files = len(files)
    print(f"📁 Images de test: {total_files}")
    print()
    
    # Initialiser le client
    client = OpenAI(api_key="EMPTY", base_url=VLLM_URL)
    
    # Statistiques
    stats = {
        "total": 0,
        "exact_match": 0,
        "errors": 0,
        "total_cer": 0.0,
        "total_wer": 0.0
    }
    
    start_time = time.time()
    
    # Évaluation
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "expected", "predicted", "correct", "cer", "wer", "time_ms"])
        
        for i, filename in enumerate(files):
            img_start = time.time()
            img_path = os.path.join(TEST_DIR, filename)
            expected = ground_truth.get(filename, "")
            
            try:
                base64_image = encode_image(img_path)
                
                # Construire le contenu du message
                if model_config.get("image_only"):
                    # Mode image seule (ex: LightOnOCR)
                    content = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}]
                else:
                    # Mode standard: image + prompt
                    content = [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        {"type": "text", "text": model_prompt}
                    ]
                
                response = client.chat.completions.create(
                    model=model_full_name,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=4096 if model_config.get("image_only") else 50
                )
                
                raw_answer = response.choices[0].message.content.strip()
                predicted = clean_plate_text(raw_answer)
                proc_time = (time.time() - img_start) * 1000
                
                # Calculer les métriques
                is_correct = (predicted == expected)
                cer = calculate_cer(predicted, expected)
                wer = calculate_wer(predicted, expected)
                
                stats["total"] += 1
                stats["total_cer"] += cer
                stats["total_wer"] += wer
                if is_correct:
                    stats["exact_match"] += 1
                
                writer.writerow([filename, expected, predicted, int(is_correct), 
                               round(cer, 4), round(wer, 4), round(proc_time, 1)])
                
            except Exception as e:
                proc_time = (time.time() - img_start) * 1000
                print(f"❌ {filename}: {e}")
                stats["errors"] += 1
                stats["total"] += 1
                stats["total_cer"] += 1.0
                stats["total_wer"] += 1.0
                writer.writerow([filename, expected, "ERROR", 0, 1.0, 1.0, round(proc_time, 1)])
            
            if (i + 1) % 100 == 0 or (i + 1) == total_files:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (total_files - i - 1) / rate if rate > 0 else 0
                acc = stats["exact_match"] / stats["total"] * 100 if stats["total"] > 0 else 0
                print(f"📊 {i+1}/{total_files} ({(i+1)/total_files*100:.1f}%) | "
                      f"Accuracy: {acc:.1f}% | ETA: {format_time(eta)}")
    
    total_time = time.time() - start_time
    avg_time = total_time / total_files if total_files > 0 else 0
    
    # Calculer les métriques finales
    accuracy = stats["exact_match"] / stats["total"] * 100 if stats["total"] > 0 else 0
    avg_cer = stats["total_cer"] / stats["total"] if stats["total"] > 0 else 0
    avg_wer = stats["total_wer"] / stats["total"] if stats["total"] > 0 else 0
    
    # Générer le rapport
    report = f"""================================================================================
RAPPORT D'ÉVALUATION - {model_config['description'].upper()}
================================================================================

📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🤖 Modèle: {model_full_name}
📁 Dataset: {TEST_DIR}
📊 Ground Truth: {GROUND_TRUTH_FILE}

================================================================================
MÉTRIQUES DE PRÉCISION
================================================================================

📊 Total images: {stats['total']}
✅ Exact Match (Accuracy): {stats['exact_match']}/{stats['total']} ({accuracy:.2f}%)
📉 Character Error Rate (CER): {avg_cer:.4f} ({avg_cer*100:.2f}%)
📉 Word Error Rate (WER): {avg_wer:.4f} ({avg_wer*100:.2f}%)
❌ Erreurs API: {stats['errors']}

================================================================================
TEMPS DE TRAITEMENT
================================================================================

⏱️  Temps total: {format_time(total_time)}
⏱️  Moyenne par image: {avg_time*1000:.0f} ms
⏱️  Images par minute: {60/avg_time:.1f}
⏱️  Images par heure: {3600/avg_time:.0f}

================================================================================
FICHIERS GÉNÉRÉS
================================================================================

📄 Résultats détaillés: {output_csv}
📊 Ce rapport: {report_file}

================================================================================
"""
    
    with open(report_file, 'w') as f:
        f.write(report)
    
    print()
    print("=" * 70)
    print("RÉSULTATS FINAUX")
    print("=" * 70)
    print(f"""
🤖 Modèle: {model_config['description']}
📊 Accuracy: {accuracy:.2f}%
📉 CER: {avg_cer*100:.2f}%
📉 WER: {avg_wer*100:.2f}%
⏱️  Temps: {format_time(total_time)} ({avg_time*1000:.0f}ms/image)

📄 Rapport: {report_file}
""")


if __name__ == "__main__":
    main()
