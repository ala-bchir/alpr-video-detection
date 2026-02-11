#!/usr/bin/env python3
"""
Pipeline complet de traitement vidéo des plaques d'immatriculation.

Workflow:
1. SAM3 (video_processor.py) → Extraction des plaques depuis la vidéo
2. auto_sort_plates.py → Filtrage par ratio
3. GLM-OCR → Lecture OCR des plaques
4. Regex filter → Garder seulement les plaques ≥7 caractères
5. Déduplication → Garder la première occurrence de chaque plaque
6. Export CSV → date_heure_passage, numero_plaque

Usage:
    python3 pipeline.py --video /chemin/vers/video.mp4 --start-time 12:30
"""

import os
import sys
import re
import csv
import argparse
import subprocess
import shutil
import base64
from datetime import datetime, timedelta
from pathlib import Path

import cv2
from openai import OpenAI
from homography import correct_perspective

# ============== CONFIGURATION ==============
DATA_DIR = "./data"
CROP_DIR = "./data/crop"
CLEAN_DIR = "./data/clean_plates"
CORRECTED_DIR = "./data/corrected_plates"
VIDEOS_DIR = "./data/videos"
RESULTS_DIR = "./results"

VLLM_URL = "http://localhost:8000/v1"
GLM_MODEL = "zai-org/GLM-OCR"

# Filtre regex: longueur minimale de la plaque
MIN_PLATE_LENGTH = 5
STANDARD_PLATE_LENGTH = 7

# Prompt OCR optimisé
OCR_PROMPT = """Read the license plate in this image.
Output ONLY the plate text in uppercase (letters and numbers only, no spaces or dashes).
If a character is unreadable or unclear, replace it with an asterisk (*).
No explanation, no formatting, just the characters on the plate."""


def check_glm_server():
    """Vérifie si le serveur GLM-OCR est actif."""
    try:
        import requests
        response = requests.get(f"{VLLM_URL}/models", timeout=5)
        if response.status_code == 200:
            models = response.json().get("data", [])
            for model in models:
                if "GLM-OCR" in model.get("id", ""):
                    return True
            return False
    except Exception:
        pass
    return False


def stop_glm_server():
    """Arrête le serveur GLM-OCR pour libérer la VRAM."""
    print("\n🛑 Arrêt de GLM-OCR pour libérer la VRAM...")
    result = subprocess.run(
        ["make", "stop-vlm"],
        cwd="/home/solayman/sam3_licencePlate_processing",
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print("✅ GLM-OCR arrêté")
    else:
        print(f"⚠️  Problème lors de l'arrêt: {result.stderr}")
    # Attendre un peu pour libérer la VRAM
    import time
    time.sleep(3)


def start_glm_server():
    """Démarre le serveur GLM-OCR."""
    print("\n🚀 Démarrage de GLM-OCR...")
    # Supprimer l'ancien container s'il existe
    subprocess.run(
        ["docker", "rm", "-f", "qwen-labeler"],
        capture_output=True, text=True
    )
    result = subprocess.run(
        ["make", "start-glm-ocr"],
        cwd="/home/solayman/sam3_licencePlate_processing",
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"❌ Erreur au démarrage de GLM-OCR: {result.stderr}")
        return False
    print("✅ GLM-OCR démarré, attente du chargement...")
    return True


def wait_for_glm_server(timeout=180):
    """Attend que le serveur GLM-OCR soit prêt."""
    import time
    start_time = time.time()
    while time.time() - start_time < timeout:
        if check_glm_server():
            print("✅ Serveur GLM-OCR prêt!")
            return True
        print(f"⏳ Attente du chargement... ({int(time.time() - start_time)}s)")
        time.sleep(10)
    print("❌ Timeout: le serveur GLM-OCR ne répond pas")
    return False


def clean_directories():
    """Nettoie les répertoires de travail."""
    for dir_path in [CROP_DIR, CLEAN_DIR, CORRECTED_DIR]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path, exist_ok=True)


def copy_video_to_input(video_path):
    """Copie la vidéo dans le dossier d'entrée si nécessaire."""
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    
    # Si la vidéo est déjà dans data/videos, pas besoin de copier
    abs_video = os.path.abspath(video_path)
    abs_videos_dir = os.path.abspath(VIDEOS_DIR)
    
    if abs_video.startswith(abs_videos_dir):
        print(f"   Vidéo déjà dans {VIDEOS_DIR}, pas de copie nécessaire")
        return video_path
    
    # Sinon, copier la vidéo (sans supprimer les fichiers existants)
    dest = os.path.join(VIDEOS_DIR, os.path.basename(video_path))
    shutil.copy2(video_path, dest)
    return dest


def get_video_fps(video_path):
    """Récupère le FPS de la vidéo."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 30.0  # Défaut: 30 fps


def extract_frame_number(filename):
    """Extrait le numéro de frame du nom de fichier."""
    # Format: plaque_f120_143025.jpg → 120
    match = re.search(r'_f(\d+)_', filename)
    if match:
        return int(match.group(1))
    return 0


def run_sam3_extraction():
    """Exécute l'extraction SAM3 via Docker."""
    print("\n" + "=" * 60)
    print("🔍 ÉTAPE 1: Extraction des plaques avec SAM3")
    print("=" * 60)
    
    result = subprocess.run(
        ["make", "run"],
        cwd="/home/solayman/sam3_licencePlate_processing",
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"❌ Erreur SAM3: {result.stderr}")
        return False
    
    crop_count = len([f for f in os.listdir(CROP_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))])
    print(f"✅ {crop_count} images extraites dans {CROP_DIR}")
    print(f"🚗 SAM3 a détecté {crop_count} passages de véhicules")
    return crop_count > 0


def run_auto_sort():
    """Exécute le tri automatique des plaques."""
    print("\n" + "=" * 60)
    print("🗂️  ÉTAPE 2: Tri automatique des plaques")
    print("=" * 60)
    
    result = subprocess.run(
        ["python3", "auto_sort_plates.py"],
        cwd="/home/solayman/sam3_licencePlate_processing",
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"❌ Erreur tri: {result.stderr}")
        return False
    
    print(result.stdout)
    clean_count = len([f for f in os.listdir(CLEAN_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))])
    print(f"✅ {clean_count} plaques valides dans {CLEAN_DIR}")
    return clean_count > 0


def encode_image(path):
    """Encode une image en base64."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def clean_plate_text(text):
    """Nettoie le texte de la plaque en gardant les * pour les caractères illisibles."""
    # Garde uniquement lettres, chiffres et astérisques
    return re.sub(r'[^A-Z0-9*]', '', text.upper())


def run_glm_ocr():
    """Exécute l'OCR avec GLM-OCR sur les plaques nettoyées."""
    print("\n" + "=" * 60)
    print("🤖 ÉTAPE 3: Lecture OCR avec GLM-OCR")
    print("=" * 60)
    
    client = OpenAI(api_key="EMPTY", base_url=VLLM_URL)
    
    # Lire depuis CORRECTED_DIR (après homographie)
    ocr_dir = CORRECTED_DIR
    files = sorted([f for f in os.listdir(ocr_dir) 
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    
    results = []
    
    for i, filename in enumerate(files):
        img_path = os.path.join(ocr_dir, filename)
        
        try:
            base64_image = encode_image(img_path)
            
            response = client.chat.completions.create(
                model=GLM_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        {"type": "text", "text": OCR_PROMPT}
                    ]
                }],
                max_tokens=50
            )
            
            raw_answer = response.choices[0].message.content.strip()
            plate_text = clean_plate_text(raw_answer)
            
            results.append({
                "filename": filename,
                "plate": plate_text,
                "frame": extract_frame_number(filename)
            })
            
            if (i + 1) % 50 == 0 or (i + 1) == len(files):
                print(f"📊 Progression: {i + 1}/{len(files)} ({(i+1)/len(files)*100:.1f}%)")
                
        except Exception as e:
            print(f"❌ Erreur OCR sur {filename}: {e}")
    
    print(f"✅ {len(results)} plaques lues")
    return results


def has_common_substring(plate1, plate2, min_length=3):
    """Vérifie si deux plaques partagent une sous-chaîne de min_length caractères consécutifs."""
    for i in range(len(plate1) - min_length + 1):
        sub = plate1[i:i + min_length]
        if '*' in sub:
            continue
        if sub in plate2:
            return True
    return False


def shared_chars_count(plate1, plate2):
    """Compte le nombre de caractères individuels en commun (indépendant de la position)."""
    from collections import Counter
    c1 = Counter(c for c in plate1 if c != '*')
    c2 = Counter(c for c in plate2 if c != '*')
    # Intersection : min de chaque caractère
    common = sum((c1 & c2).values())
    return common


def are_duplicates(plate1, plate2, min_common=3):
    """Vérifie si deux plaques sont des doublons (3+ chars consécutifs OU 3+ chars individuels en commun)."""
    return has_common_substring(plate1, plate2) or shared_chars_count(plate1, plate2) >= min_common


def is_french_format(plate):
    """Vérifie si la plaque suit le format français: 2 lettres + 3 chiffres + 2 lettres."""
    if len(plate) < 7:
        return False
    return (plate[0:2].isalpha() and
            plate[2:5].isdigit() and
            plate[5:7].isalpha())


def count_stars(plate):
    """Compte le nombre d'astérisques dans une plaque."""
    return plate.count('*')


def pick_best_plate(group):
    """Parmi un groupe de doublons, choisit la meilleure plaque."""
    # 1. Préférer les plaques complètes (sans étoile)
    complete = [r for r in group if '*' not in r["plate"]]
    
    if complete:
        # 2. Parmi les complètes, préférer le format français
        french = [r for r in complete if is_french_format(r["plate"])]
        if french:
            return french[0]
        return complete[0]
    
    # 3. Sinon garder celle avec le moins d'étoiles
    group.sort(key=lambda r: count_stars(r["plate"]))
    return group[0]


def apply_filters(ocr_results, fps):
    """Applique le filtre de longueur et le padding."""
    print("\n" + "=" * 60)
    print("🔤 ÉTAPE 4: Filtre longueur")
    print("=" * 60)
    
    MAX_PLATE_LENGTH = 10  # au-delà c'est du bruit OCR
    
    # Filtre: garder seulement les plaques entre MIN et MAX longueur
    filtered = [r for r in ocr_results 
                if MIN_PLATE_LENGTH <= len(r["plate"]) <= MAX_PLATE_LENGTH]
    print(f"📋 Après filtre ({MIN_PLATE_LENGTH}-{MAX_PLATE_LENGTH} chars): {len(filtered)}/{len(ocr_results)}")
    
    # Compléter les plaques courtes avec des * pour atteindre STANDARD_PLATE_LENGTH
    for r in filtered:
        if len(r["plate"]) < STANDARD_PLATE_LENGTH:
            missing = STANDARD_PLATE_LENGTH - len(r["plate"])
            r["plate"] = r["plate"] + "*" * missing
    
    return filtered


def calculate_passage_times(results, start_time, fps):
    """Calcule l'heure de passage pour chaque plaque."""
    print("\n" + "=" * 60)
    print("⏱️  ÉTAPE 5: Calcul des heures de passage")
    print("=" * 60)
    
    # Parse start_time (format HH:MM ou HH:MM:SS)
    time_parts = start_time.split(":")
    if len(time_parts) == 2:
        base_time = datetime.now().replace(
            hour=int(time_parts[0]),
            minute=int(time_parts[1]),
            second=0,
            microsecond=0
        )
    else:
        base_time = datetime.now().replace(
            hour=int(time_parts[0]),
            minute=int(time_parts[1]),
            second=int(time_parts[2]),
            microsecond=0
        )
    
    for r in results:
        # Calculer le temps écoulé depuis le début de la vidéo
        seconds_elapsed = r["frame"] / fps
        passage_time = base_time + timedelta(seconds=seconds_elapsed)
        r["passage_time"] = passage_time.strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"✅ Heures calculées (FPS: {fps:.1f})")
    return results


def export_csv(results, video_name):
    """Exporte les résultats en CSV."""
    print("\n" + "=" * 60)
    print("📄 ÉTAPE 6: Export CSV")
    print("=" * 60)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_base = os.path.splitext(os.path.basename(video_name))[0]
    output_file = os.path.join(RESULTS_DIR, f"pipeline_{video_base}_{timestamp}.csv")
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["date_heure_passage", "numero_plaque", "format"])
        
        for r in results:
            writer.writerow([r["passage_time"], r["plate"], r.get("format", "UNKNOWN")])
    
    print(f"✅ CSV exporté: {output_file}")
    print(f"📊 Total: {len(results)} plaques uniques")
    
    # Afficher un aperçu
    print("\n📋 Aperçu (5 premières lignes):")
    print("-" * 40)
    for r in results[:5]:
        print(f"  {r['passage_time']} | {r['plate']} [{r.get('format', '')}]")
    if len(results) > 5:
        print(f"  ... et {len(results) - 5} autres")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de traitement vidéo des plaques d'immatriculation"
    )
    parser.add_argument(
        "--video", "-v",
        type=str,
        required=True,
        help="Chemin vers la vidéo à traiter"
    )
    parser.add_argument(
        "--start-time", "-t",
        type=str,
        required=True,
        help="Heure de début de la vidéo (format HH:MM ou HH:MM:SS)"
    )
    parser.add_argument(
        "--time-window", "-w",
        type=int,
        default=3,
        help="Fenêtre temporelle de déduplication en secondes (défaut: 3)"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎬 PIPELINE DE TRAITEMENT VIDÉO")
    print("=" * 60)
    print(f"📹 Vidéo: {args.video}")
    print(f"⏰ Heure de départ: {args.start_time}")
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Vérifications
    if not os.path.exists(args.video):
        print(f"❌ Erreur: La vidéo '{args.video}' n'existe pas")
        sys.exit(1)
    
    # Récupérer le FPS de la vidéo
    fps = get_video_fps(args.video)
    print(f"📊 FPS vidéo: {fps:.1f}")
    
    # Nettoyer les répertoires
    print("\n🧹 Nettoyage des répertoires de travail...")
    clean_directories()
    
    # Copier la vidéo
    print("📁 Copie de la vidéo...")
    copy_video_to_input(args.video)
    
    # Arrêter GLM-OCR si actif (pour libérer VRAM pour SAM3)
    if check_glm_server():
        stop_glm_server()
    
    # Étape 1: SAM3
    if not run_sam3_extraction():
        print("❌ Échec de l'extraction SAM3")
        sys.exit(1)
    
    # Étape 2: Tri automatique
    if not run_auto_sort():
        print("❌ Aucune plaque valide après le tri")
        sys.exit(1)
    
    # Étape 2.5: Correction de perspective (homographie)
    print("\n" + "=" * 60)
    print("📐 ÉTAPE 2.5: Correction de perspective (homographie)")
    print("=" * 60)
    from homography import process_directory
    process_directory(CLEAN_DIR, CORRECTED_DIR)
    
    # Démarrer GLM-OCR pour l'étape OCR
    if not start_glm_server():
        print("❌ Impossible de démarrer GLM-OCR")
        sys.exit(1)
    
    if not wait_for_glm_server():
        print("❌ Le serveur GLM-OCR n'a pas démarré à temps")
        sys.exit(1)
    
    # Étape 3: OCR
    ocr_results = run_glm_ocr()
    if not ocr_results:
        print("❌ Échec de l'OCR")
        sys.exit(1)
    
    # Étape 4: Filtre longueur
    filtered_results = apply_filters(ocr_results, fps)
    if not filtered_results:
        print("⚠️  Aucune plaque valide après les filtres")
        sys.exit(0)
    
    # Étape 5: Correction regex multi-format
    print("\n" + "=" * 60)
    print("📐 ÉTAPE 5: Correction regex multi-format")
    print("=" * 60)
    from regex_augmente import recognize_plate
    corrected = 0
    for r in filtered_results:
        result = recognize_plate(r["plate"])
        if result.plate and result.format != "UNKNOWN":
            r["plate"] = result.plate
            r["format"] = result.format
            corrected += 1
        else:
            r["format"] = "UNKNOWN"
    print(f"✅ {corrected}/{len(filtered_results)} plaques corrigées et formatées")
    
    # Étape 6: Déduplication intelligente
    print("\n" + "=" * 60)
    print(f"🔄 ÉTAPE 6: Déduplication (fenêtre {args.time_window}s)")
    print("=" * 60)
    from dedup_plates import deduplicate
    deduplicated = deduplicate(filtered_results, fps, time_window=args.time_window)
    print(f"📋 {len(filtered_results)} → {len(deduplicated)} plaques uniques")
    
    # Étape 7: Calcul des heures
    final_results = calculate_passage_times(deduplicated, args.start_time, fps)
    
    # Étape 8: Export
    output_file = export_csv(final_results, args.video)
    
    print("\n" + "=" * 60)
    print("✅ PIPELINE TERMINÉ AVEC SUCCÈS")
    print("=" * 60)
    print(f"📄 Résultat: {output_file}")


if __name__ == "__main__":
    main()
