#!/usr/bin/env python3
"""
Batch Processor — Traitement multi-vidéos / multi-postes.

Scanne data/videos/ pour trouver les dossiers de postes (CA1, CA3, CA4, …),
traite toutes les vidéos de chaque poste et génère un CSV par poste.

Format vidéo attendu : 2_YYYYMMDD_HHMMSS_0025d5.avi
  → La date et l'heure de début sont extraites du nom du fichier.

Colonnes CSV : Poste, Date, Timestamp, Categorie, Plaque

Usage:
    python3 batch_processor.py
    python3 batch_processor.py --postes CA1 CA3
    python3 batch_processor.py --dry-run
    python3 batch_processor.py --frame-skip 3 --time-window 3
"""

import os
import sys
import re
import csv
import json
import argparse
import subprocess
import shutil
import base64
import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2
from openai import OpenAI

# ============== CONFIGURATION ==============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
VIDEOS_BASE_DIR = os.path.join(DATA_DIR, "videos")
CROP_DIR = os.path.join(DATA_DIR, "crop")
CLEAN_DIR = os.path.join(DATA_DIR, "clean_plates")
CORRECTED_DIR = os.path.join(DATA_DIR, "corrected_plates")
WORK_VIDEO_DIR = os.path.join(DATA_DIR, "work_video")  # Dossier temporaire pour SAM3
CROPS_BASE_DIR = os.path.join(DATA_DIR, "crops_per_video")  # Crops isolés par vidéo
RESULTS_DIR = os.path.join(BASE_DIR, "results")

VLLM_URL = "http://localhost:8000/v1"
GLM_MODEL = "zai-org/GLM-OCR"

MIN_PLATE_LENGTH = 5
STANDARD_PLATE_LENGTH = 7
MAX_PLATE_LENGTH = 10

OCR_PROMPT = """Read the license plate in this image.
Output ONLY the plate text in uppercase (letters and numbers only, no spaces or dashes).
If a character is unreadable or unclear, replace it with an asterisk (*).
No explanation, no formatting, just the characters on the plate."""

# Regex pour parser le nom de la vidéo : 2_YYYYMMDD_HHMMSS_0025d5.avi
VIDEO_NAME_PATTERN = re.compile(
    r'^\d+_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})_[0-9a-fA-F]+\.(avi|AVI)$',
    re.IGNORECASE
)


# ============== UTILITAIRES ==============

def parse_video_name(filename):
    """
    Parse le nom de la vidéo pour extraire la date et l'heure de début.
    Format: 2_YYYYMMDD_HHMMSS_0025d5.avi
    Retourne (date_str, start_datetime) ou (None, None) si le parsing échoue.
    """
    match = VIDEO_NAME_PATTERN.match(filename)
    if not match:
        return None, None

    year, month, day = match.group(1), match.group(2), match.group(3)
    hour, minute, second = match.group(4), match.group(5), match.group(6)

    date_str = f"{year}-{month}-{day}"
    start_dt = datetime(
        int(year), int(month), int(day),
        int(hour), int(minute), int(second)
    )
    return date_str, start_dt


def discover_postes(base_dir, filter_postes=None):
    """
    Découvre les dossiers de postes dans le répertoire de base.
    Retourne une liste triée de (nom_poste, chemin_absolu).
    """
    postes = []
    for entry in sorted(os.listdir(base_dir)):
        full_path = os.path.join(base_dir, entry)
        if os.path.isdir(full_path):
            if filter_postes and entry not in filter_postes:
                continue
            postes.append((entry, full_path))
    return postes


def discover_videos(poste_dir):
    """
    Découvre toutes les vidéos .avi dans un dossier de poste.
    Retourne une liste triée de chemins absolus vers les fichiers vidéo.
    """
    videos = []
    for f in sorted(os.listdir(poste_dir)):
        if f.lower().endswith(('.avi', '.mp4', '.mkv', '.mov')):
            full_path = os.path.join(poste_dir, f)
            if os.path.isfile(full_path):
                videos.append(full_path)
    return videos


def load_mask_config(poste_dir):
    """
    Charge le fichier mask.json d'un poste.
    Retourne la liste des zones de masquage ou [] si pas de fichier.
    """
    mask_file = os.path.join(poste_dir, "mask.json")
    if not os.path.exists(mask_file):
        return []
    try:
        with open(mask_file, 'r') as f:
            data = json.load(f)
        zones = data.get("mask_zones", [])
        if zones:
            print(f"   🎭 Mask chargé: {len(zones)} zone(s) à masquer")
        return zones
    except (json.JSONDecodeError, IOError) as e:
        print(f"   ⚠️  Erreur lecture mask.json: {e}")
        return []


def get_video_fps(video_path):
    """Récupère le FPS de la vidéo."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 25.0


def extract_frame_number(filename):
    """Extrait le numéro de frame du nom de fichier crop."""
    match = re.search(r'_f(\d+)_', filename)
    if match:
        return int(match.group(1))
    return 0


# ============== GLM-OCR SERVER ==============

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
    except Exception:
        pass
    return False


def stop_glm_server():
    """Arrête le serveur GLM-OCR pour libérer la VRAM."""
    print("🛑 Arrêt de GLM-OCR pour libérer la VRAM...")
    subprocess.run(
        ["make", "stop-vlm"],
        cwd=BASE_DIR,
        capture_output=True, text=True
    )
    time.sleep(3)


def start_glm_server():
    """Démarre le serveur GLM-OCR."""
    print("🚀 Démarrage de GLM-OCR...")
    subprocess.run(
        ["docker", "rm", "-f", "qwen-labeler"],
        capture_output=True, text=True
    )
    result = subprocess.run(
        ["make", "start-glm-ocr"],
        cwd=BASE_DIR,
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"❌ Erreur au démarrage de GLM-OCR: {result.stderr}")
        return False
    print("✅ GLM-OCR démarré, attente du chargement...")
    return True


def wait_for_glm_server(timeout=180):
    """Attend que le serveur GLM-OCR soit prêt."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if check_glm_server():
            print("✅ Serveur GLM-OCR prêt!")
            return True
        print(f"⏳ Attente du chargement... ({int(time.time() - start_time)}s)")
        time.sleep(10)
    print("❌ Timeout: le serveur GLM-OCR ne répond pas")
    return False


# ============== PIPELINE STEPS ==============

def clean_work_directories():
    """Nettoie les répertoires de travail entre chaque vidéo."""
    for dir_path in [CROP_DIR, CLEAN_DIR, CORRECTED_DIR, WORK_VIDEO_DIR]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path, exist_ok=True)


def prepare_video_for_sam3(video_path):
    """
    Copie la vidéo dans le dossier de travail pour SAM3.
    SAM3 via Docker lit depuis data/videos, donc on utilise un dossier temporaire.
    """
    dest = os.path.join(WORK_VIDEO_DIR, os.path.basename(video_path))
    shutil.copy2(video_path, dest)
    return dest


def run_sam3_extraction(frame_skip=1, mask_zones=None):
    """Exécute l'extraction SAM3 via Docker avec sortie en temps réel."""
    # Construire la commande Docker
    docker_cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "--ipc=host",
        "--ulimit", "memlock=-1",
        "--ulimit", "stack=67108864",
        "-e", f"FRAME_SKIP={frame_skip}",
    ]

    # Passer MASK_ZONES comme env var Docker
    if mask_zones:
        docker_cmd += ["-e", f"MASK_ZONES={json.dumps(mask_zones)}"]

    docker_cmd += [
        "-v", f"{WORK_VIDEO_DIR}:/app/data/videos",
        "-v", f"{os.path.join(DATA_DIR, 'images')}:/app/data/images",
        "-v", f"{CROP_DIR}:/app/data/crop",
        "-v", f"{os.path.join(BASE_DIR, 'sam3_weights.pt')}:/app/sam3_weights.pt",
        "-v", f"{os.path.join(BASE_DIR, 'video_processor.py')}:/app/video_processor.py",
        "sam3-video-processor",
    ]

    # Streaming: on affiche la sortie en temps réel
    process = subprocess.Popen(
        docker_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # Lire et afficher chaque ligne en temps réel
    for line in process.stdout:
        line = line.rstrip()
        if line:
            print(f"      {line}")

    process.wait()

    if process.returncode != 0:
        print(f"   ❌ SAM3 erreur (code {process.returncode})")
        return False

    crop_count = len([f for f in os.listdir(CROP_DIR)
                      if f.endswith(('.jpg', '.png', '.jpeg'))])
    print(f"   ✅ {crop_count} crops extraits")
    return crop_count > 0


def run_auto_sort():
    """Exécute le tri automatique des plaques."""
    result = subprocess.run(
        ["python3", "auto_sort_plates.py"],
        cwd=BASE_DIR,
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"   ❌ Erreur tri: {result.stderr[:300]}")
        return False

    clean_count = len([f for f in os.listdir(CLEAN_DIR)
                       if f.endswith(('.jpg', '.png', '.jpeg'))])
    print(f"   ✅ {clean_count} plaques valides après tri")
    return clean_count > 0


def run_homography():
    """Correction de perspective (homographie)."""
    from homography import process_directory
    process_directory(CLEAN_DIR, CORRECTED_DIR)


def encode_image(path):
    """Encode une image en base64."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def clean_plate_text(text):
    """Nettoie le texte de la plaque."""
    return re.sub(r'[^A-Z0-9*]', '', text.upper())


def run_glm_ocr():
    """Exécute l'OCR avec GLM-OCR sur les plaques corrigées."""
    client = OpenAI(api_key="EMPTY", base_url=VLLM_URL)

    files = sorted([f for f in os.listdir(CORRECTED_DIR)
                    if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

    if not files:
        print("   ⚠️  Aucune image à OCR")
        return []

    results = []
    for i, filename in enumerate(files):
        img_path = os.path.join(CORRECTED_DIR, filename)
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
        except Exception as e:
            print(f"   ❌ Erreur OCR sur {filename}: {e}")

    print(f"   ✅ {len(results)} plaques lues par OCR")
    return results


def apply_filters(ocr_results):
    """Applique le filtre de longueur et le padding."""
    filtered = [r for r in ocr_results
                if MIN_PLATE_LENGTH <= len(r["plate"]) <= MAX_PLATE_LENGTH]

    for r in filtered:
        if len(r["plate"]) < STANDARD_PLATE_LENGTH:
            missing = STANDARD_PLATE_LENGTH - len(r["plate"])
            r["plate"] = r["plate"] + "*" * missing

    print(f"   📋 Filtre longueur: {len(filtered)}/{len(ocr_results)}")
    return filtered


def apply_regex_correction(results):
    """Correction regex multi-format."""
    from regex_augmente import recognize_plate
    corrected = 0
    for r in results:
        result = recognize_plate(r["plate"])
        if result.plate and result.format != "UNKNOWN":
            r["plate"] = result.plate
            r["format"] = result.format
            corrected += 1
        else:
            r["format"] = "UNKNOWN"
    print(f"   ✅ {corrected}/{len(results)} plaques corrigées (regex)")
    return results


def apply_deduplication(results, fps, time_window=3):
    """Déduplication intelligente."""
    from dedup_plates import deduplicate

    dedup1 = deduplicate(results, fps, time_window=time_window)
    dedup2 = deduplicate(dedup1, fps, time_window=time_window)
    print(f"   📋 Dédup: {len(results)} → {len(dedup2)} plaques uniques")
    return dedup2


def calculate_timestamps(results, start_datetime, fps):
    """
    Calcule l'heure de passage pour chaque plaque.
    Retourne une liste de dicts avec 'date', 'timestamp', 'plate'.
    """
    final = []
    for r in results:
        seconds_elapsed = r["frame"] / fps
        passage_time = start_datetime + timedelta(seconds=seconds_elapsed)
        final.append({
            "date": start_datetime.strftime("%Y-%m-%d"),
            "timestamp": passage_time.strftime("%H:%M:%S"),
            "plate": r["plate"],
            "format": r.get("format", "UNKNOWN"),
            "frame": r["frame"]
        })
    return final


def apply_same_second_filter(results):
    """Filtre anti-hallucination : une seule plaque par seconde."""
    from collections import defaultdict

    FORMAT_PRIORITY = {"SIV": 0, "FNI": 1, "UNKNOWN": 2}
    groups = defaultdict(list)
    for r in results:
        key = f"{r['date']} {r['timestamp']}"
        groups[key].append(r)

    cleaned = []
    for time_key, plates in sorted(groups.items()):
        best = min(plates, key=lambda r: (
            FORMAT_PRIORITY.get(r.get("format", "UNKNOWN"), 2),
            r["plate"].count("*")
        ))
        cleaned.append(best)

    if len(results) != len(cleaned):
        print(f"   🧹 Anti-hallucination: {len(results)} → {len(cleaned)}")
    return cleaned


def remove_exact_duplicates(results):
    """Supprime les doublons exacts (même texte de plaque)."""
    seen = set()
    unique = []
    for r in results:
        if r["plate"] not in seen:
            seen.add(r["plate"])
            unique.append(r)
    if len(results) != len(unique):
        print(f"   🔁 Doublons exacts: {len(results)} → {len(unique)}")
    return unique


# ============== TRAITEMENT D'UNE VIDÉO ==============

def process_single_video(video_path, frame_skip=1, time_window=3, glm_ready=False):
    """
    Traite une seule vidéo à travers le pipeline complet.
    Retourne une liste de résultats [{date, timestamp, plate}, ...] ou [].
    """
    video_name = os.path.basename(video_path)
    date_str, start_dt = parse_video_name(video_name)

    if not date_str:
        print(f"   ⚠️  Nom de vidéo non reconnu: {video_name}, ignoré")
        return []

    print(f"\n   📹 {video_name} (début: {start_dt.strftime('%H:%M:%S')})")

    # Nettoyer les dossiers de travail
    clean_work_directories()

    # Copier la vidéo dans le dossier de travail
    prepare_video_for_sam3(video_path)

    # Récupérer le FPS
    fps = get_video_fps(video_path)

    # Étape 1: SAM3
    print(f"   🔍 SAM3 (frame_skip={frame_skip})...")
    if not run_sam3_extraction(frame_skip=frame_skip):
        print(f"   ⚠️  Aucune plaque détectée, passage à la vidéo suivante")
        return []

    # Étape 2: Tri automatique
    print("   🗂️  Auto-sort...")
    if not run_auto_sort():
        print(f"   ⚠️  Aucune plaque valide après tri")
        return []

    # Étape 3: Homographie
    print("   📐 Homographie...")
    run_homography()

    # Étape 4: OCR
    print("   🤖 OCR GLM...")
    ocr_results = run_glm_ocr()
    if not ocr_results:
        print(f"   ⚠️  OCR n'a retourné aucun résultat")
        return []

    # Étape 5: Filtre longueur
    filtered = apply_filters(ocr_results)
    if not filtered:
        print(f"   ⚠️  Aucune plaque valide après filtrage")
        return []

    # Étape 6: Correction regex
    corrected = apply_regex_correction(filtered)

    # Étape 7: Déduplication
    deduped = apply_deduplication(corrected, fps, time_window)

    # Étape 8: Calcul des timestamps
    timestamped = calculate_timestamps(deduped, start_dt, fps)

    # Étape 9: Filtre même seconde
    cleaned = apply_same_second_filter(timestamped)

    # Étape 10: Doublons exacts
    unique = remove_exact_duplicates(cleaned)

    print(f"   ✅ {len(unique)} plaques détectées dans cette vidéo")
    return unique


# ============== EXPORT CSV ==============

def export_poste_csv(poste_name, all_results):
    """
    Exporte les résultats d'un poste entier dans un seul CSV.
    Colonnes: Poste, Date, Timestamp, Categorie, Plaque
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(RESULTS_DIR, f"batch_{poste_name}_{timestamp}.csv")

    # Trier tous les résultats par date puis timestamp
    all_results.sort(key=lambda r: (r["date"], r["timestamp"]))

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Poste", "Date", "Timestamp", "Categorie", "Plaque"])

        for r in all_results:
            writer.writerow([
                poste_name,
                r["date"],
                r["timestamp"],
                "VL",
                r["plate"]
            ])

    print(f"\n📄 CSV exporté: {output_file}")
    print(f"📊 Total: {len(all_results)} plaques")

    # Aperçu
    if all_results:
        print("\n📋 Aperçu (5 premières lignes):")
        print("-" * 60)
        for r in all_results[:5]:
            print(f"  {poste_name} | {r['date']} | {r['timestamp']} | VL | {r['plate']}")
        if len(all_results) > 5:
            print(f"  ... et {len(all_results) - 5} autres")

    return output_file


# ============== MAIN ==============

def main():
    parser = argparse.ArgumentParser(
        description="Batch Processor - Traitement multi-vidéos / multi-postes"
    )
    parser.add_argument(
        "--postes", "-p",
        nargs="+",
        default=None,
        help="Liste des postes à traiter (ex: CA1 CA3). Par défaut: tous."
    )
    parser.add_argument(
        "--frame-skip", "-f",
        type=int,
        default=3,
        help="Sauter N frames entre chaque traitement SAM3 (défaut: 3)"
    )
    parser.add_argument(
        "--time-window", "-w",
        type=int,
        default=3,
        help="Fenêtre temporelle de déduplication en secondes (défaut: 3)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche les postes et vidéos détectés sans traitement"
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=0,
        help="Nombre max de vidéos à traiter par poste (0 = toutes)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("🎬 BATCH PROCESSOR — TRAITEMENT MULTI-POSTES")
    print("=" * 70)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏭️  Frame skip: {args.frame_skip}")
    print(f"🔄 Fenêtre dédup: {args.time_window}s")

    # Découvrir les postes
    postes = discover_postes(VIDEOS_BASE_DIR, args.postes)
    if not postes:
        print(f"❌ Aucun poste trouvé dans {VIDEOS_BASE_DIR}")
        sys.exit(1)

    print(f"\n📂 {len(postes)} poste(s) détecté(s): {', '.join(p[0] for p in postes)}")

    # Découvrir les vidéos par poste
    poste_videos = {}
    total_videos = 0
    for poste_name, poste_dir in postes:
        videos = discover_videos(poste_dir)
        # Filtrer uniquement les vidéos dont le nom est parsable
        valid_videos = []
        for v in videos:
            date_str, start_dt = parse_video_name(os.path.basename(v))
            if date_str:
                valid_videos.append(v)
        poste_videos[poste_name] = valid_videos
        total_videos += len(valid_videos)
        print(f"   📁 {poste_name}: {len(valid_videos)} vidéo(s)")

    if total_videos == 0:
        print("❌ Aucune vidéo valide trouvée")
        sys.exit(1)

    # Mode dry-run : afficher les vidéos et sortir
    if args.dry_run:
        print("\n" + "=" * 70)
        print("🔍 MODE DRY-RUN — Liste des vidéos détectées")
        print("=" * 70)
        for poste_name, videos in poste_videos.items():
            print(f"\n📁 {poste_name} ({len(videos)} vidéos):")
            for v in videos:
                vname = os.path.basename(v)
                date_str, start_dt = parse_video_name(vname)
                size_mb = os.path.getsize(v) / (1024 * 1024)
                print(f"   🎥 {vname} | {date_str} {start_dt.strftime('%H:%M:%S')} | {size_mb:.0f} MB")
        print(f"\n📊 Total: {total_videos} vidéos dans {len(postes)} postes")
        print("✅ Dry-run terminé (aucun traitement effectué)")
        return

    # ============ TRAITEMENT RÉEL ============

    def fmt_duration(secs):
        """Formate une durée en texte lisible."""
        h, r = divmod(int(secs), 3600)
        m, s = divmod(r, 60)
        if h > 0:
            return f"{h}h{m:02d}m{s:02d}s"
        elif m > 0:
            return f"{m}m{s:02d}s"
        return f"{s}s"

    def fmt_size(bytes_val):
        """Formate une taille en MB/GB."""
        mb = bytes_val / (1024 * 1024)
        if mb >= 1024:
            return f"{mb / 1024:.1f} GB"
        return f"{mb:.0f} MB"

    def log_progress(msg):
        """Écrit un message dans le fichier de log + stdout."""
        print(msg)
        try:
            with open(progress_log, 'a') as f:
                f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
        except Exception:
            pass

    # Fichier de log pour suivi en temps réel
    os.makedirs(RESULTS_DIR, exist_ok=True)
    progress_log = os.path.join(RESULTS_DIR, "batch_progress.log")
    with open(progress_log, 'w') as f:
        f.write(f"=== BATCH PROCESSOR — Démarré {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        f.write(f"Postes: {', '.join(p[0] for p in postes)}\n")
        f.write(f"Vidéos total: {total_videos}\n\n")

    log_progress(f"\n📝 Log: {progress_log}")
    log_progress(f"   (suivre: tail -f {progress_log})\n")

    generated_csvs = []
    global_start = time.time()

    for poste_idx, (poste_name, poste_dir) in enumerate(postes):
        videos = poste_videos[poste_name]
        if args.max_videos > 0:
            videos = videos[:args.max_videos]

        poste_start = time.time()
        poste_total_size = sum(os.path.getsize(v) for v in videos)

        log_progress("\n" + "=" * 70)
        log_progress(f"📂 POSTE {poste_idx + 1}/{len(postes)}: {poste_name} ({len(videos)} vidéos, {fmt_size(poste_total_size)})")
        log_progress("=" * 70)

        # Charger le mask pour ce poste
        mask_zones = load_mask_config(poste_dir)

        # Préparer le dossier de crops par vidéo
        poste_crops_dir = os.path.join(CROPS_BASE_DIR, poste_name)
        if os.path.exists(poste_crops_dir):
            shutil.rmtree(poste_crops_dir)
        os.makedirs(poste_crops_dir, exist_ok=True)

        # ══════════════════════════════════════════════════════
        # PHASE 1 : SAM3 + Sort + Homographie sur TOUTES les vidéos
        # ══════════════════════════════════════════════════════
        log_progress(f"\n{'━' * 70}")
        log_progress(f"🔬 PHASE 1/{poste_name}: SAM3 + Tri + Homographie ({len(videos)} vidéos)")
        log_progress(f"{'━' * 70}")

        # Arrêter GLM-OCR si actif (libérer la VRAM pour SAM3)
        if check_glm_server():
            stop_glm_server()

        videos_with_crops = []  # (video_path, crops_dir, video_name, fps, date_str, start_dt)
        video_times = []

        for vid_idx, video_path in enumerate(videos):
            video_name = os.path.basename(video_path)
            video_size = os.path.getsize(video_path)
            video_start = time.time()

            date_str, start_dt = parse_video_name(video_name)
            fps = get_video_fps(video_path)

            # Progress bar
            progress_pct = (vid_idx / len(videos)) * 100
            bar_len = 20
            filled = int(bar_len * vid_idx / len(videos))
            bar = "█" * filled + "░" * (bar_len - filled)

            if video_times:
                avg_t = sum(video_times) / len(video_times)
                eta = fmt_duration(avg_t * (len(videos) - vid_idx))
            else:
                eta = "calcul..."

            log_progress(f"\n{'─' * 60}")
            log_progress(f"[{bar}] {progress_pct:.0f}% — Phase 1 — Vidéo {vid_idx + 1}/{len(videos)} — ETA: {eta}")
            log_progress(f"📹 {video_name} ({fmt_size(video_size)}, {start_dt.strftime('%H:%M:%S')}, FPS:{fps:.0f})")

            # Nettoyer les dossiers de travail
            clean_work_directories()
            prepare_video_for_sam3(video_path)

            # SAM3
            log_progress(f"   🔍 SAM3 (frame_skip={args.frame_skip})...")
            step_start = time.time()
            has_crops = run_sam3_extraction(frame_skip=args.frame_skip, mask_zones=mask_zones)
            sam3_dur = time.time() - step_start
            log_progress(f"   ⏱️  SAM3: {fmt_duration(sam3_dur)}")

            if not has_crops:
                log_progress(f"   ⏭️  Aucune plaque → vidéo suivante")
                video_times.append(time.time() - video_start)
                continue

            # Auto-sort
            log_progress("   🗂️  Auto-sort...")
            if not run_auto_sort():
                log_progress(f"   ⏭️  Rien après tri → vidéo suivante")
                video_times.append(time.time() - video_start)
                continue

            # Homographie
            log_progress("   📐 Homographie...")
            run_homography()

            # Sauvegarder les crops corrigés dans un dossier par vidéo
            video_base = os.path.splitext(video_name)[0]
            video_crop_dir = os.path.join(poste_crops_dir, video_base)
            os.makedirs(video_crop_dir, exist_ok=True)

            corrected_count = 0
            for f in os.listdir(CORRECTED_DIR):
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    shutil.copy2(os.path.join(CORRECTED_DIR, f), os.path.join(video_crop_dir, f))
                    corrected_count += 1

            video_dur = time.time() - video_start
            video_times.append(video_dur)

            if corrected_count > 0:
                videos_with_crops.append({
                    "video_path": video_path,
                    "crops_dir": video_crop_dir,
                    "video_name": video_name,
                    "fps": fps,
                    "date_str": date_str,
                    "start_dt": start_dt,
                    "crop_count": corrected_count
                })
                log_progress(f"   ✅ {corrected_count} crops sauvegardés ({fmt_duration(video_dur)})")
            else:
                log_progress(f"   ⏭️  Aucun crop valide ({fmt_duration(video_dur)})")

        phase1_dur = time.time() - poste_start
        log_progress(f"\n📊 Phase 1 terminée: {len(videos_with_crops)}/{len(videos)} vidéos avec plaques ({fmt_duration(phase1_dur)})")
        total_crops = sum(v["crop_count"] for v in videos_with_crops)
        log_progress(f"   📷 Total crops à OCR: {total_crops}")

        if not videos_with_crops:
            log_progress(f"\n⚠️  Poste {poste_name}: aucune plaque détectée dans aucune vidéo")
            continue

        # ══════════════════════════════════════════════════════
        # PHASE 2 : OCR + Filtres sur TOUS les crops
        # ══════════════════════════════════════════════════════
        log_progress(f"\n{'━' * 70}")
        log_progress(f"🤖 PHASE 2/{poste_name}: OCR + Filtres ({len(videos_with_crops)} vidéos, {total_crops} crops)")
        log_progress(f"{'━' * 70}")

        # Démarrer GLM-OCR UNE SEULE FOIS pour tout le poste
        if not check_glm_server():
            log_progress("   🚀 Démarrage GLM-OCR (une seule fois pour tout le poste)...")
            if not start_glm_server():
                log_progress("   ❌ Impossible de démarrer GLM-OCR")
                continue
            if not wait_for_glm_server():
                log_progress("   ❌ GLM-OCR timeout")
                continue

        phase2_start = time.time()
        poste_results = []

        for vid_idx, vinfo in enumerate(videos_with_crops):
            log_progress(f"\n   {'─' * 40}")
            log_progress(f"   📋 OCR {vid_idx + 1}/{len(videos_with_crops)}: {vinfo['video_name']} ({vinfo['crop_count']} crops)")

            # Lire les crops de cette vidéo pour l'OCR
            client = OpenAI(api_key="EMPTY", base_url=VLLM_URL)
            files = sorted([f for f in os.listdir(vinfo["crops_dir"])
                           if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

            ocr_results = []
            for filename in files:
                img_path = os.path.join(vinfo["crops_dir"], filename)
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
                    ocr_results.append({
                        "filename": filename,
                        "plate": plate_text,
                        "frame": extract_frame_number(filename)
                    })
                except Exception as e:
                    pass  # Silencieux pour la vitesse

            if not ocr_results:
                log_progress(f"   ⚠️  OCR sans résultat")
                continue

            log_progress(f"   ✅ {len(ocr_results)} plaques lues")

            # Filtres
            filtered = apply_filters(ocr_results)
            if not filtered:
                continue

            corrected = apply_regex_correction(filtered)
            deduped = apply_deduplication(corrected, vinfo["fps"], args.time_window)
            timestamped = calculate_timestamps(deduped, vinfo["start_dt"], vinfo["fps"])
            cleaned = apply_same_second_filter(timestamped)
            unique = remove_exact_duplicates(cleaned)

            poste_results.extend(unique)
            log_progress(f"   📊 {len(unique)} plaques uniques (total poste: {len(poste_results)})")

        phase2_dur = time.time() - phase2_start
        log_progress(f"\n📊 Phase 2 terminée: {len(poste_results)} plaques ({fmt_duration(phase2_dur)})")

        # ── Export CSV pour ce poste ──
        poste_dur = time.time() - poste_start
        if poste_results:
            csv_path = export_poste_csv(poste_name, poste_results)
            generated_csvs.append(csv_path)
            log_progress(f"\n✅ Poste {poste_name}: {len(poste_results)} plaques en {fmt_duration(poste_dur)}")
        else:
            log_progress(f"\n⚠️  Poste {poste_name}: aucun résultat ({fmt_duration(poste_dur)})")

        # Arrêter GLM-OCR pour le prochain poste (SAM3 a besoin de VRAM)
        if check_glm_server():
            stop_glm_server()

    # ============ RÉSUMÉ FINAL ============
    elapsed = time.time() - global_start

    summary = []
    summary.append("\n" + "=" * 70)
    summary.append("✅ BATCH PROCESSOR TERMINÉ")
    summary.append("=" * 70)
    summary.append(f"⏱️  Durée totale: {fmt_duration(elapsed)}")
    summary.append(f"📂 Postes traités: {len(postes)}")
    summary.append(f"🎬 Vidéos traitées: {total_videos}")

    summary.append(f"\n📄 Fichiers CSV générés:")
    for csv_path in generated_csvs:
        summary.append(f"   → {csv_path}")

    if not generated_csvs:
        summary.append("   ⚠️  Aucun CSV généré")

    summary.append(f"\n📝 Log complet: {progress_log}")

    for line in summary:
        log_progress(line)


if __name__ == "__main__":
    main()
