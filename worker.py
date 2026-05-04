#!/usr/bin/env python3
import os
import sys
import time
import signal
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

# ============================================================
#  CONFIGURATION
# ============================================================
NEXTCLOUD_URL         = os.environ.get("NEXTCLOUD_URL", "").rstrip("/") # Sécurité slash
NEXTCLOUD_USER        = os.environ.get("NEXTCLOUD_USER", "")
NEXTCLOUD_PASS        = os.environ.get("NEXTCLOUD_PASS", "")
NEXTCLOUD_VIDEOS_DIR  = os.environ.get("NEXTCLOUD_VIDEOS_DIR", "/videos_a_traiter").strip("/")
NEXTCLOUD_RESULTS_DIR = os.environ.get("NEXTCLOUD_RESULTS_DIR", "/resultats").strip("/")
POSTE_NAME            = os.environ.get("POSTE_NAME", "UNKNOWN")

FRAME_SKIP            = int(os.environ.get("FRAME_SKIP", "3"))
TIME_WINDOW           = int(os.environ.get("TIME_WINDOW", "3"))
GLM_MODEL             = os.environ.get("GLM_MODEL", "zai-org/GLM-OCR")
VLLM_GPU_UTILIZATION  = os.environ.get("VLLM_GPU_UTILIZATION", "0.3")

BASE_DIR      = "/app"
DATA_DIR      = "/app/data"
VIDEOS_DIR    = "/app/data/videos"
RESULTS_DIR   = "/app/results"
WEIGHTS_PATH  = "/app/sam3_weights.pt"
VLLM_URL      = "http://localhost:8000/v1"
VLLM_PORT     = 8000

_vllm_process = None

# ============================================================
#  FONCTIONS RCLONE (CORRIGÉES)
# ============================================================

def _get_webdav_root():
    return f"{NEXTCLOUD_URL}/remote.php/dav/files/{NEXTCLOUD_USER}"

def _build_rclone_webdav_flags() -> list:
    # Utilisation du mot de passe directement si rclone obscure échoue
    return [
        "--webdav-url",    _get_webdav_root(),
        "--webdav-user",   NEXTCLOUD_USER,
        "--webdav-pass",   NEXTCLOUD_PASS, 
        "--webdav-vendor", "nextcloud",
        "--no-check-certificate", # Optionnel: utile si SSL auto-signé
    ]

def download_videos_from_nextcloud() -> bool:
    print(f"\n📥 Téléchargement depuis : {NEXTCLOUD_VIDEOS_DIR}")
    os.makedirs(VIDEOS_DIR, exist_ok=True)

    rclone_cmd = [
        "rclone", "copy",
        f":webdav:{NEXTCLOUD_VIDEOS_DIR}",
        VIDEOS_DIR,
        "--include", "*.{mp4,avi,mkv,mov,MP4,AVI}",
        "--transfers", "4",
        "--buffer-size", "32M"
    ] + _build_rclone_webdav_flags()

    try:
        result = subprocess.run(rclone_cmd, timeout=600) # Timeout 10min pour le DL
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Erreur rclone download: {e}")
        return False

def upload_results_to_nextcloud(result_files: list):
    if not result_files: return
    print(f"\n📤 Upload vers : {NEXTCLOUD_RESULTS_DIR}/{POSTE_NAME}")
    
    remote_dest = f":webdav:{NEXTCLOUD_RESULTS_DIR}/{POSTE_NAME}"

    for result_file in result_files:
        rclone_cmd = [
            "rclone", "copy", result_file, remote_dest
        ] + _build_rclone_webdav_flags()
        
        subprocess.run(rclone_cmd)

# ============================================================
#  GESTION DU SERVEUR vLLM (CORRIGÉE)
# ============================================================

def start_vllm_server():
    global _vllm_process
    print(f"🚀 Lancement vLLM ({GLM_MODEL})...")

    cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", GLM_MODEL,
        "--port", str(VLLM_PORT),
        "--max-model-len", "2048",
        "--gpu-memory-utilization", VLLM_GPU_UTILIZATION,
        "--trust-remote-code",
        "--disable-log-requests",
    ]

    log_file = open("/app/vllm_server.log", "w")
    _vllm_process = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)

def stop_vllm_server():
    global _vllm_process
    if _vllm_process:
        print("🛑 Arrêt vLLM...")
        _vllm_process.terminate()
        try:
            _vllm_process.wait(timeout=15)
        except:
            _vllm_process.kill()
        _vllm_process = None

# ============================================================
#  PIPELINE PRINCIPALE
# ============================================================

def main():
    # Capture des signaux RunPod (SIGTERM)
    signal.signal(signal.SIGTERM, lambda s, f: stop_vllm_server() or sys.exit(0))

    print("--- DÉMARRAGE WORKER ---")
    
    # 1. Start vLLM en tâche de fond immédiatement
    start_vllm_server()

    # 2. Download vidéos pendant que le modèle charge
    if not download_videos_from_nextcloud():
        print("❌ Échec téléchargement. Arrêt.")
        stop_vllm_server()
        return

    # 3. Import et Monkey-patch de batch_processor
    try:
        import batch_processor as bp
        # On force les chemins pour RunPod
        bp.VIDEOS_BASE_DIR = VIDEOS_DIR
        bp.RESULTS_DIR     = RESULTS_DIR
        bp.CROP_DIR        = os.path.join(DATA_DIR, "crop")
        
        # Lancement du traitement (qui appelle video_processor.py)
        print("\n🎬 Début du traitement des vidéos...")
        result_files = bp.run_full_pipeline_for_poste(POSTE_NAME, FRAME_SKIP, TIME_WINDOW)
        
        # 4. Upload des résultats
        upload_results_to_nextcloud(result_files)
        
    except Exception as e:
        print(f"❌ Erreur fatale pipeline: {e}")
    finally:
        stop_vllm_server()
        print("✅ Fin du worker.")

if __name__ == "__main__":
    main()