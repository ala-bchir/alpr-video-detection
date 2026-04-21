#!/usr/bin/env python3
"""
worker.py — Worker RunPod pour le traitement cloud de vidéos ALPR.

Ce script est le point d'entrée du conteneur RunPod. Il orchestre:
  1. Téléchargement des vidéos depuis Nextcloud via rclone (WebDAV)
  2. Démarrage du serveur GLM-OCR (vLLM) en arrière-plan
  3. Traitement de chaque vidéo : SAM3 → OCR → filtres → CSV
  4. Upload des CSV de résultats sur Nextcloud

ARCHITECTURE RUNPOD (pas de Docker-in-Docker) :
  Sur le poste local, batch_processor.py lance SAM3 via `docker run`.
  Ici, on remplace cette fonction par un appel direct à video_processor.py
  (même machine, même conteneur).

Variables d'environnement obligatoires :
  NEXTCLOUD_URL          — URL de base Nextcloud (ex: https://cloud.example.com)
  NEXTCLOUD_USER         — Nom d'utilisateur Nextcloud
  NEXTCLOUD_PASS         — Mot de passe ou token d'application Nextcloud
  NEXTCLOUD_VIDEOS_DIR   — Chemin distant des vidéos (ex: /videos_a_traiter/CA1)
  NEXTCLOUD_RESULTS_DIR  — Chemin de destination des CSV (ex: /resultats)
  POSTE_NAME             — Nom du poste traité (ex: CA1)

Variables d'environnement optionnelles :
  FRAME_SKIP             — Frames à sauter par SAM3 (défaut: 3)
  TIME_WINDOW            — Fenêtre de déduplication en secondes (défaut: 3)
  GLM_MODEL              — Modèle OCR vLLM (défaut: zai-org/GLM-OCR)
  VIDEO_LIST             — Vidéos spécifiques séparées par virgule (défaut: toutes)
  MAX_VIDEOS             — Nombre max de vidéos à traiter (0 = toutes)
  HF_TOKEN               — Token HuggingFace (pour modèles privés)
  VLLM_GPU_UTILIZATION   — Part de VRAM pour vLLM (défaut: 0.4)
"""

import os
import sys
import time
import signal
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


# ============================================================
#  CONFIGURATION (depuis les variables d'environnement)
# ============================================================

NEXTCLOUD_URL          = os.environ.get("NEXTCLOUD_URL", "")
NEXTCLOUD_USER         = os.environ.get("NEXTCLOUD_USER", "")
NEXTCLOUD_PASS         = os.environ.get("NEXTCLOUD_PASS", "")
NEXTCLOUD_VIDEOS_DIR   = os.environ.get("NEXTCLOUD_VIDEOS_DIR", "/videos_a_traiter")
NEXTCLOUD_RESULTS_DIR  = os.environ.get("NEXTCLOUD_RESULTS_DIR", "/resultats")
POSTE_NAME             = os.environ.get("POSTE_NAME", "UNKNOWN")

FRAME_SKIP             = int(os.environ.get("FRAME_SKIP", "3"))
TIME_WINDOW            = int(os.environ.get("TIME_WINDOW", "3"))
GLM_MODEL              = os.environ.get("GLM_MODEL", "zai-org/GLM-OCR")
VIDEO_LIST             = os.environ.get("VIDEO_LIST", "")
MAX_VIDEOS             = int(os.environ.get("MAX_VIDEOS", "0"))
HF_TOKEN               = os.environ.get("HF_TOKEN", "")
VLLM_GPU_UTILIZATION   = os.environ.get("VLLM_GPU_UTILIZATION", "0.3")

# Chemins internes au conteneur
BASE_DIR      = "/app"
DATA_DIR      = "/app/data"
VIDEOS_DIR    = "/app/data/videos"
RESULTS_DIR   = "/app/results"
WEIGHTS_PATH  = "/app/sam3_weights.pt"
VLLM_URL      = "http://localhost:8000/v1"
VLLM_PORT     = 8000

# Handle global du processus vLLM
_vllm_process = None


# ============================================================
#  VALIDATION
# ============================================================

def validate_config():
    """Vérifie que toutes les variables d'environnement obligatoires sont définies."""
    required = {
        "NEXTCLOUD_URL":         NEXTCLOUD_URL,
        "NEXTCLOUD_USER":        NEXTCLOUD_USER,
        "NEXTCLOUD_PASS":        NEXTCLOUD_PASS,
        "NEXTCLOUD_VIDEOS_DIR":  NEXTCLOUD_VIDEOS_DIR,
        "NEXTCLOUD_RESULTS_DIR": NEXTCLOUD_RESULTS_DIR,
        "POSTE_NAME":            POSTE_NAME,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        print(f"❌ Variables d'environnement manquantes : {', '.join(missing)}")
        print("   → Configure-les dans l'interface RunPod (Pod > Edit > Environment Variables)")
        sys.exit(1)

    if not os.path.exists(WEIGHTS_PATH):
        print(f"❌ Poids SAM3 introuvables : {WEIGHTS_PATH}")
        sys.exit(1)

    print("✅ Configuration validée")
    print(f"   Poste          : {POSTE_NAME}")
    print(f"   Vidéos source  : {NEXTCLOUD_URL}{NEXTCLOUD_VIDEOS_DIR}")
    print(f"   Résultats vers : {NEXTCLOUD_URL}{NEXTCLOUD_RESULTS_DIR}")
    print(f"   Frame skip     : {FRAME_SKIP}")
    print(f"   Time window    : {TIME_WINDOW}s")
    print(f"   Modèle OCR     : {GLM_MODEL}")


# ============================================================
#  NEXTCLOUD — WebDAV via rclone
# ============================================================

def _get_webdav_root():
    """Construit l'URL WebDAV racine de l'utilisateur Nextcloud."""
    url = NEXTCLOUD_URL.rstrip("/")
    return f"{url}/remote.php/dav/files/{NEXTCLOUD_USER}"


def _rclone_obscure(password: str) -> str:
    """Chiffre le mot de passe avec `rclone obscure` (format attendu par rclone)."""
    result = subprocess.run(
        ["rclone", "obscure", password],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        return result.stdout.strip()
    # Fallback : retourner le mdp en clair (rclone accepte aussi --webdav-pass en clair
    # si on utilise --no-check-password, mais obscure est préférable)
    print("   ⚠️  rclone obscure a échoué, utilisation du mot de passe en clair")
    return password


def _build_rclone_webdav_flags() -> list:
    """Retourne les flags rclone communs pour la connexion WebDAV Nextcloud."""
    return [
        "--webdav-url",    _get_webdav_root(),
        "--webdav-user",   NEXTCLOUD_USER,
        "--webdav-pass",   _rclone_obscure(NEXTCLOUD_PASS),
        "--webdav-vendor", "nextcloud",
    ]


def download_videos_from_nextcloud() -> bool:
    """
    Télécharge les vidéos depuis Nextcloud vers /app/data/videos/.
    Utilise rclone en mode WebDAV inline (pas de fichier de config).
    Retourne True si au moins une vidéo a été téléchargée.
    """
    print(f"\n{'='*60}")
    print(f"📥  Téléchargement des vidéos depuis Nextcloud")
    print(f"    Source : {NEXTCLOUD_VIDEOS_DIR}")
    print(f"{'='*60}")

    os.makedirs(VIDEOS_DIR, exist_ok=True)

    rclone_cmd = [
        "rclone", "copy",
        f":webdav:{NEXTCLOUD_VIDEOS_DIR}",
        VIDEOS_DIR,
        "--include", "*.avi",
        "--include", "*.AVI",
        "--include", "*.mp4",
        "--include", "*.MP4",
        "--include", "*.mkv",
        "--include", "*.mov",
        "--include", "*.265",
        "--progress",
        "--stats", "15s",
        "--transfers", "4",
    ] + _build_rclone_webdav_flags()

    print("   🔄 rclone copy en cours...")
    result = subprocess.run(rclone_cmd, text=True)

    if result.returncode != 0:
        print(f"   ❌ Erreur rclone (code {result.returncode})")
        return False

    videos = _list_local_videos()
    if not videos:
        print("   ⚠️  Aucune vidéo téléchargée (dossier distant vide ?)")
        return False

    print(f"   ✅ {len(videos)} vidéo(s) disponible(s) :")
    for v in videos:
        size_mb = os.path.getsize(v) / (1024 * 1024)
        print(f"      • {os.path.basename(v)}  ({size_mb:.0f} MB)")
    return True


def upload_results_to_nextcloud(result_files: list):
    """
    Upload les fichiers CSV de résultats sur Nextcloud.
    Les résultats sont placés dans NEXTCLOUD_RESULTS_DIR/POSTE_NAME/.
    """
    if not result_files:
        print("   ⚠️  Aucun fichier à uploader")
        return

    print(f"\n{'='*60}")
    print(f"📤  Upload des résultats sur Nextcloud")
    print(f"    Destination : {NEXTCLOUD_RESULTS_DIR}/{POSTE_NAME}/")
    print(f"{'='*60}")

    remote_dest = f":webdav:{NEXTCLOUD_RESULTS_DIR}/{POSTE_NAME}"

    for result_file in result_files:
        filename = os.path.basename(result_file)
        print(f"   📄 Upload : {filename} ...", end=" ", flush=True)

        rclone_cmd = [
            "rclone", "copy",
            result_file,
            remote_dest,
        ] + _build_rclone_webdav_flags()

        result = subprocess.run(rclone_cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print("✅")
        else:
            print(f"❌\n   Erreur : {result.stderr[:300]}")


# ============================================================
#  vLLM — Serveur GLM-OCR en arrière-plan
# ============================================================

def start_vllm_server():
    """
    Démarre le serveur vLLM pour GLM-OCR en arrière-plan.
    On le démarre tôt pour qu'il charge le modèle pendant le téléchargement
    des vidéos, maximisant ainsi l'utilisation du temps d'attente.
    """
    global _vllm_process

    print(f"\n{'='*60}")
    print(f"🚀  Démarrage du serveur GLM-OCR (vLLM)")
    print(f"    Modèle : {GLM_MODEL}")
    print(f"    GPU utilization : {VLLM_GPU_UTILIZATION}")
    print(f"{'='*60}")

    env = os.environ.copy()
    if HF_TOKEN:
        env["HF_TOKEN"] = HF_TOKEN
        env["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

    cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model",                   GLM_MODEL,
        "--port",                    str(VLLM_PORT),
        "--max-model-len",           "2048",
        "--gpu-memory-utilization",  VLLM_GPU_UTILIZATION,
        "--allowed-local-media-path", "/",
        "--trust-remote-code",
        "--no-enable-log-requests",
    ]

    log_path = "/app/vllm_server.log"
    log_file = open(log_path, "w")

    _vllm_process = subprocess.Popen(cmd, stdout=log_file, stderr=log_file, env=env)
    print(f"   ⏳ Serveur démarré (PID {_vllm_process.pid})")
    print(f"   📄 Logs : {log_path}")
    print(f"   (Le modèle se charge pendant le téléchargement des vidéos...)")


def wait_for_vllm_server(timeout: int = 300) -> bool:
    """
    Attend que le serveur vLLM soit prêt en interrogeant /v1/models.
    Retourne True si le serveur répond avant le timeout.
    """
    import requests as _requests

    print(f"\n   ⏳ Attente du serveur GLM-OCR (max {timeout}s)...")
    start = time.time()

    while time.time() - start < timeout:
        # Vérifier que le processus est toujours vivant
        if _vllm_process and _vllm_process.poll() is not None:
            print(f"\n   ❌ Le serveur vLLM s'est arrêté prématurément (code {_vllm_process.returncode})")
            print(f"   → Consulte /app/vllm_server.log pour les détails")
            return False

        try:
            resp = _requests.get(f"{VLLM_URL}/models", timeout=5)
            if resp.status_code == 200:
                elapsed = int(time.time() - start)
                print(f"\n   ✅ Serveur GLM-OCR prêt ! ({elapsed}s)")
                return True
        except Exception:
            pass

        elapsed = int(time.time() - start)
        print(f"   ⏳ Chargement du modèle... {elapsed}s/{timeout}s", end="\r")
        time.sleep(10)

    print(f"\n   ❌ Timeout ({timeout}s) : le serveur GLM-OCR ne répond pas")
    print(f"   → Consulte /app/vllm_server.log pour les détails")
    return False


def stop_vllm_server():
    """Arrête proprement le serveur vLLM."""
    global _vllm_process
    if _vllm_process and _vllm_process.poll() is None:
        print("🛑 Arrêt du serveur GLM-OCR...")
        _vllm_process.terminate()
        try:
            _vllm_process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            _vllm_process.kill()
        _vllm_process = None
        print("   ✅ Serveur arrêté")


# ============================================================
#  SAM3 — Extraction directe (sans Docker)
# ============================================================

def run_sam3_direct(frame_skip: int = 3, mask_zones=None) -> bool:
    """
    Version directe de l'extraction SAM3 pour RunPod (remplace
    run_sam3_extraction() de batch_processor qui utilisait Docker).

    Appelle video_processor.py comme sous-processus Python normal,
    en passant les paramètres via variables d'environnement (même
    mécanisme que le Dockerfile original utilisait avec -e FRAME_SKIP=...).
    """
    import json as _json

    print(f"   🔍 SAM3 direct (frame_skip={frame_skip})...")

    env = os.environ.copy()
    env["FRAME_SKIP"]   = str(frame_skip)
    env["PYTHONPATH"]   = "/app/sam3_repo"
    env["WEIGHTS_PATH"] = WEIGHTS_PATH
    # Combine max_split_size (fragmentation) + expandable_segments (gros modèles comme SAM3)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"

    if mask_zones:
        env["MASK_ZONES"] = _json.dumps(mask_zones)
    else:
        env.pop("MASK_ZONES", None)

    result = subprocess.run(
        ["python3", "/app/video_processor.py"],
        env=env,
        # Pas de capture : on laisse stdout/stderr s'afficher en temps réel
    )

    if result.returncode != 0:
        print(f"   ❌ SAM3 erreur (code {result.returncode})")
        return False

    # Compter les crops produits
    import batch_processor as bp
    crop_count = len([
        f for f in os.listdir(bp.CROP_DIR)
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ])
    print(f"   ✅ {crop_count} crops extraits")
    return crop_count > 0


# ============================================================
#  LISTE DES VIDÉOS LOCALES
# ============================================================

def _list_local_videos() -> list:
    """
    Liste les vidéos locales dans VIDEOS_DIR.
    Applique les filtres VIDEO_LIST et MAX_VIDEOS si définis.
    """
    if not os.path.exists(VIDEOS_DIR):
        return []

    extensions = ('.avi', '.mp4', '.mkv', '.mov', '.265')
    videos = sorted([
        os.path.join(VIDEOS_DIR, f)
        for f in os.listdir(VIDEOS_DIR)
        if f.lower().endswith(extensions)
    ])

    # Filtre par liste explicite de fichiers
    if VIDEO_LIST:
        wanted = {v.strip() for v in VIDEO_LIST.split(",") if v.strip()}
        videos = [v for v in videos if os.path.basename(v) in wanted]

    # Limite max
    if MAX_VIDEOS > 0:
        videos = videos[:MAX_VIDEOS]

    return videos


# ============================================================
#  PIPELINE COMPLÈTE
# ============================================================

def run_pipeline() -> list:
    """
    Exécute la pipeline complète de traitement pour toutes les vidéos.

    Strategy de monkey-patching :
      - On importe batch_processor normalement
      - On remplace run_sam3_extraction → run_sam3_direct (sans Docker)
      - On reconfigure les chemins pour pointer vers /app/data/
      - Le reste de la pipeline (OCR, filtres, CSV) est réutilisé tel quel

    Retourne la liste des fichiers CSV générés.
    """
    sys.path.insert(0, BASE_DIR)
    import batch_processor as bp

    # ── Monkey-patch : remplacer l'appel Docker par l'appel direct ──
    bp.run_sam3_extraction = run_sam3_direct

    # ── Reconfigurer les chemins du batch_processor ──────────────
    bp.BASE_DIR         = BASE_DIR
    bp.DATA_DIR         = DATA_DIR
    bp.VIDEOS_BASE_DIR  = VIDEOS_DIR
    bp.CROP_DIR         = os.path.join(DATA_DIR, "crop")
    bp.CLEAN_DIR        = os.path.join(DATA_DIR, "clean_plates")
    bp.CORRECTED_DIR    = os.path.join(DATA_DIR, "corrected_plates")
    bp.WORK_VIDEO_DIR   = os.path.join(DATA_DIR, "work_video")
    bp.IMAGES_DIR       = os.path.join(DATA_DIR, "images")
    bp.RESULTS_DIR      = RESULTS_DIR
    bp.VLLM_URL         = VLLM_URL

    # ── Lister les vidéos ─────────────────────────────────────────
    videos = _list_local_videos()
    if not videos:
        print("❌ Aucune vidéo locale à traiter")
        return []

    print(f"\n{'='*60}")
    print(f"🎬  PIPELINE — {len(videos)} vidéo(s) | Poste : {POSTE_NAME}")
    print(f"    frame_skip={FRAME_SKIP}  time_window={TIME_WINDOW}s")
    print(f"{'='*60}")

    # Préparer les répertoires
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for d in [bp.CROP_DIR, bp.CLEAN_DIR, bp.CORRECTED_DIR,
              bp.WORK_VIDEO_DIR, bp.IMAGES_DIR]:
        os.makedirs(d, exist_ok=True)

    # ── Charger la config de masque si présente ───────────────────
    mask_zones = bp.load_mask_config(VIDEOS_DIR)

    # ── Traiter chaque vidéo ──────────────────────────────────────
    all_results = []
    for idx, video_path in enumerate(videos, 1):
        video_name = os.path.basename(video_path)
        print(f"\n[{idx}/{len(videos)}] 📹 {video_name}")

        results = bp.process_single_video(
            video_path=video_path,
            frame_skip=FRAME_SKIP,
            time_window=TIME_WINDOW,
            glm_ready=True,
        )
        all_results.extend(results)

    # ── Dédoublonnage global inter-vidéos ─────────────────────────
    if all_results:
        before = len(all_results)
        all_results = bp.remove_exact_duplicates(all_results)
        if len(all_results) < before:
            print(f"   🔁 Doublons inter-vidéos supprimés : {before} → {len(all_results)}")

    # ── Export CSV ────────────────────────────────────────────────
    result_files = []
    if all_results:
        csv_file = bp.export_poste_csv(POSTE_NAME, all_results)
        result_files.append(csv_file)
        print(f"\n📊 Total : {len(all_results)} plaque(s) détectée(s) pour le poste {POSTE_NAME}")
    else:
        print(f"\n⚠️  Aucune plaque détectée pour le poste {POSTE_NAME}")

    return result_files


# ============================================================
#  NETTOYAGE & SIGNAUX
# ============================================================

def _cleanup_on_exit(signum, frame):
    """Handler pour arrêt propre sur SIGTERM/SIGINT (RunPod envoie SIGTERM)."""
    print(f"\n⚠️  Signal {signum} reçu — arrêt propre en cours...")
    stop_vllm_server()
    sys.exit(0)


# ============================================================
#  POINT D'ENTRÉE
# ============================================================

def main():
    start_time = datetime.now()

    print("=" * 60)
    print("🏃  WORKER RUNPOD — ALPR PIPELINE")
    print(f"    Démarrage : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Enregistrer les handlers de signal (RunPod envoie SIGTERM à l'arrêt)
    signal.signal(signal.SIGTERM, _cleanup_on_exit)
    signal.signal(signal.SIGINT,  _cleanup_on_exit)

    try:
        # ── Étape 1 : Validation de la configuration ──────────────
        print("\n[1/5] Validation de la configuration...")
        validate_config()

        # ── Étape 2 : Démarrage du serveur GLM-OCR ────────────────
        # On le démarre EN PREMIER pour qu'il charge le modèle pendant
        # le téléchargement des vidéos (optimisation temps d'attente).
        print("\n[2/5] Démarrage du serveur GLM-OCR...")
        start_vllm_server()

        # ── Étape 3 : Téléchargement des vidéos ───────────────────
        print("\n[3/5] Téléchargement des vidéos depuis Nextcloud...")
        if not download_videos_from_nextcloud():
            print("❌ Aucune vidéo à traiter — arrêt du worker")
            stop_vllm_server()
            sys.exit(1)

        # ── Étape 4 : Attendre que vLLM soit prêt ─────────────────
        print("\n[4/5] Vérification du serveur GLM-OCR...")
        if not wait_for_vllm_server(timeout=300):
            print("❌ Serveur GLM-OCR non disponible — arrêt du worker")
            stop_vllm_server()
            sys.exit(1)

        # ── Étape 5 : Pipeline SAM3 + OCR + CSV ───────────────────
        print("\n[5/5] Exécution de la pipeline ALPR...")
        result_files = run_pipeline()

        # ── Upload des résultats ───────────────────────────────────
        upload_results_to_nextcloud(result_files)

        # ── Arrêt propre ──────────────────────────────────────────
        stop_vllm_server()

        elapsed = datetime.now() - start_time
        hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)

        print("\n" + "=" * 60)
        print("🎉  WORKER TERMINÉ AVEC SUCCÈS")
        print(f"    Durée totale : {hours:02d}h {minutes:02d}m {seconds:02d}s")
        print(f"    CSV uploadés : {len(result_files)}")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ ERREUR FATALE : {e}")
        import traceback
        traceback.print_exc()
        stop_vllm_server()
        sys.exit(1)


if __name__ == "__main__":
    main()
