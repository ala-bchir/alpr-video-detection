#!/usr/bin/env python3
"""
modal_app.py — Application Modal pour le pipeline ALPR.

Remplace le Dockerfile.runpod + worker.py pour le déploiement serverless.
Chaque vidéo est traitée sur son propre GPU L40S en parallèle.

Usage:
    # Tester sur une vidéo
    modal run modal_app.py --poste CA1 --max-videos 1

    # Traiter tout un poste (10 vidéos en parallèle)
    modal run modal_app.py --poste CA1

    # Tester que l'image se construit correctement
    modal run modal_app.py::check_deps

    # Déploiement permanent (optionnel)
    modal deploy modal_app.py

Prérequis:
    pip install modal && python -m modal setup
    modal secret create nextcloud-creds \
        NEXTCLOUD_URL="https://cloud.itecmobility.com" \
        NEXTCLOUD_USER="Ala" \
        NEXTCLOUD_PASS="..." \
        NEXTCLOUD_VIDEOS_DIR="/lorient/lorient1" \
        NEXTCLOUD_RESULTS_DIR="/resultats"
    modal secret create hf-token HF_TOKEN=hf_...
    modal volume create alpr-weights
    modal volume put alpr-weights sam3_weights.pt /sam3_weights.pt
"""

import modal
import os
import sys
import time
import json
import subprocess
import shutil
from datetime import datetime

# ============================================================
#  CONFIGURATION MODAL
# ============================================================

app = modal.App("alpr-pipeline")

# Volume persistant pour les poids SAM3 et le cache HuggingFace
volume = modal.Volume.from_name("alpr-weights", create_if_missing=True)
VOL_PATH = "/vol"

# Secrets (créés via CLI : modal secret create ...)
nextcloud_secret = modal.Secret.from_name("nextcloud-creds")
hf_secret = modal.Secret.from_name("hf-token")


# ============================================================
#  IMAGE (remplace le Dockerfile)
# ============================================================

alpr_image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:24.03-py3")

    # ── 1. Dépendances système ─────────────────────────────────
    .apt_install(
        "git", "curl", "unzip", "ffmpeg",
        "libgl1", "libglib2.0-0", "libgomp1", "sed",
    )

    # ── 2. rclone (client WebDAV pour Nextcloud) ───────────────
    .run_commands("curl https://rclone.org/install.sh | bash")

    # ── 3. Nettoyage OpenCV NVIDIA corrompu ─────────────────────
    .run_commands(
        "rm -rf /usr/local/lib/python3.10/dist-packages/cv2",
        "rm -rf /usr/local/lib/python3.10/dist-packages/opencv_*",
    )

    # ── 4. vLLM ────────────────────────────────────────────────
    .pip_install("vllm")

    # ── 4b. Recompilation flash-attn (nécessite un GPU) ────────
    .run_commands(
        "MAX_JOBS=4 pip install --no-cache-dir --force-reinstall flash-attn --no-build-isolation",
        gpu="L40S",
    )

    # ── 5. Recompilation soxr pour Numpy 2.x ───────────────────
    .run_commands(
        "pip install --no-cache-dir --force-reinstall --no-binary=soxr soxr"
    )

    # ── 6. Dépendances SAM3 (sans numpy<2) ─────────────────────
    .pip_install(
        "scipy", "contourpy", "iopath>=0.1.10",
        "ftfy==6.1.1", "timm>=1.0.17", "regex",
    )

    # ── 7. Dépendances Python du projet ────────────────────────
    .pip_install(
        "openai", "requests", "colorama", "Pillow",
        "matplotlib", "pycocotools", "huggingface_hub", "einops",
        "decord", "moviepy", "av", "hydra-core", "omegaconf",
        "psutil", "tqdm",
    )

    # ── 8. SAM3 repo + install ─────────────────────────────────
    .run_commands(
        "git clone https://github.com/facebookresearch/sam3.git /app/sam3_repo",
        "pip install --no-cache-dir --no-deps -e /app/sam3_repo",
    )

    # ── 8a. Patch fused.py — désactiver le cast bf16 forcé ─────
    .run_commands(
        r"""printf '%s\n' \
        '# Patched: removed forced bfloat16 cast for float32 inference' \
        'import torch' \
        '' \
        'def addmm_act(activation, linear, mat1):' \
        '    x = torch.nn.functional.linear(mat1, linear.weight, linear.bias)' \
        '    if activation in [torch.nn.functional.gelu, torch.nn.GELU]:' \
        '        return torch.nn.functional.gelu(x)' \
        '    if activation in [torch.nn.functional.relu, torch.nn.ReLU]:' \
        '        return torch.nn.functional.relu(x)' \
        '    raise ValueError(f"Unexpected activation {activation}")' \
        > /app/sam3_repo/sam3/perflib/fused.py"""
    )

    # ── 8b. Transformers >= 5.0 (pour glm_ocr) ────────────────
    .pip_install("transformers>=5.0.0")

    # ── 9. Répertoires de travail ──────────────────────────────
    .run_commands(
        "mkdir -p /app/data/videos /app/data/images /app/data/crop "
        "/app/data/clean_plates /app/data/corrected_plates "
        "/app/data/work_video /app/data/tmp_h265 /app/results"
    )

    # ── 10. Variables d'environnement ──────────────────────────
    .env({
        "PYTHONPATH": "/app/sam3_repo:/app/code",
        "HF_HOME": f"{VOL_PATH}/huggingface",
        "TRANSFORMERS_CACHE": f"{VOL_PATH}/huggingface",
    })

    # ── 11. Code source Python ─────────────────────────────────
    # Copie les fichiers .py nécessaires au pipeline
    .add_local_file("video_processor.py", "/app/code/video_processor.py")
    .add_local_file("batch_processor.py", "/app/code/batch_processor.py")
    .add_local_file("auto_sort_plates.py", "/app/code/auto_sort_plates.py")
    .add_local_file("homography.py", "/app/code/homography.py")
    .add_local_file("regex_augmente.py", "/app/code/regex_augmente.py")
    .add_local_file("dedup_plates.py", "/app/code/dedup_plates.py")
)


# ============================================================
#  CONSTANTES INTERNES
# ============================================================

CODE_DIR = "/app/code"
DATA_DIR = "/app/data"
VIDEOS_DIR = "/app/data/videos"
RESULTS_DIR = "/app/results"
VLLM_URL = "http://localhost:8000/v1"
VLLM_PORT = 8000


# ============================================================
#  HELPERS NEXTCLOUD (rclone WebDAV)
# ============================================================

def _get_webdav_root():
    """Construit l'URL WebDAV racine Nextcloud de façon robuste."""
    url = os.environ["NEXTCLOUD_URL"].rstrip("/")
    user = os.environ["NEXTCLOUD_USER"]
    
    # Si l'utilisateur a déjà mis le chemin complet, on l'utilise tel quel
    if "remote.php/dav" in url:
        return url
    
    # Sinon on le construit
    return f"{url}/remote.php/dav/files/{user}"


def _rclone_obscure(password):
    """Chiffre le mot de passe pour rclone."""
    result = subprocess.run(
        ["rclone", "obscure", password],
        capture_output=True, text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else password


def _build_rclone_flags():
    """Flags rclone communs pour la connexion WebDAV."""
    return [
        "--webdav-url", _get_webdav_root(),
        "--webdav-user", os.environ["NEXTCLOUD_USER"],
        "--webdav-pass", _rclone_obscure(os.environ["NEXTCLOUD_PASS"]),
        "--webdav-vendor", "nextcloud",
    ]


def _download_single_video(video_name, dest_dir, poste_name):
    """Télécharge UNE vidéo spécifique depuis Nextcloud."""
    base_dir = os.environ.get("NEXTCLOUD_VIDEOS_DIR", "/lorient/lorient1")
    videos_dir = f"{base_dir}/{poste_name}"
    os.makedirs(dest_dir, exist_ok=True)

    # rclone copy télécharge un fichier spécifique avec --include
    cmd = [
        "rclone", "copy",
        f":webdav:{videos_dir}",
        dest_dir,
        "--include", video_name,
        "--progress",
        "--stats", "10s",
    ] + _build_rclone_flags()

    print(f"   📥 Téléchargement de {video_name} depuis {videos_dir}...")
    result = subprocess.run(cmd, text=True)
    dest_file = os.path.join(dest_dir, video_name)

    if result.returncode != 0 or not os.path.exists(dest_file):
        print(f"   ❌ Échec du téléchargement de {video_name}")
        return None

    size_mb = os.path.getsize(dest_file) / (1024 * 1024)
    print(f"   ✅ {video_name} ({size_mb:.0f} MB)")
    return dest_file


def _list_remote_videos(poste_name):
    """Liste les fichiers vidéo d'un poste sur Nextcloud via rclone lsf."""
    base_dir = os.environ.get("NEXTCLOUD_VIDEOS_DIR", "/lorient/lorient1")
    videos_dir = f"{base_dir}/{poste_name}"

    print(f"   📂 Dossier Nextcloud : {videos_dir}")

    cmd = [
        "rclone", "lsf",
        f":webdav:{videos_dir}",
        "--include", "*.avi",
        "--include", "*.AVI",
        "--include", "*.mp4",
        "--include", "*.MP4",
        "--include", "*.mkv",
        "--include", "*.mov",
        "--include", "*.265",
    ] + _build_rclone_flags()

    print(f"   🔍 Debug: Listing root directory :webdav:/ ...")
    debug_cmd = ["rclone", "lsf", ":webdav:/"] + _build_rclone_flags()
    debug_res = subprocess.run(debug_cmd, capture_output=True, text=True)
    print(f"      Root contains: {debug_res.stdout.strip().replace(chr(10), ', ')}")

    print(f"   🔍 Debug: Listing :webdav:{base_dir} ...")
    debug_cmd2 = ["rclone", "lsf", f":webdav:{base_dir}"] + _build_rclone_flags()
    debug_res2 = subprocess.run(debug_cmd2, capture_output=True, text=True)
    if debug_res2.returncode == 0:
        print(f"      {base_dir} contains: {debug_res2.stdout.strip().replace(chr(10), ', ')}")
    else:
        print(f"      ❌ {base_dir} introuvable ! ({debug_res2.stderr.strip()})")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Erreur rclone lsf: {result.stderr[:300]}")
        return []

    return sorted([f.strip() for f in result.stdout.strip().split("\n") if f.strip()])


def _upload_results(result_files, poste_name):
    """Upload les CSV de résultats sur Nextcloud."""
    results_dir = os.environ.get("NEXTCLOUD_RESULTS_DIR", "/resultats")
    remote_dest = f":webdav:{results_dir}/{poste_name}"

    for filepath in result_files:
        filename = os.path.basename(filepath)
        cmd = ["rclone", "copy", filepath, remote_dest] + _build_rclone_flags()
        result = subprocess.run(cmd, capture_output=True, text=True)
        status = "✅" if result.returncode == 0 else "❌"
        print(f"   {status} Upload: {filename}")


# ============================================================
#  FONCTION : LISTER LES VIDÉOS (sans GPU)
# ============================================================

@app.function(
    image=alpr_image,
    secrets=[nextcloud_secret],
    timeout=120,
)
def fetch_video_list(poste_name: str) -> list:
    """Liste les vidéos d'un poste sur Nextcloud. Pas besoin de GPU."""
    print(f"📋 Récupération des vidéos du poste {poste_name}...")
    videos = _list_remote_videos(poste_name)
    print(f"   ✅ {len(videos)} vidéo(s) trouvée(s)")
    for v in videos:
        print(f"      • {v}")
    return videos


# ============================================================
#  FONCTION : VÉRIFIER LES DÉPENDANCES
# ============================================================

@app.function(image=alpr_image, gpu="L40S", timeout=120)
def check_deps():
    """Vérifie que toutes les dépendances sont correctement compilées."""
    import torch
    import flash_attn
    import soxr
    import vllm
    from sam3.model_builder import build_sam3_image_model

    print(f"✅ PyTorch     {torch.__version__}")
    print(f"✅ flash-attn  {flash_attn.__version__}")
    print(f"✅ soxr        OK")
    print(f"✅ vLLM        {vllm.__version__}")
    print(f"✅ SAM3        OK")
    print(f"✅ GPU         {torch.cuda.get_device_name(0)}")
    print(f"✅ VRAM        {torch.cuda.mem_get_info()[1] / 1024**3:.0f} Go")


# ============================================================
#  CLASSE WORKER : SAM3 + vLLM SUR LE MÊME GPU
# ============================================================

@app.cls(
    image=alpr_image,
    gpu="L40S",
    volumes={VOL_PATH: volume},
    secrets=[nextcloud_secret, hf_secret],
    timeout=86400,               # 24h max par vidéo pour éviter les interruptions
    scaledown_window=180,        # Garde le container chaud 3 min entre les vidéos
)
class ALPRWorker:
    """
    Worker ALPR complet. Au démarrage du conteneur :
      1. Charge les poids SAM3 depuis le Volume
      2. Démarre vLLM (GLM-OCR) en subprocess
      3. Attend que vLLM soit prêt

    Chaque appel à process_video() traite une vidéo complète.
    Modal instancie automatiquement N workers en parallèle si N vidéos sont lancées.
    """

    @modal.enter()
    def setup(self):
        """Initialisation du conteneur (appelé UNE SEULE FOIS)."""
        print("=" * 60)
        print("🚀 ALPR Worker — Initialisation")
        print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        # Vérifier les poids sur le Volume
        self.weights_path = f"{VOL_PATH}/sam3_weights.pt"
        if not os.path.exists(self.weights_path):
            raise RuntimeError(
                f"❌ Poids SAM3 introuvables: {self.weights_path}\n"
                "   → Upload avec: modal volume put alpr-weights sam3_weights.pt /sam3_weights.pt"
            )

        weights_size = os.path.getsize(self.weights_path) / (1024**3)
        print(f"✅ Poids SAM3 trouvés ({weights_size:.1f} Go)")

        # Exposer le chemin pour video_processor.py
        os.environ["WEIGHTS_PATH"] = self.weights_path

        # Démarrer vLLM
        self._start_vllm()

    def _start_vllm(self):
        """Démarre le serveur vLLM GLM-OCR en arrière-plan."""
        glm_model = os.environ.get("GLM_MODEL", "zai-org/GLM-OCR")
        gpu_util = os.environ.get("VLLM_GPU_UTILIZATION", "0.3")

        print(f"\n🚀 Démarrage GLM-OCR (vLLM)...")
        print(f"   Modèle: {glm_model}")
        print(f"   GPU utilization: {gpu_util}")

        env = os.environ.copy()
        cmd = [
            "python3", "-m", "vllm.entrypoints.openai.api_server",
            "--model", glm_model,
            "--port", str(VLLM_PORT),
            "--max-model-len", "2048",
            "--gpu-memory-utilization", gpu_util,
            "--allowed-local-media-path", "/",
            "--trust-remote-code",
            "--no-enable-log-requests",
        ]

        self._vllm_log = open("/tmp/vllm.log", "w")
        self._vllm_proc = subprocess.Popen(
            cmd, stdout=self._vllm_log, stderr=self._vllm_log, env=env,
        )
        print(f"   ⏳ PID {self._vllm_proc.pid}")

        # Attendre que vLLM soit prêt
        self._wait_for_vllm()

    def _wait_for_vllm(self, timeout=300):
        """Attend que vLLM réponde sur /v1/models."""
        import requests as _requests

        start = time.time()
        while time.time() - start < timeout:
            if self._vllm_proc.poll() is not None:
                raise RuntimeError(
                    f"❌ vLLM crashé (code {self._vllm_proc.returncode}). "
                    "Consulte /tmp/vllm.log"
                )
            try:
                resp = _requests.get(f"{VLLM_URL}/models", timeout=5)
                if resp.status_code == 200:
                    elapsed = int(time.time() - start)
                    print(f"\n   ✅ GLM-OCR prêt ({elapsed}s)")
                    return
            except Exception:
                pass

            elapsed = int(time.time() - start)
            print(f"   ⏳ Chargement modèle... {elapsed}s/{timeout}s", end="\r")
            time.sleep(10)

        raise RuntimeError(f"❌ vLLM timeout ({timeout}s)")

    @modal.method()
    def process_video(self, video_name: str, poste_name: str,
                      frame_skip: int = 3, time_window: int = 3) -> list:
        """
        Traite UNE vidéo à travers le pipeline complet :
        Nextcloud download → SAM3 → tri → homographie → OCR → filtres → dédup.
        Retourne la liste des plaques détectées (list de dicts).
        """
        print(f"\n{'='*60}")
        print(f"📹 {video_name} | Poste: {poste_name}")
        print(f"   frame_skip={frame_skip}  time_window={time_window}")
        print(f"{'='*60}")

        # Préparer les imports
        sys.path.insert(0, CODE_DIR)
        import batch_processor as bp

        # ── Monkey-patch des chemins ────────────────────────────
        bp.BASE_DIR = CODE_DIR
        bp.DATA_DIR = DATA_DIR
        bp.VIDEOS_BASE_DIR = VIDEOS_DIR
        bp.CROP_DIR = os.path.join(DATA_DIR, "crop")
        bp.CLEAN_DIR = os.path.join(DATA_DIR, "clean_plates")
        bp.CORRECTED_DIR = os.path.join(DATA_DIR, "corrected_plates")
        bp.WORK_VIDEO_DIR = os.path.join(DATA_DIR, "work_video")
        bp.IMAGES_DIR = os.path.join(DATA_DIR, "images")
        bp.RESULTS_DIR = RESULTS_DIR
        bp.VLLM_URL = VLLM_URL

        # ── Monkey-patch SAM3 : appel direct au lieu de Docker ──
        bp.run_sam3_extraction = self._run_sam3_direct

        # ── Préparer les répertoires ────────────────────────────
        for d in [bp.CROP_DIR, bp.CLEAN_DIR, bp.CORRECTED_DIR,
                  bp.WORK_VIDEO_DIR, bp.IMAGES_DIR, RESULTS_DIR, VIDEOS_DIR]:
            os.makedirs(d, exist_ok=True)

        # ── Télécharger cette vidéo ─────────────────────────────
        video_path = _download_single_video(video_name, VIDEOS_DIR, poste_name)
        if not video_path:
            return []

        # ── Pipeline complète ───────────────────────────────────
        mask_zones = bp.load_mask_config(VIDEOS_DIR)
        results = bp.process_single_video(
            video_path=video_path,
            frame_skip=frame_skip,
            time_window=time_window,
            glm_ready=True,
        )

        # ── Nettoyage ──────────────────────────────────────────
        if os.path.exists(video_path):
            os.remove(video_path)

        print(f"✅ {video_name}: {len(results)} plaques détectées")
        return results

    def _run_sam3_direct(self, frame_skip=5, mask_zones=None):
        """Exécution directe de SAM3 (remplace l'appel Docker)."""
        env = os.environ.copy()
        env["FRAME_SKIP"] = str(frame_skip)
        env["PYTHONPATH"] = "/app/sam3_repo"
        env["WEIGHTS_PATH"] = self.weights_path
        env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        
        # Passer les vrais chemins au script SAM3 (remplace les montages Docker)
        import batch_processor as bp
        env["INPUT_VIDEO_DIR"] = bp.WORK_VIDEO_DIR
        env["OUTPUT_IMAGES_DIR"] = bp.IMAGES_DIR
        env["OUTPUT_CROP_DIR"] = bp.CROP_DIR

        if mask_zones:
            env["MASK_ZONES"] = json.dumps(mask_zones)
        else:
            env.pop("MASK_ZONES", None)

        result = subprocess.run(
            ["python3", f"{CODE_DIR}/video_processor.py"],
            env=env,
        )

        if result.returncode != 0:
            print(f"   ❌ SAM3 erreur (code {result.returncode})")
            return False

        sys.path.insert(0, CODE_DIR)
        import batch_processor as bp
        crop_count = len([
            f for f in os.listdir(bp.CROP_DIR)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])
        print(f"   ✅ {crop_count} crops extraits")
        return crop_count > 0

    @modal.exit()
    def cleanup(self):
        """Arrêt propre de vLLM à la fin du conteneur."""
        if hasattr(self, '_vllm_proc') and self._vllm_proc.poll() is None:
            print("🛑 Arrêt de vLLM...")
            self._vllm_proc.terminate()
            try:
                self._vllm_proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._vllm_proc.kill()
            print("   ✅ vLLM arrêté")


# ============================================================
#  FONCTION : MERGER + EXPORTER + UPLOADER
# ============================================================

@app.function(
    image=alpr_image,
    secrets=[nextcloud_secret],
    timeout=300,
)
def merge_and_upload(all_results: list, poste_name: str) -> int:
    """
    Fusionne les résultats de toutes les vidéos, déduplique,
    exporte en CSV, et uploade sur Nextcloud.
    Tourne sur un container SANS GPU (juste du CPU).
    """
    sys.path.insert(0, CODE_DIR)
    import batch_processor as bp

    bp.RESULTS_DIR = RESULTS_DIR
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Flatten
    merged = [plate for batch in all_results for plate in batch]

    if not merged:
        print("⚠️ Aucune plaque détectée sur l'ensemble des vidéos")
        return 0

    # Déduplication globale inter-vidéos
    before = len(merged)
    merged = bp.remove_exact_duplicates(merged)
    if len(merged) < before:
        print(f"🔁 Doublons inter-vidéos supprimés: {before} → {len(merged)}")

    # Export CSV
    csv_file = bp.export_poste_csv(poste_name, merged)

    # Upload sur Nextcloud
    print(f"\n📤 Upload des résultats...")
    _upload_results([csv_file], poste_name)

    return len(merged)


# ============================================================
#  POINT D'ENTRÉE LOCAL (tourne sur ton PC)
# ============================================================

@app.local_entrypoint()
def main(
    poste: str = "CA1",
    frame_skip: int = 3,
    time_window: int = 3,
    max_videos: int = 0,
    video_list: str = "",
):
    """
    Orchestrateur principal. Tourne sur ton PC, dispatche le travail sur Modal.

    Args:
        poste:       Nom du poste (ex: CA1, CA3)
        frame_skip:  Frames à sauter par SAM3 (défaut: 3)
        time_window: Fenêtre de déduplication en sec (défaut: 3)
        max_videos:  Nombre max de vidéos (0 = toutes)
        video_list:  Liste de vidéos séparées par virgule (optionnel)

    Exemples:
        modal run modal_app.py --poste CA1
        modal run modal_app.py --poste CA1 --max-videos 2
        modal run modal_app.py --poste CA1 --video-list "vid1.avi,vid2.avi"
    """
    start_time = datetime.now()

    print("=" * 60)
    print("🏃 ALPR Pipeline — Modal")
    print(f"   Poste       : {poste}")
    print(f"   Frame skip  : {frame_skip}")
    print(f"   Time window : {time_window}s")
    print(f"   Démarrage   : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ── 1. Lister les vidéos sur Nextcloud ──────────────────────
    print("\n[1/4] 📋 Récupération de la liste des vidéos...")
    videos = fetch_video_list.remote(poste)

    if not videos:
        print("❌ Aucune vidéo trouvée sur Nextcloud")
        return

    # Filtrer
    if video_list:
        wanted = {v.strip() for v in video_list.split(",") if v.strip()}
        videos = [v for v in videos if v in wanted]

    if max_videos > 0:
        videos = videos[:max_videos]

    print(f"\n   📹 {len(videos)} vidéo(s) à traiter:")
    for v in videos:
        print(f"      • {v}")

    # ── 2. Traitement parallèle ─────────────────────────────────
    print(f"\n[2/4] 🚀 Lancement sur {len(videos)} GPU(s) L40S en parallèle...")

    worker = ALPRWorker()
    all_results = list(
        worker.process_video.starmap(
            [
                (video, poste, frame_skip, time_window)
                for video in videos
            ]
        )
    )

    # ── 3. Fusion + dédup + export + upload ─────────────────────
    print("\n[3/4] 📊 Fusion des résultats et upload...")
    total_plates = merge_and_upload.remote(all_results, poste)

    # ── 4. Résumé ──────────────────────────────────────────────
    elapsed = datetime.now() - start_time
    hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"\n{'='*60}")
    print(f"🎉 PIPELINE TERMINÉ")
    print(f"   Durée totale   : {hours:02d}h {minutes:02d}m {seconds:02d}s")
    print(f"   Vidéos traitées: {len(videos)}")
    print(f"   Plaques totales: {total_plates}")
    print(f"   GPU utilisés   : L40S × {len(videos)}")
    print(f"{'='*60}")
