import os
import json
import cv2
import torch
import gc
import numpy as np
from PIL import Image
from datetime import datetime
from tqdm import tqdm

# ==============================================================================
# 1. CONFIGURATION CUDA (Cohabitation avec vLLM sur RunPod)
# ==============================================================================
# cuDNN ne peut pas s'initialiser quand vLLM occupe une partie de la VRAM.
# On le désactive pour forcer PyTorch à utiliser ses convolutions natives.
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ==============================================================================
# 2. FIX DYNAMIQUE DU DEVICE (Annule le vieux patch Docker)
# ==============================================================================
_pos_enc_path = "/app/sam3_repo/sam3/model/position_encoding.py"
if os.path.exists(_pos_enc_path):
    with open(_pos_enc_path, 'r') as _f:
        _content = _f.read()
    if 'device="cpu"' in _content:
        _content = _content.replace('device="cpu"', 'device="cuda"')
        with open(_pos_enc_path, 'w') as _f:
            _f.write(_content)
        print("🔧 Fix appliqué : position_encoding.py (device='cpu' → 'cuda')")

# ==============================================================================
# 3. PATCH fused.py — Désactiver le cast forcé bf16 de addmm_act
# ==============================================================================
# Meta force bf16 dans addmm_act pour un kernel fusionné optimisé.
# En cohabitation vLLM (cuDNN off, float32), ça provoque un crash dtype.
# On remplace par un fallback standard linear + activation.
_fused_path = "/app/sam3_repo/sam3/perflib/fused.py"
if os.path.exists(_fused_path):
    with open(_fused_path, 'r') as _f:
        _fused_content = _f.read()
    if '.to(torch.bfloat16)' in _fused_content:
        _patched = '''# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
# pyre-unsafe
# PATCHED: removed forced bfloat16 conversion to support float32 inference

import torch

def addmm_act(activation, linear, mat1):
    """Fallback: standard matmul + activation (no forced bf16 cast)."""
    x = torch.nn.functional.linear(mat1, linear.weight, linear.bias)
    if activation in [torch.nn.functional.gelu, torch.nn.GELU]:
        return torch.nn.functional.gelu(x)
    if activation in [torch.nn.functional.relu, torch.nn.ReLU]:
        return torch.nn.functional.relu(x)
    raise ValueError(f"Unexpected activation {activation}")
'''
        with open(_fused_path, 'w') as _f:
            _f.write(_patched)
        print("🔧 Fix appliqué : fused.py (suppression cast bf16 forcé)")

# Imports SAM3 (APRÈS les patchs ci-dessus)
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ==============================================================================
# 4. CONSTANTES & CONFIGURATION ALPR
# ==============================================================================
INPUT_VIDEO_DIR = "/app/data/videos"
OUTPUT_IMAGES_DIR = "/app/data/images"
OUTPUT_CROP_DIR = "/app/data/crop"
WEIGHTS_PATH = "/app/sam3_weights.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_MAPPING = {
    "plaque": "license plate",
    "plaque2": "number plate",
}

VEHICLE_MAPPING = {
    "VL": "car",
    "PL": "Heavy-Duty Truck",
    "MOTO": "motorcycle",
    "BUS": "bus",
    "VUL": "light commercial vehicle",
}

CONFIDENCE_THRESHOLD = 0.4
MIN_BOX_AREA_THRESHOLD = 0
FRAME_SKIP = int(os.environ.get('FRAME_SKIP', '3'))

# ==============================================================================
# 5. FONCTIONS UTILITAIRES
# ==============================================================================

def load_mask_zones():
    mask_json = os.environ.get('MASK_ZONES', '')
    if not mask_json:
        return []
    try:
        zones = json.loads(mask_json)
        if zones:
            print(f"   🔲 {len(zones)} zone(s) de masquage chargée(s)")
        return zones
    except json.JSONDecodeError:
        print("   ⚠️ MASK_ZONES invalide, masquage désactivé")
        return []

MASK_ZONES = load_mask_zones()


def apply_mask(frame, mask_zones):
    for zone in mask_zones:
        x1 = max(0, int(zone.get('x1', 0)))
        y1 = max(0, int(zone.get('y1', 0)))
        x2 = min(frame.shape[1], int(zone.get('x2', 0)))
        y2 = min(frame.shape[0], int(zone.get('y2', 0)))
        frame[y1:y2, x1:x2] = 0
    return frame


# ==============================================================================
# 6. INITIALISATION DU MODÈLE SAM3
# ==============================================================================

def setup_model():
    print(f"🏗️ Chargement de SAM 3 sur {DEVICE}...")

    if torch.cuda.is_available():
        torch.cuda.init()
        torch.cuda.empty_cache()
        gc.collect()
        free_mem = torch.cuda.mem_get_info()[0] / (1024**3)
        total_mem = torch.cuda.mem_get_info()[1] / (1024**3)
        print(f"   📊 GPU: {free_mem:.1f} Go libres / {total_mem:.1f} Go total")

    print(f"   🔧 cuDNN: {torch.backends.cudnn.enabled}")

    model = build_sam3_image_model(
        checkpoint_path=WEIGHTS_PATH,
        device=DEVICE
    )
    model = model.float()
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"   ✅ Modèle SAM3 chargé ({n_params:.0f}M params)")

    return Sam3Processor(model)


# ==============================================================================
# 7. PIPELINE DE TRAITEMENT VIDÉO
# ==============================================================================

def process_videos():
    os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_CROP_DIR, exist_ok=True)

    video_files = sorted([
        f for f in os.listdir(INPUT_VIDEO_DIR)
        if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov'))
    ])
    if not video_files:
        print("⚠️ Aucune vidéo trouvée dans /videos")
        return

    processor = setup_model()

    for video_name in video_files:
        video_path = os.path.join(INPUT_VIDEO_DIR, video_name)
        cap = cv2.VideoCapture(video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_area = frame_width * frame_height

        frame_count = 0
        saved_count = 0
        error_count = 0

        print(f"\n🎬 Analyse de : {video_name}")
        pbar = tqdm(total=total_frames, desc="Progression", unit="frame")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            pbar.update(1)

            if frame_count % FRAME_SKIP == 0:
                if MASK_ZONES:
                    frame = apply_mask(frame, MASK_ZONES)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)

                try:
                    with torch.inference_mode():
                        inference_state = processor.set_image(pil_img)
                        best_detection = None

                        for ls_label, sam_prompt in LABEL_MAPPING.items():
                            output = processor.set_text_prompt(
                                state=inference_state, prompt=sam_prompt
                            )

                            boxes = output.get("boxes")
                            scores = output.get("scores")

                            if boxes is not None and len(boxes) > 0:
                                boxes_np = boxes.float().cpu().numpy()
                                scores_np = scores.float().cpu().numpy()

                                for box, score in zip(boxes_np, scores_np):
                                    score_val = float(score)
                                    if score_val < CONFIDENCE_THRESHOLD:
                                        continue

                                    x1, y1, x2, y2 = box
                                    area = (x2 - x1) * (y2 - y1)
                                    ratio = area / total_area

                                    if ratio < MIN_BOX_AREA_THRESHOLD:
                                        continue

                                    if best_detection is None or score_val > best_detection['score']:
                                        best_detection = {
                                            'label': ls_label,
                                            'score': score_val,
                                            'ratio': ratio,
                                            'box': box,
                                        }

                        if best_detection:
                            vehicle_category = "INCONNU"
                            best_vehicle = None
                            try:
                                for v_label, v_prompt in VEHICLE_MAPPING.items():
                                    v_output = processor.set_text_prompt(
                                        state=inference_state, prompt=v_prompt
                                    )
                                    v_boxes = v_output.get("boxes")
                                    v_scores = v_output.get("scores")

                                    if v_boxes is not None and len(v_boxes) > 0:
                                        v_scores_np = v_scores.float().cpu().numpy()
                                        max_idx = v_scores_np.argmax()
                                        v_score_val = float(v_scores_np[max_idx])

                                        if v_score_val >= CONFIDENCE_THRESHOLD:
                                            if best_vehicle is None or v_score_val > best_vehicle['score']:
                                                best_vehicle = {'label': v_label, 'score': v_score_val}
                            except Exception as e:
                                tqdm.write(f"⚠️ Erreur véhicule frame {frame_count}: {e}")

                            if best_vehicle:
                                vehicle_category = best_vehicle['label']

                            x1, y1, x2, y2 = map(int, best_detection['box'])
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(frame_width, x2), min(frame_height, y2)
                            plate_crop = frame[y1:y2, x1:x2]

                            timestamp = datetime.now().strftime("%H%M%S")
                            filename = (
                                f"{best_detection['label']}_f{frame_count}"
                                f"_{timestamp}_{vehicle_category}.jpg"
                            )

                            cv2.imwrite(
                                os.path.join(OUTPUT_IMAGES_DIR, filename), frame
                            )
                            cv2.imwrite(
                                os.path.join(OUTPUT_CROP_DIR, filename), plate_crop
                            )

                            detections_file = os.path.join(
                                OUTPUT_CROP_DIR, "detections.json"
                            )
                            detections = {}
                            if os.path.exists(detections_file):
                                try:
                                    with open(detections_file, 'r') as df:
                                        detections = json.load(df)
                                except (json.JSONDecodeError, IOError):
                                    pass
                            detections[filename] = vehicle_category
                            with open(detections_file, 'w') as df:
                                json.dump(detections, df, indent=2)

                            saved_count += 1
                            tqdm.write(
                                f"📸 Plaque : {best_detection['label']} "
                                f"(conf: {best_detection['score']:.1%}) | "
                                f"Véhicule: {vehicle_category}"
                            )

                except Exception as e:
                    error_count += 1
                    if error_count <= 3:
                        import traceback
                        tqdm.write(f"❌ Erreur frame {frame_count}: {e}")
                        tqdm.write(traceback.format_exc())
                    elif error_count == 4:
                        tqdm.write("❌ Erreurs suivantes supprimées...")

            frame_count += 1

            if frame_count % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        pbar.close()
        cap.release()
        if error_count > 0:
            print(f"⚠️ {error_count} erreurs sur {frame_count} frames")
        print(f"✅ Vidéo terminée : {saved_count} images extraites.")


if __name__ == "__main__":
    process_videos()
    print("Done")
    