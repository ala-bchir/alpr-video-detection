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
# 1. CONFIGURATION CUDA
# ==============================================================================
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    torch.cuda.empty_cache()

# [Les patchs de fix dynamiques (position_encoding et fused) restent identiques ici]

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ==============================================================================
# 4. CONSTANTES
# ==============================================================================
INPUT_VIDEO_DIR = os.environ.get("INPUT_VIDEO_DIR", "/app/data/videos")
OUTPUT_IMAGES_DIR = os.environ.get("OUTPUT_IMAGES_DIR", "/app/data/images")
OUTPUT_CROP_DIR = os.environ.get("OUTPUT_CROP_DIR", "/app/data/crop")
WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", "/app/sam3_weights.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_MAPPING = {"plaque": "license plate", "plaque2": "number plate"}
VEHICLE_MAPPING = {
    "VL": "car", "PL": "Heavy-Duty Truck", "MOTO": "motorcycle",
    "BUS": "bus", "VUL": "light commercial vehicle",
}

CONFIDENCE_THRESHOLD = 0.4
FRAME_SKIP = int(os.environ.get('FRAME_SKIP', '3'))

# ==============================================================================
# 5. FONCTIONS UTILITAIRES
# ==============================================================================

def load_mask_zones():
    mask_json = os.environ.get('MASK_ZONES', '')
    if not mask_json: return []
    try:
        return json.loads(mask_json)
    except: return []

MASK_ZONES = load_mask_zones()

def apply_mask(frame, mask_zones):
    for zone in mask_zones:
        x1, y1 = max(0, int(zone.get('x1', 0))), max(0, int(zone.get('y1', 0)))
        x2, y2 = min(frame.shape[1], int(zone.get('x2', 0))), min(frame.shape[0], int(zone.get('y2', 0)))
        frame[y1:y2, x1:x2] = 0
    return frame

def setup_model():
    print(f"🏗️ Chargement de SAM 3 sur {DEVICE}...")
    model = build_sam3_image_model(checkpoint_path=WEIGHTS_PATH, device=DEVICE)
    model = model.float().eval()
    return Sam3Processor(model)

# ==============================================================================
# 7. PIPELINE CORRIGÉ
# ==============================================================================

def process_videos():
    os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_CROP_DIR, exist_ok=True)

    video_files = sorted([f for f in os.listdir(INPUT_VIDEO_DIR) if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov'))])
    if not video_files: return

    processor = setup_model()
    detections_file = os.path.join(OUTPUT_CROP_DIR, "detections.json")
    
    # CHARGEMENT UNIQUE DU JSON
    all_detections = {}
    if os.path.exists(detections_file):
        try:
            with open(detections_file, 'r') as f: all_detections = json.load(f)
        except: pass

    for video_name in video_files:
        video_path = os.path.join(INPUT_VIDEO_DIR, video_name)
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_area = frame_width * frame_height

        frame_count = 0
        saved_count = 0

        print(f"\n🎬 Analyse : {video_name}")
        pbar = tqdm(total=total_frames, desc="Progression")

        while True: # Utilisation de while True pour contrôle total
            ret, frame = cap.read()
            if not ret or frame is None: # Double sécu sur la lecture
                break
            
            pbar.update(1)
            
            if frame_count % FRAME_SKIP == 0:
                if MASK_ZONES: frame = apply_mask(frame, MASK_ZONES)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)

                try:
                    # Changement dtype: float16 est plus stable que bfloat16 sur certaines configs mixtes
                    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                        inference_state = processor.set_image(pil_img)
                        best_det = None

                        # 1. Détection Plaque
                        for ls_label, sam_prompt in LABEL_MAPPING.items():
                            output = processor.set_text_prompt(state=inference_state, prompt=sam_prompt)
                            boxes, scores = output.get("boxes"), output.get("scores")

                            if boxes is not None and len(boxes) > 0:
                                b_np, s_np = boxes.float().cpu().numpy(), scores.float().cpu().numpy()
                                for b, s in zip(b_np, s_np):
                                    if s > CONFIDENCE_THRESHOLD:
                                        if best_det is None or s > best_det['score']:
                                            best_det = {'label': ls_label, 'score': float(s), 'box': b}

                        # 2. Détection Véhicule (si plaque trouvée)
                        if best_det:
                            vehicle_cat = "INCONNU"
                            best_v = None
                            for v_label, v_prompt in VEHICLE_MAPPING.items():
                                v_out = processor.set_text_prompt(state=inference_state, prompt=v_prompt)
                                v_sc = v_out.get("scores")
                                if v_sc is not None and len(v_sc) > 0:
                                    v_sc_np = v_sc.float().cpu().numpy()
                                    max_s = float(v_sc_np.max())
                                    if max_s >= CONFIDENCE_THRESHOLD:
                                        if best_v is None or max_s > best_v['score']:
                                            best_v = {'label': v_label, 'score': max_s}
                            
                            if best_v: vehicle_cat = best_v['label']

                            # Sauvegarde
                            x1, y1, x2, y2 = map(int, best_det['box'])
                            plate_crop = frame[max(0,y1):min(frame_height,y2), max(0,x1):min(frame_width,x2)]
                            
                            timestamp = datetime.now().strftime("%H%M%S")
                            filename = f"{best_det['label']}_f{frame_count}_{timestamp}_{vehicle_cat}.jpg"

                            cv2.imwrite(os.path.join(OUTPUT_IMAGES_DIR, filename), frame)
                            cv2.imwrite(os.path.join(OUTPUT_CROP_DIR, filename), plate_crop)
                            
                            # MISE À JOUR DU DICTIONNAIRE EN MÉMOIRE (Pas d'écriture disque ici)
                            all_detections[filename] = vehicle_cat
                            saved_count += 1

                        # NETTOYAGE MÉMOIRE IMMÉDIAT
                        del inference_state
                except Exception as e:
                    tqdm.write(f"❌ Erreur frame {frame_count}: {e}")

            frame_count += 1
            if frame_count % 30 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        cap.release()
        pbar.close()

    # SAUVEGARDE UNIQUE DU JSON À LA FIN DE TOUTES LES VIDÉOS
    with open(detections_file, 'w') as f:
        json.dump(all_detections, f, indent=2)
    print(f"✅ Terminé. {len(all_detections)} total detections sauvegardées.")

if __name__ == "__main__":
    process_videos()