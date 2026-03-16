import os
import json
import cv2
import torch
import numpy as np
from PIL import Image
from datetime import datetime
from tqdm import tqdm

# SAM 3 - Imports officiels
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# --- CONFIGURATION ---
INPUT_VIDEO_DIR = "/app/data/videos"
OUTPUT_FRAMES_DIR = "/app/data/vehicle_frames"  # Photos entières contenant des véhicules
WEIGHTS_PATH = "/app/sam3_weights.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Catégories de véhicules à détecter (label CSV → prompt SAM3)
VEHICLE_MAPPING = {
    "VL": "car",
    "PL": "truck",
    "MOTO": "motorcycle",
    "BUS": "bus",
    "VAN": "van",
}

# --- PARAMÈTRES DE FILTRAGE ---
CONFIDENCE_THRESHOLD = 0.2  # Ajustable
MIN_BOX_AREA_THRESHOLD = 0.02   # Ignore les objets qui font moins de 2% de l'image (pour éviter les faux positifs lointains)
FRAME_SKIP = int(os.environ.get('FRAME_SKIP', '3'))

# --- MASK ZONES (chargé via env var) ---
def load_mask_zones():
    """Charge les zones de masquage depuis la variable d'env MASK_ZONES (JSON)."""
    mask_json = os.environ.get('MASK_ZONES', '')
    if not mask_json:
        return []
    try:
        zones = json.loads(mask_json)
        if zones:
            print(f"🎭 {len(zones)} zone(s) de masquage chargée(s)")
        return zones
    except json.JSONDecodeError:
        print("⚠️ MASK_ZONES invalide, masquage désactivé")
        return []

def apply_mask(frame, mask_zones):
    """Applique les zones de masquage (noir) sur le frame."""
    for zone in mask_zones:
        x1 = max(0, int(zone.get('x1', 0)))
        y1 = max(0, int(zone.get('y1', 0)))
        x2 = min(frame.shape[1], int(zone.get('x2', 0)))
        y2 = min(frame.shape[0], int(zone.get('y2', 0)))
        frame[y1:y2, x1:x2] = 0
    return frame

MASK_ZONES = load_mask_zones()

def setup_model():
    print(f"🏗️ Chargement de SAM 3 sur {DEVICE} (Mode Standard Float32)...")
    model = build_sam3_image_model(
        checkpoint_path=WEIGHTS_PATH,
        device=DEVICE
    )
    model.float() 
    model.eval()
    return Sam3Processor(model)

def process_videos():
    os.makedirs(OUTPUT_FRAMES_DIR, exist_ok=True)
    
    video_files = [f for f in os.listdir(INPUT_VIDEO_DIR) if f.endswith(('.mp4', '.avi', '.mkv', '.mov'))]
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
        
        print(f"\n🎬 Analyse de : {video_name}")
        pbar = tqdm(total=total_frames, desc="Progression", unit="frame")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            pbar.update(1)

            if frame_count % FRAME_SKIP == 0:
                # Appliquer le masque sur une copie pour le traitement (pour sauvegarder la frame originale propre)
                if MASK_ZONES:
                    frame_to_process = apply_mask(frame.copy(), MASK_ZONES)
                else:
                    frame_to_process = frame
                
                rgb_frame = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                
                try:
                    inference_state = processor.set_image(pil_img)
                    vehicle_detected = None

                    # Recherche des véhicules - on s'arrête dès qu'on en trouve un seul confiant (Optimisation de vitesse)
                    for v_label, sam_prompt in VEHICLE_MAPPING.items():
                        output = processor.set_text_prompt(state=inference_state, prompt=sam_prompt)
                        
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

                                if ratio >= MIN_BOX_AREA_THRESHOLD:
                                    vehicle_detected = {'label': v_label, 'score': score_val}
                                    break # On a trouvé un véhicule valide dans cette catégorie
                        
                        if vehicle_detected:
                            break # On arrête de tester les autres types de véhicules pour aller plus vite

                    if vehicle_detected:
                        timestamp = datetime.now().strftime("%H%M%S")
                        filename = f"vehicle_{vehicle_detected['label']}_f{frame_count}_{timestamp}.jpg"
                        
                        # Sauvegarder l'image entière ORIGINALE
                        cv2.imwrite(os.path.join(OUTPUT_FRAMES_DIR, filename), frame)
                        
                        # Enregistrer la détection
                        detections_file = os.path.join(OUTPUT_FRAMES_DIR, "vehicle_detections.json")
                        detections = {}
                        if os.path.exists(detections_file):
                            try:
                                with open(detections_file, 'r') as df:
                                    detections = json.load(df)
                            except (json.JSONDecodeError, IOError):
                                pass
                        
                        detections[filename] = {
                            "type": vehicle_detected['label'],
                            "confidence": vehicle_detected['score'],
                            "frame": frame_count
                        }
                        
                        with open(detections_file, 'w') as df:
                            json.dump(detections, df, indent=2)
                        
                        saved_count += 1
                        tqdm.write(f"📸 Frame capturée (Véhicule: {vehicle_detected['label']}, conf: {vehicle_detected['score']:.1%}) - Frame {frame_count}")
                
                except Exception as e:
                    tqdm.write(f"❌ Erreur sur la frame {frame_count}: {e}")

            frame_count += 1
            
        pbar.close()
        cap.release()
        print(f"✅ Vidéo terminée : {saved_count} frames extraites.")

if __name__ == "__main__":
    process_videos()
