import os
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
OUTPUT_IMAGES_DIR = "/app/data/images"  # Photos entières
OUTPUT_CROP_DIR = "/app/data/crop"      # Crops des plaques
WEIGHTS_PATH = "/app/sam3_weights.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_MAPPING = {
    "plaque": "license plate",
    "plaque2": "number plate",
}

# --- PARAMÈTRES DE FILTRAGE ---
CONFIDENCE_THRESHOLD = 0.4  # Plus permissif pour capter plus de plaques
MIN_BOX_AREA_THRESHOLD = 0   # Désactivé - on garde toutes les tailles
FRAME_SKIP = 3

def setup_model():
    print(f"🏗️ Chargement de SAM 3 sur {DEVICE} (Mode Standard Float32)...")
    
    # On force le chargement en Float32 pur (le format le plus compatible)
    model = build_sam3_image_model(
        checkpoint_path=WEIGHTS_PATH,
        device=DEVICE
    )
    model.float() # Force tous les poids en Float32
    model.eval()
    
    return Sam3Processor(model)

def process_videos():
    os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_CROP_DIR, exist_ok=True)
    
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
                # Préparation de l'image
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                
                try:
                    # Inférence sans aucun changement de type (Float32 partout)
                    inference_state = processor.set_image(pil_img)
                    best_detection = None

                    for ls_label, sam_prompt in LABEL_MAPPING.items():
                        output = processor.set_text_prompt(state=inference_state, prompt=sam_prompt)
                        
                        boxes = output.get("boxes")
                        scores = output.get("scores")

                        if boxes is not None and len(boxes) > 0:
                            # Calculs simples sur CPU
                            boxes_np = boxes.float().cpu().numpy()
                            scores_np = scores.float().cpu().numpy()

                            for box, score in zip(boxes_np, scores_np):
                                score_val = float(score)
                                if score_val < CONFIDENCE_THRESHOLD:
                                    continue

                                x1, y1, x2, y2 = box
                                area = (x2 - x1) * (y2 - y1)
                                ratio = area / total_area

                                # Filtre : on ignore si l'objet est trop petit (< 5%)
                                if ratio < MIN_BOX_AREA_THRESHOLD:
                                    continue

                                if best_detection is None or score_val > best_detection['score']:
                                    best_detection = {'label': ls_label, 'score': score_val, 'ratio': ratio, 'box': box}

                    if best_detection:
                        # Crop de la zone de la plaque détectée
                        x1, y1, x2, y2 = map(int, best_detection['box'])
                        # S'assurer que les coordonnées sont dans les limites de l'image
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame_width, x2), min(frame_height, y2)
                        plate_crop = frame[y1:y2, x1:x2]
                        
                        timestamp = datetime.now().strftime("%H%M%S")
                        filename = f"{best_detection['label']}_f{frame_count}_{timestamp}.jpg"
                        
                        # Sauvegarder l'image entière dans data/images
                        cv2.imwrite(os.path.join(OUTPUT_IMAGES_DIR, filename), frame)
                        
                        # Sauvegarder le crop de la plaque dans data/crop
                        cv2.imwrite(os.path.join(OUTPUT_CROP_DIR, filename), plate_crop)
                        
                        saved_count += 1
                        tqdm.write(f"📸 Plaque capturée : {best_detection['label']} (conf: {best_detection['score']:.1%})")
                
                except Exception as e:
                    tqdm.write(f"❌ Erreur sur la frame {frame_count}: {e}")

            frame_count += 1
            
        pbar.close()
        cap.release()
        print(f"✅ Vidéo terminée : {saved_count} images extraites.")

if __name__ == "__main__":
    process_videos()