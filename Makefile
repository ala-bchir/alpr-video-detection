# ============================================
# MAKEFILE - Labélisation des plaques avec Qwen VL
# ============================================
# Machine: 64GB RAM, RTX 5090 32GB VRAM
# ============================================

# Variables
IMAGE_NAME=sam3-video-processor
VLM_CONTAINER=qwen-labeler
PWD=$(shell pwd)

# Modèles disponibles (testés et validés)
QWEN_7B=Qwen/Qwen2.5-VL-7B-Instruct
QWEN_2B_OCR=prithivMLmods/Qwen2-VL-OCR-2B-Instruct
GLM_OCR=zai-org/GLM-OCR
ROLM_OCR=reducto/RolmOCR

.PHONY: build run all clean sort stop-vlm start-qwen-7b start-qwen-2b-ocr label-7b label-2b-ocr

# ============================================
# COMMANDES GÉNÉRALES
# ============================================

all: build run

build:
	docker build -t $(IMAGE_NAME) .

run:
	docker run --rm --gpus all \
		--ipc=host \
		--ulimit memlock=-1 \
		--ulimit stack=67108864 \
		-v $(PWD)/data/videos:/app/data/videos \
		-v $(PWD)/data/images:/app/data/images \
		-v $(PWD)/data/crop:/app/data/crop \
		-v $(PWD)/sam3_weights.pt:/app/sam3_weights.pt \
		-v $(PWD)/video_processor.py:/app/video_processor.py \
		$(IMAGE_NAME)

clean-outputs:
	sudo rm -rf /home/solayman/sam3/outputs/*.jpg

sort:
	docker run --rm \
		-v $(PWD)/data/crop:/app/data/crop \
		-v $(PWD)/data/clean_plates:/app/data/clean_plates \
		-v $(PWD)/data/rejected_plates:/app/data/rejected_plates \
		-v $(PWD)/auto_sort_plates.py:/app/auto_sort_plates.py \
		$(IMAGE_NAME) python /app/auto_sort_plates.py

# ============================================
# EXTRACTION DE VÉHICULES
# ============================================
extract-vehicles:
	docker run --rm --gpus all \
		--ipc=host \
		-v $(PWD)/data/videos:/app/data/videos \
		-v $(PWD)/data/vehicle_frames:/app/data/vehicle_frames \
		-v $(PWD)/sam3_weights.pt:/app/sam3_weights.pt \
		-v $(PWD)/extract_vehicles.py:/app/extract_vehicles.py \
		$(IMAGE_NAME) python3 /app/extract_vehicles.py

# ============================================
# QWEN VL - Vision Language Models pour OCR
# ============================================

# Arrêter le serveur VLM actif
stop-vlm:
	@echo "🛑 Arrêt du serveur VLM..."
	@docker stop $(VLM_CONTAINER) 2>/dev/null || true
	@docker rm $(VLM_CONTAINER) 2>/dev/null || true
	@echo "✅ Serveur arrêté"

# -------------------------------------------
# QWEN 2.5-VL-7B (Haute précision)
# VRAM: ~16GB | Vitesse: 145ms/image | Précision: 99.9%
# -------------------------------------------
start-qwen-7b:
	@echo "🚀 Démarrage de Qwen2.5-VL-7B..."
	@docker run -d \
		--gpus all \
		-p 8000:8000 \
		--shm-size=32g \
		--name $(VLM_CONTAINER) \
		vllm/vllm-openai:latest \
		--model $(QWEN_7B) \
		--max-model-len 2048 \
		--gpu-memory-utilization 0.9
	@echo "⏳ Chargement (~2-5 min)... Logs: docker logs -f $(VLM_CONTAINER)"

label-7b:
	@echo "📝 Labélisation avec Qwen2.5-VL-7B..."
	pip install openai --break-system-packages -q 2>/dev/null || pip install openai -q
	python3 label_gen_qwen.py --model qwen25-vl-7b

# -------------------------------------------
# QWEN 2-VL-2B-OCR (Rapide - Recommandé)
# VRAM: ~5GB | Vitesse: 91ms/image | Précision: 98.2%
# -------------------------------------------
start-qwen-2b-ocr:
	@echo "🚀 Démarrage de Qwen2-VL-2B-OCR..."
	@docker run -d \
		--gpus all \
		-p 8000:8000 \
		--shm-size=32g \
		--name $(VLM_CONTAINER) \
		vllm/vllm-openai:latest \
		--model $(QWEN_2B_OCR) \
		--max-model-len 2048 \
		--gpu-memory-utilization 0.9
	@echo "⏳ Chargement (~1-2 min)... Logs: docker logs -f $(VLM_CONTAINER)"

label-2b-ocr:
	@echo "📝 Labélisation avec Qwen2-VL-2B-OCR..."
	pip install openai --break-system-packages -q 2>/dev/null || pip install openai -q
	python3 label_gen_qwen.py --model qwen2-vl-2b-ocr

# ============================================
# DATASET - Ground Truth de test
# ============================================

generate-gt:
	@echo "📊 Génération du ground truth..."
	python3 generate_ground_truth.py

# ============================================
# ÉVALUATION DES MODÈLES
# ============================================

eval-7b:
	@echo "📊 Évaluation de Qwen2.5-VL-7B..."
	python3 evaluate_models.py --model qwen25-vl-7b

eval-2b-ocr:
	@echo "📊 Évaluation de Qwen2-VL-2B-OCR..."
	python3 evaluate_models.py --model qwen2-vl-2b-ocr

# -------------------------------------------
# GLM-OCR (0.9B - Ultra léger et spécialisé)
# VRAM: ~2GB | Vitesse: Très rapide
# Nécessite image custom avec Transformers nightly
# -------------------------------------------
build-glm-ocr:
	@echo "🔨 Construction de l'image GLM-OCR..."
	docker build -t vllm-glm-ocr -f Dockerfile.glm-ocr .
	@echo "✅ Image vllm-glm-ocr prête"

start-glm-ocr:
	@echo "🚀 Démarrage de GLM-OCR..."
	@docker run -d \
		--gpus all \
		-p 8000:8000 \
		--shm-size=32g \
		--name $(VLM_CONTAINER) \
		vllm-glm-ocr \
		--model $(GLM_OCR) \
		--max-model-len 2048 \
		--gpu-memory-utilization 0.9 \
		--allowed-local-media-path /
	@echo "⏳ Chargement (~1-2 min)... Logs: docker logs -f $(VLM_CONTAINER)"

eval-glm-ocr:
	@echo "📊 Évaluation de GLM-OCR..."
	python3 evaluate_models.py --model glm-ocr

# -------------------------------------------
# RolmOCR (7B - basé sur Qwen2.5-VL)
# VRAM: ~16GB | Optimisé pour OCR
# -------------------------------------------
start-rolm-ocr:
	@echo "🚀 Démarrage de RolmOCR..."
	@docker run -d \
		--gpus all \
		-p 8000:8000 \
		--shm-size=32g \
		--name $(VLM_CONTAINER) \
		vllm/vllm-openai:latest \
		--model $(ROLM_OCR) \
		--max-model-len 2048 \
		--gpu-memory-utilization 0.9
	@echo "⏳ Chargement (~2-5 min)... Logs: docker logs -f $(VLM_CONTAINER)"

eval-rolm-ocr:
	@echo "📊 Évaluation de RolmOCR..."
	python3 evaluate_models.py --model rolm-ocr

# -------------------------------------------
# Chandra (datalab-to/chandra)
# Utilise son propre conteneur Docker
# -------------------------------------------
start-chandra:
	@echo "🚀 Démarrage de Chandra..."
	@docker run -d \
		--gpus all \
		-p 8000:8000 \
		--shm-size=32g \
		--name $(VLM_CONTAINER) \
		vllm/vllm-openai:latest \
		--model datalab-to/chandra \
		--max-model-len 2048 \
		--gpu-memory-utilization 0.9
	@echo "⏳ Chargement... Logs: docker logs -f $(VLM_CONTAINER)"

eval-chandra:
	@echo "📊 Évaluation de Chandra..."
	python3 evaluate_models.py --model chandra

# -------------------------------------------
# LightOnOCR (1B - Léger et spécialisé OCR)
# VRAM: ~3GB | Vitesse: Rapide
# -------------------------------------------
start-lighton-ocr:
	@echo "🚀 Démarrage de LightOnOCR-2-1B..."
	@docker run -d \
		--gpus all \
		-p 8000:8000 \
		--shm-size=32g \
		--name $(VLM_CONTAINER) \
		vllm/vllm-openai:latest \
		--model lightonai/LightOnOCR-2-1B \
		--limit-mm-per-prompt '{"image": 1}' \
		--mm-processor-cache-gb 0 \
		--no-enable-prefix-caching
	@echo "⏳ Chargement (~1-2 min)... Logs: docker logs -f $(VLM_CONTAINER)"

eval-lighton-ocr:
	@echo "📊 Évaluation de LightOnOCR..."
	python3 evaluate_models.py --model lighton-ocr

# -------------------------------------------
# Dots.OCR (rednote-hilab)
# -------------------------------------------
start-dots-ocr:
	@echo "🚀 Démarrage de Dots.OCR..."
	@docker run -d \
		--gpus all \
		-p 8000:8000 \
		--shm-size=32g \
		--name $(VLM_CONTAINER) \
		vllm/vllm-openai:latest \
		--model rednote-hilab/dots.ocr \
		--max-model-len 2048 \
		--gpu-memory-utilization 0.9 \
		--trust-remote-code
	@echo "⏳ Chargement... Logs: docker logs -f $(VLM_CONTAINER)"

eval-dots-ocr:
	@echo "📊 Évaluation de Dots.OCR..."
	python3 evaluate_models.py --model dots-ocr

# -------------------------------------------
# PaddleOCR-VL (PaddlePaddle)
# Prompt spécial: "OCR:"
# -------------------------------------------
start-paddle-ocr:
	@echo "🚀 Démarrage de PaddleOCR-VL..."
	@docker run -d \
		--gpus all \
		-p 8000:8000 \
		--shm-size=32g \
		--name $(VLM_CONTAINER) \
		vllm/vllm-openai:latest \
		--model PaddlePaddle/PaddleOCR-VL \
		--trust-remote-code \
		--max-num-batched-tokens 16384 \
		--no-enable-prefix-caching \
		--mm-processor-cache-gb 0
	@echo "⏳ Chargement... Logs: docker logs -f $(VLM_CONTAINER)"

eval-paddle-ocr:
	@echo "📊 Évaluation de PaddleOCR-VL..."
	python3 evaluate_models.py --model paddle-ocr

# -------------------------------------------
# DeepSeek-OCR-2 (3B - nécessite vLLM nightly)
# -------------------------------------------
start-deepseek-ocr2:
	@echo "🚀 Démarrage de DeepSeek-OCR-2..."
	@docker run -d \
		--gpus all \
		-p 8000:8000 \
		--shm-size=32g \
		--name $(VLM_CONTAINER) \
		vllm/vllm-openai:nightly \
		--model deepseek-ai/DeepSeek-OCR-2 \
		--trust-remote-code \
		--max-model-len 4096 \
		--gpu-memory-utilization 0.9
	@echo "⏳ Chargement... Logs: docker logs -f $(VLM_CONTAINER)"

eval-deepseek-ocr2:
	@echo "📊 Évaluation de DeepSeek-OCR-2..."
	python3 evaluate_models.py --model deepseek-ocr2

# ============================================
# PIPELINE COMPLET
# ============================================
# Usage: make pipeline START_TIME=12:30 TIME_WINDOW=3 FRAME_SKIP=1
# La vidéo doit être placée dans data/videos/ (une seule vidéo)

VIDEOS_DIR=data/videos
TIME_WINDOW?=3
FRAME_SKIP?=1

pipeline:
	@echo "🎬 Lancement du pipeline complet..."
	@if [ -z "$(START_TIME)" ]; then \
		echo "❌ Erreur: START_TIME est requis"; \
		echo "Usage: make pipeline START_TIME=12:30 TIME_WINDOW=3 FRAME_SKIP=1"; \
		exit 1; \
	fi
	@VIDEO_FILE=$$(ls -1 $(VIDEOS_DIR)/*.mp4 $(VIDEOS_DIR)/*.avi $(VIDEOS_DIR)/*.mkv $(VIDEOS_DIR)/*.mov 2>/dev/null | head -1); \
	if [ -z "$$VIDEO_FILE" ]; then \
		echo "❌ Aucune vidéo trouvée dans $(VIDEOS_DIR)/"; \
		echo "📁 Formats supportés: .mp4, .avi, .mkv, .mov"; \
		exit 1; \
	fi; \
	echo "📹 Vidéo détectée: $$VIDEO_FILE"; \
	echo "⏰ Heure de départ: $(START_TIME)"; \
	echo "🔄 Fenêtre dédup: $(TIME_WINDOW)s"; \
	echo "⏭️  Frame skip: $(FRAME_SKIP)"; \
	pip install requests opencv-python --break-system-packages -q 2>/dev/null || pip install requests opencv-python -q; \
	python3 pipeline.py --video "$$VIDEO_FILE" --start-time "$(START_TIME)" --time-window $(TIME_WINDOW) --frame-skip $(FRAME_SKIP)

# ============================================
# NETTOYAGE
# ============================================
clean:
	docker rmi $(IMAGE_NAME)