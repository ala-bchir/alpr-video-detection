# Utilisation de la version d'octobre 2025 (plus stable pour Blackwell)
FROM nvcr.io/nvidia/pytorch:25.10-py3

WORKDIR /app

# 1. Dépendances système
RUN apt-get update && apt-get install -y \
    git libgl1 libglib2.0-0 libgomp1 ffmpeg sed \
    && rm -rf /var/lib/apt/lists/*

# 2. On prépare le terrain pour OpenCV
# On désinstalle TOUTE trace d'OpenCV qui pourrait venir de l'image de base ou des requirements
COPY requirements.txt .
RUN pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless

# 3. Installation des requirements SANS OpenCV d'abord
RUN pip install --no-cache-dir $(grep -ivE "torch|torchvision|torchaudio|opencv" requirements.txt)

# 4. Installation de la version SPECIFIQUE d'OpenCV qui règle le bug DictValue
RUN pip install --no-cache-dir opencv-python-headless==4.8.0.74

# 5. Installation SAM 3
RUN git clone https://github.com/facebookresearch/sam3.git /app/sam3_repo
RUN pip install -e /app/sam3_repo
RUN sed -i 's/device="cuda"/device="cpu"/g' /app/sam3_repo/sam3/model/position_encoding.py

# 6. Fichiers
RUN mkdir -p /app/videos /app/data/images /app/data/crop
COPY video_processor.py .
COPY sam3_weights.pt .

ENV PYTHONPATH="/app/sam3_repo"

CMD ["python", "video_processor.py"]