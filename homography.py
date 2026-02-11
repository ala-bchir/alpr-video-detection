#!/usr/bin/env python3
"""
Script de correction de perspective (homographie) des plaques d'immatriculation.
Redresse les plaques inclinées pour améliorer l'OCR.
"""

import os
import cv2
import numpy as np
import shutil
from pathlib import Path

# ============== CONFIGURATION ==============
CLEAN_DIR = "data/clean_plates"
CORRECTED_DIR = "data/corrected_plates"
# ==========================================


def order_points(pts):
    """Ordonne les 4 points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    
    # top-left = plus petite somme, bottom-right = plus grande somme
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # top-right = plus petite différence, bottom-left = plus grande différence
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    
    return rect


def correct_perspective(img):
    """
    Détecte les contours de la plaque et applique une correction de perspective.
    Retourne l'image corrigée ou l'image originale si la correction échoue.
    """
    original = img.copy()
    h, w = img.shape[:2]
    
    # Convertir en niveaux de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Flou pour réduire le bruit
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Détection des bords avec Canny
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilater pour connecter les bords
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Trouver les contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return original
    
    # Trier par aire décroissante et prendre le plus grand
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for contour in contours[:5]:  # Tester les 5 plus grands contours
        # Approximer le contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        # Si on a un quadrilatère
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype("float32")
            rect = order_points(pts)
            
            # Vérifier que les points sont assez espacés (pas un artefact)
            area = cv2.contourArea(approx)
            if area < (h * w * 0.1):  # Au moins 10% de l'image
                continue
            
            # Calculer les dimensions cibles
            widthA = np.linalg.norm(rect[2] - rect[3])
            widthB = np.linalg.norm(rect[1] - rect[0])
            maxWidth = max(int(widthA), int(widthB))
            
            heightA = np.linalg.norm(rect[1] - rect[2])
            heightB = np.linalg.norm(rect[0] - rect[3])
            maxHeight = max(int(heightA), int(heightB))
            
            if maxWidth < 20 or maxHeight < 10:
                continue
            
            # Points de destination (rectangle droit)
            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]
            ], dtype="float32")
            
            # Appliquer la transformation
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
            
            return warped
    
    # Fallback: correction par rotation via minAreaRect
    # Utile quand le contour n'est pas un quadrilatère parfait
    if contours:
        largest = contours[0]
        rect = cv2.minAreaRect(largest)
        angle = rect[2]
        
        # Ajuster l'angle
        if angle < -45:
            angle += 90
        elif angle > 45:
            angle -= 90
        
        # Si l'angle est significatif (> 2°), corriger
        if abs(angle) > 2:
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h), 
                                      flags=cv2.INTER_CUBIC, 
                                      borderMode=cv2.BORDER_REPLICATE)
            return rotated
    
    return original


def process_directory(input_dir, output_dir):
    """Traite toutes les images d'un dossier."""
    # Nettoyer le dossier de sortie
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    input_path = Path(input_dir)
    images = sorted([f for f in input_path.iterdir() 
                    if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    
    if not images:
        print(f"❌ Aucune image trouvée dans {input_dir}")
        return 0
    
    print(f"🔍 Correction de perspective sur {len(images)} images...")
    
    corrected_count = 0
    kept_count = 0
    
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        result = correct_perspective(img)
        
        # Vérifier si l'image a été modifiée
        if result.shape != img.shape or not np.array_equal(result, img):
            corrected_count += 1
        else:
            kept_count += 1
        
        cv2.imwrite(str(Path(output_dir) / img_path.name), result)
    
    print(f"✅ {corrected_count} plaques redressées, {kept_count} inchangées")
    return len(images)


if __name__ == "__main__":
    print("=" * 50)
    print("📐 CORRECTION DE PERSPECTIVE DES PLAQUES")
    print("=" * 50)
    process_directory(CLEAN_DIR, CORRECTED_DIR)
