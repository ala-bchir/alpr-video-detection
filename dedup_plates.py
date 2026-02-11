#!/usr/bin/env python3
"""
Script de déduplication intelligente des plaques d'immatriculation.
Regroupe les plaques similaires dans une fenêtre temporelle configurable.
"""

from collections import Counter


def has_common_substring(plate1, plate2, min_length=3):
    """Vérifie si deux plaques partagent une sous-chaîne de min_length caractères consécutifs."""
    for i in range(len(plate1) - min_length + 1):
        sub = plate1[i:i + min_length]
        if '*' in sub:
            continue
        if sub in plate2:
            return True
    return False


def shared_chars_count(plate1, plate2):
    """Compte le nombre de caractères individuels en commun (indépendant de la position)."""
    c1 = Counter(c for c in plate1 if c not in ('*', '-', ' '))
    c2 = Counter(c for c in plate2 if c not in ('*', '-', ' '))
    return sum((c1 & c2).values())


def are_duplicates(plate1, plate2, min_common=3):
    """
    Vérifie si deux plaques sont des doublons.
    Critère : 3+ caractères consécutifs en commun OU 3+ caractères individuels en commun.
    """
    # Nettoyer pour la comparaison (enlever tirets et espaces)
    p1 = plate1.replace('-', '').replace(' ', '')
    p2 = plate2.replace('-', '').replace(' ', '')
    return has_common_substring(p1, p2) or shared_chars_count(p1, p2) >= min_common


def count_stars(plate):
    """Compte le nombre d'étoiles dans une plaque."""
    return plate.count('*')


def pick_best_plate(group):
    """
    Choisit la meilleure plaque d'un groupe de doublons.
    Priorité : SIV > FNI > UNKNOWN, puis moins d'étoiles.
    """
    FORMAT_PRIORITY = {"SIV": 0, "FNI": 1, "UNKNOWN": 2}
    
    group.sort(key=lambda r: (
        FORMAT_PRIORITY.get(r.get("format", "UNKNOWN"), 2),
        count_stars(r["plate"])
    ))
    return group[0]


def deduplicate(results, fps, time_window=3):
    """
    Déduplication intelligente basée sur le temps et la similarité.
    
    Args:
        results: liste de dicts avec 'plate', 'frame', 'format', 'file'
        fps: frames par seconde de la vidéo
        time_window: fenêtre temporelle en secondes (configurable)
    
    Returns:
        liste dédupliquée avec les meilleures plaques
    """
    if not results:
        return []
    
    FRAME_WINDOW = int(fps * time_window)
    
    # Trier par frame (ordre chronologique)
    results.sort(key=lambda x: x["frame"])
    
    used = set()
    deduplicated = []
    
    for i, r in enumerate(results):
        if i in used:
            continue
        
        group = [r]
        used.add(i)
        
        for j in range(i + 1, len(results)):
            if j in used:
                continue
            
            frame_diff = abs(results[j]["frame"] - r["frame"])
            if frame_diff > FRAME_WINDOW:
                break  # Liste triée, les suivants sont encore plus loin
            
            # Vérifier similarité
            if are_duplicates(r["plate"], results[j]["plate"]):
                group.append(results[j])
                used.add(j)
        
        best = pick_best_plate(group)
        deduplicated.append(best)
    
    return deduplicated


if __name__ == "__main__":
    # Test rapide
    test_data = [
        {"plate": "EA-335-MN", "frame": 100, "format": "SIV", "file": "a.jpg"},
        {"plate": "FA-335-ML", "frame": 125, "format": "SIV", "file": "b.jpg"},
        {"plate": "BX-683-MR", "frame": 500, "format": "SIV", "file": "c.jpg"},
    ]
    
    result = deduplicate(test_data, fps=25, time_window=3)
    print(f"Avant: {len(test_data)} → Après: {len(result)}")
    for r in result:
        print(f"  {r['plate']} [{r['format']}]")
