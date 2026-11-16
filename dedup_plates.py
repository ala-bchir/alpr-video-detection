#!/usr/bin/env python3
"""
Déduplication intelligente des plaques par distance d'édition (Levenshtein).

Principe : deux plaques sont considérées comme doublons si elles sont
proches dans le temps ET si leur distance de Levenshtein est ≤ seuil.
  - Plaque ≤ 6 chars normalisés → 1 édition tolérée
  - Plaque ≥ 7 chars normalisés → 2 éditions tolérées

La comparaison se fait contre la MEILLEURE plaque du groupe courant
(pas contre l'ancre initiale) pour ne pas rater les variantes en chaîne.
"""

import re


def _normalize(plate):
    """Supprime tirets, espaces et étoiles pour la comparaison."""
    return re.sub(r'[\s\-*]', '', plate.upper())


def _edit_distance(s1, s2):
    """Distance de Levenshtein entre deux chaînes (O(m*n) mais m,n ≤ 9)."""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if s1[i - 1] == s2[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def are_duplicates(plate1, plate2):
    """
    True si les deux plaques sont probablement le même véhicule (erreur OCR).

    Règle : max_edits = max(1, len_diff)
      - Même longueur           → 1 substitution tolérée (ex. O↔0, I↔1)
      - Différence de 1 char    → 1 insertion/suppression (ex. '123AB' → '123AB5')
      - Différence de 2 chars   → 2 insertions/suppressions pures, zéro substitution
                                  (ex. 'DJ359' → 'DJ359RL' — plaque tronquée)
      - Différence > 2          → rejet (plaques trop différentes)

    Cela empêche les faux positifs comme 'AB-123-CD' vs 'AC-124-CD' (dist=2, rejeté).
    """
    p1, p2 = _normalize(plate1), _normalize(plate2)
    if not p1 or not p2:
        return False
    len_diff = abs(len(p1) - len(p2))
    if len_diff > 2:
        return False
    max_edits = max(1, len_diff)
    return _edit_distance(p1, p2) <= max_edits


def _pick_best(group):
    """
    Choisit la plaque la plus fiable dans un groupe de doublons.
    Priorité : SIV > FNI > UNKNOWN, puis minimum d'étoiles.
    """
    if not group:
        return None
    FORMAT_PRIORITY = {"SIV": 0, "FNI": 1, "UNKNOWN": 2}
    return min(group, key=lambda r: (
        FORMAT_PRIORITY.get(r.get("format", "UNKNOWN"), 2),
        r["plate"].count("*")
    ))


def deduplicate(results, fps, time_window=3):
    """
    Déduplication par fenêtre temporelle + distance d'édition.

    Args:
        results   : liste de dicts avec 'plate', 'frame', (optionnel 'format')
        fps       : images par seconde de la vidéo source
        time_window: fenêtre en secondes dans laquelle deux lectures du même
                     véhicule peuvent apparaître (défaut 3 s)

    Returns:
        liste dédupliquée, une entrée par véhicule distinct.
    """
    if not results:
        return []

    frame_window = int(fps * time_window)
    results = sorted(results, key=lambda r: r["frame"])

    used = set()
    deduplicated = []

    for i, anchor in enumerate(results):
        if i in used:
            continue

        group = [anchor]
        used.add(i)
        best = anchor        # meilleure plaque du groupe
        latest_frame = anchor["frame"]  # frame la plus récente du groupe

        for j in range(i + 1, len(results)):
            if j in used:
                continue
            # Double garde :
            #   1. plafond absolu depuis l'ancre  → empêche la fusion infinie en chaîne
            #   2. fenêtre glissante depuis latest → permet la fusion de variantes consécutives
            if results[j]["frame"] - anchor["frame"] > frame_window * 3:
                break
            if results[j]["frame"] - latest_frame > frame_window:
                continue

            if are_duplicates(best["plate"], results[j]["plate"]):
                group.append(results[j])
                used.add(j)
                latest_frame = results[j]["frame"]
                candidate = _pick_best([best, results[j]])
                if candidate is not best:
                    best = candidate

        deduplicated.append(best)

    return deduplicated


if __name__ == "__main__":
    test_data = [
        {"plate": "HN-862-DE", "frame": 625, "format": "SIV"},
        {"plate": "HN-862-QS", "frame": 630, "format": "SIV"},
        {"plate": "HN-062-DE", "frame": 635, "format": "SIV"},
        {"plate": "HN-862-JE", "frame": 640, "format": "SIV"},
        {"plate": "HN-662-DE", "frame": 645, "format": "SIV"},
        {"plate": "HN-862-QE", "frame": 650, "format": "SIV"},
        {"plate": "BX-683-MR", "frame": 1200, "format": "SIV"},  # plaque différente
        {"plate": "AB-123-CD", "frame": 1210, "format": "SIV"},  # plaque différente
    ]

    result = deduplicate(test_data, fps=25, time_window=3)
    print(f"Avant: {len(test_data)} → Après: {len(result)}")
    for r in result:
        print(f"  {r['plate']} [{r.get('format','?')}]")
