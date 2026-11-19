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


# ============== DÉDUP NIVEAU POSTE (toutes vidéos / voies confondues) ==============

# Paires de caractères que l'OCR confond fréquemment (symétriques)
_CONFUSABLE_PAIRS = {
    frozenset(p) for p in [
        "O0", "Q0", "D0", "OQ", "DQ", "OD", "I1", "J1", "L1", "IJ", "S5", "B8", "G6",
        "T7", "A4", "Z2", "Z7", "CG", "56", "06", "68", "38", "98", "08", "69", "EF", "PR",
        "HM", "HN", "MN", "KX", "VY", "UV", "VW", "LI", "BR", "RK", "QG", "CQ",
    ]
}

_SIV_RE = re.compile(r'^[A-HJ-NP-TV-Z]{2}-?[0-9]{3}-?[A-HJ-NP-TV-Z]{2}$')
_FNI_RE = re.compile(r'^[0-9]{1,4}\s?[A-Z]{1,3}\s?[0-9]{2,3}$')


def _confusable_match(p1, p2, max_subs=2):
    """Même longueur, ≤ max_subs substitutions, toutes entre caractères confusables."""
    if len(p1) != len(p2):
        return False
    subs = [(a, b) for a, b in zip(p1, p2) if a != b]
    return len(subs) <= max_subs and all(frozenset((a, b)) in _CONFUSABLE_PAIRS for a, b in subs)


def are_near_duplicates(plate1, plate2, very_close=False):
    """
    Règle élargie pour deux lectures proches dans le temps (même passage, même
    voie ou voie voisine filmant le même véhicule) :
      1. are_duplicates()              — 1 édition (ou troncature pure)
      2. lecture partielle contenue    — 'FE635' ⊂ 'FE635VM' (≥ 4 caractères)
      3. ≤ 2 substitutions confusables — 'GC293SV' / 'CC283SV' (G↔C, 9↔8)
      4. very_close (lectures quasi simultanées) : ≤ 2 éditions quelconques
         — 'HG542XB' / 'NG542AB'
    """
    p1, p2 = _normalize(plate1), _normalize(plate2)
    if not p1 or not p2:
        return False
    if very_close and _edit_distance(p1, p2) <= 2:
        return True
    if are_duplicates(plate1, plate2):
        return True
    short, long_ = sorted((p1, p2), key=len)
    if len(short) >= 4 and len(short) < len(long_) and short in long_:
        return True
    return _confusable_match(p1, p2)


def _plate_format(r):
    fmt = r.get("format")
    if fmt in ("SIV", "FNI"):
        return fmt
    plate = r["plate"].upper()
    if "*" in plate:
        return "UNKNOWN"
    if _SIV_RE.match(plate):
        return "SIV"
    if _FNI_RE.match(plate):
        return "FNI"
    return "UNKNOWN"


def is_overlay_text(plate):
    """Texte incrusté par la caméra (ex. 'CAM1', 'CAM107SZ', '300CAM1') — pas une plaque."""
    p = _normalize(plate)
    return p.startswith("CAM") or "CAM1" in p


def _merge_near_passages(rows, time_window_s, close_window_s=3):
    """
    Une passe : regroupe (union-find) les lectures distantes de ≤ time_window_s s
    et similaires (are_near_duplicates, règle élargie à 2 éditions quelconques
    si ≤ close_window_s s). Retourne une lecture par groupe, à l'heure
    du premier passage. rows : liste de (datetime, dict) triée par temps.
    """
    from collections import Counter

    parent = list(range(len(rows)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            dt = (rows[j][0] - rows[i][0]).total_seconds()
            if dt > time_window_s:
                break
            if are_near_duplicates(rows[i][1]["plate"], rows[j][1]["plate"],
                                   very_close=dt <= close_window_s):
                parent[find(j)] = find(i)

    groups = {}
    for i in range(len(rows)):
        groups.setdefault(find(i), []).append(i)

    FORMAT_PRIORITY = {"SIV": 0, "FNI": 1, "UNKNOWN": 2}
    merged = []
    for idx in groups.values():
        counts = Counter(_normalize(rows[k][1]["plate"]) for k in idx)
        best_k = min(idx, key=lambda k: (
            FORMAT_PRIORITY[_plate_format(rows[k][1])],
            rows[k][1]["plate"].count("*"),
            -counts[_normalize(rows[k][1]["plate"])],
            k,
        ))
        first_t, first = rows[idx[0]]  # idx est trié par temps
        best = dict(rows[best_k][1])
        best["date"] = first["date"]
        best["timestamp"] = first["timestamp"]
        if best.get("category", "INCONNU") == "INCONNU":
            for k in idx:
                if rows[k][1].get("category", "INCONNU") != "INCONNU":
                    best["category"] = rows[k][1]["category"]
                    break
        merged.append((first_t, best))
    merged.sort(key=lambda x: x[0])
    return merged


def deduplicate_poste(results, time_window_s=10, close_window_s=3):
    """
    Déduplication finale d'un poste, toutes vidéos et voies confondues.

    0. Supprime le texte incrusté par la caméra ('CAM1…').
    1. Fusionne les variantes OCR d'un même passage (≤ time_window_s s,
       are_near_duplicates ; jusqu'à 2 caractères quelconques d'écart si
       ≤ close_window_s s) — y compris le même véhicule filmé par les caméras
       voie 1 et voie 2. Garde la meilleure lecture (SIV > FNI > autre, moins
       de '*', plus fréquente) à l'heure du premier passage. Répété jusqu'à
       stabilité (une fusion peut rapprocher deux groupes).
    2. Supprime les plaques strictement identiques sur tout le poste
       (une seule occurrence par plaque, la plus ancienne).

    results : dicts avec 'date' (YYYY-MM-DD), 'timestamp' (HH:MM:SS), 'plate',
              optionnels 'format', 'category'.
    """
    from datetime import datetime

    if not results:
        return []

    results = [r for r in results if not is_overlay_text(r["plate"])]

    def _ts(r):
        try:
            return datetime.strptime(f"{r['date']} {r['timestamp']}", "%Y-%m-%d %H:%M:%S")
        except (KeyError, ValueError):
            return None

    untimed = [r for r in results if _ts(r) is None]
    rows = sorted([(_ts(r), r) for r in results if _ts(r) is not None], key=lambda x: x[0])

    while True:
        merged = _merge_near_passages(rows, time_window_s, close_window_s)
        if len(merged) == len(rows):
            break
        rows = merged

    seen = set()
    unique = []
    for _, r in rows:
        key = _normalize(r["plate"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)

    return unique + untimed


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
