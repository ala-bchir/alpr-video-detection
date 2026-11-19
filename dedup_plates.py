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

        best = dict(best)
        # Toutes les lectures OCR du groupe, pour le vote final au niveau du poste
        best["readings"] = [r for g in group for r in g.get("readings", [g["plate"]])]
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


_SIV_SHAPE_RE = re.compile(r'^([A-Z]{2})(\d{3})([A-Z]{2})$')


def _siv_parts(plate):
    """('AB', '123', 'CD') pour une lecture complète de forme SIV, sinon None."""
    if "*" in plate:
        return None
    m = _SIV_SHAPE_RE.match(_normalize(plate))
    return m.groups() if m else None


def _same_vehicle_digits(plate1, plate2, max_letter_diff=0):
    """
    Même bloc de 3 chiffres ET au moins un groupe de lettres identique à
    max_letter_diff lettre près : 'EG-301-HA' / 'SQ-301-HA', 'EQ-237-MA' / 'FQ-237-JV'.
    """
    a, b = _siv_parts(plate1), _siv_parts(plate2)
    if not a or not b or a[1] != b[1]:
        return False
    return (_edit_distance(a[0], b[0]) <= max_letter_diff) or \
           (_edit_distance(a[2], b[2]) <= max_letter_diff)


def _partial_fuzzy_match(plate1, plate2):
    """
    Lecture partielle ('*', ≥ 5 caractères lus) qui correspond à une portion
    contiguë d'une plaque complète avec au plus 1 caractère différent :
    'OT373**' / 'DT-373-AD', '451VO**' / 'FR-451-VQ'.
    """
    if ("*" in plate1) == ("*" in plate2):
        return False
    partial, full = (plate1, plate2) if "*" in plate1 else (plate2, plate1)
    p, f = _normalize(partial), _normalize(full)
    if len(p) < 5 or len(p) >= len(f):
        return False
    for k in range(len(f) - len(p) + 1):
        if sum(x != y for x, y in zip(p, f[k:k + len(p)])) <= 1:
            return True
    return False


def _longest_common_run(a, b):
    best = 0
    for i in range(len(a)):
        for j in range(len(b)):
            k = 0
            while i + k < len(a) and j + k < len(b) and a[i + k] == b[j + k]:
                k += 1
            best = max(best, k)
    return best


def _weak_reading_overlap(plate1, plate2, min_run=3):
    """
    Une lecture « faible » (partielle '*' ou hors format SIV/FNI) et une SIV
    complète partageant ≥ min_run caractères consécutifs : 'EY2005*' / 'EY-200-SK',
    '361XJH*' / 'DT-361-XJ', 'BU241Y*' / 'BJ-234-TY' (non : 2 seulement).
    """
    def weak(p):
        return "*" in p or not (_SIV_RE.match(p.upper()) or _FNI_RE.match(p.upper()))
    if weak(plate1) == weak(plate2):
        return False
    strong = plate2 if weak(plate1) else plate1
    if not _SIV_RE.match(strong.upper()):
        return False
    return _longest_common_run(_normalize(plate1), _normalize(plate2)) >= min_run


# Fenêtres (secondes) des règles de fusion d'un même passage
CLOSE_WINDOW_S = 3      # ≤ 2 caractères quelconques d'écart
PARTIAL_WINDOW_S = 5    # partielle ≈ portion d'une plaque complète (1 caractère d'écart)
NEAR_WINDOW_S = 10      # are_near_duplicates
PARTIAL_RUN_WINDOW_S = 2   # lecture faible (partielle / hors format) ≥ 3 car. consécutifs
                            # communs avec une SIV complète
PARKED_WINDOW_S = 60    # mêmes chiffres + un groupe de lettres IDENTIQUE : chaîne sans
                        # trou > 60 s (véhicule lent, arrêté ou garé relu pendant des minutes).
                        # NB : « à 1 lettre près » fusionnait des véhicules différents en
                        # trafic dense (EP-700-AA / EV-700-ZZ sur E5) → abandonné.


def same_passage(plate1, plate2, dt):
    """Deux lectures séparées de dt secondes appartiennent-elles au même passage ?"""
    if dt <= NEAR_WINDOW_S and are_near_duplicates(plate1, plate2, very_close=dt <= CLOSE_WINDOW_S):
        return True
    if dt <= PARTIAL_WINDOW_S and _partial_fuzzy_match(plate1, plate2):
        return True
    if dt <= PARTIAL_RUN_WINDOW_S and _weak_reading_overlap(plate1, plate2):
        return True
    return dt <= PARKED_WINDOW_S and _same_vehicle_digits(plate1, plate2, max_letter_diff=0)


def _vote_plate(readings):
    """
    Vote caractère par caractère entre les lectures complètes de forme SIV d'un
    groupe : chaque lecture se trompe à un endroit différent, la majorité par
    position corrige ces erreurs isolées ('FF280LL', 'FF260AA', 'FF263AA',
    'FE260AA' → 'FF-260-AA'). Égalité sur une position → caractère de la lecture
    la plus cohérente avec l'ensemble du groupe. Retourne None si moins de
    2 lectures SIV ou si le résultat n'est pas une SIV valide.
    """
    from collections import Counter
    sivs = ["".join(p) for p in (_siv_parts(r) for r in readings) if p]
    if len(sivs) < 2:
        return None
    pos_counts = [Counter(s[i] for s in sivs) for i in range(7)]
    fb = max(sivs, key=lambda s: sum(pos_counts[i][c] for i, c in enumerate(s)))
    voted = ""
    for i in range(7):
        counts = Counter(s[i] for s in sivs)
        top = max(counts.values())
        voted += fb[i] if counts.get(fb[i]) == top else counts.most_common(1)[0][0]
    plate = f"{voted[:2]}-{voted[2:5]}-{voted[5:]}"
    return plate if _SIV_RE.match(plate) else None


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


_OVERLAY_RE = re.compile(r'[CG]V?A[MN][1I]|CAM|EURO|ZERO')


def is_overlay_text(plate):
    """
    Texte incrusté par la caméra ou parasite, pas une plaque :
      'CAM1…' et ses lectures déformées ('GAM1000', 'CVAM1', 'CAN1', 'J00CAMI'),
      'CAM1' tronqué en tête d'une lecture complète ('AM1562VV', 'M1E606ZN'),
      'EURO00', 'ZERO00'.
    Une lecture au format SIV ou FNI valide n'est jamais considérée comme une
    incrustation (ex. 'AM-123-BC' est une vraie plaque).
    """
    raw = plate.upper().strip()
    p = _normalize(plate)
    if _SIV_RE.match(raw) or _SIV_RE.match(p) or _FNI_RE.match(raw):
        return False
    if _OVERLAY_RE.search(p):
        return True
    # 'CAM1' tronqué : seulement sur une lecture complète (une partielle 'AM166**'
    # peut être le début d'une vraie plaque AM-166-…)
    return "*" not in raw and len(p) >= 7 and (p.startswith("AM1") or p.startswith("M1"))


def _merge_near_passages(rows):
    """
    Une passe : regroupe (union-find) les lectures d'un même passage
    (same_passage) et retourne une lecture par groupe — plaque choisie par vote
    caractère par caractère — à l'heure du premier passage.
    rows : liste de (datetime, dict) triée par temps.
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
            if dt > PARKED_WINDOW_S:
                break
            if same_passage(rows[i][1]["plate"], rows[j][1]["plate"], dt):
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
        all_readings = [x for k in idx for x in rows[k][1].get("readings", [rows[k][1]["plate"]])]
        voted = _vote_plate(all_readings)
        if voted:
            best["plate"] = voted
            best["format"] = "SIV"
        best["readings"] = all_readings
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


def deduplicate_poste(results):
    """
    Déduplication finale d'un poste, toutes vidéos et voies confondues.

    0. Supprime le texte incrusté par la caméra ('CAM1…', is_overlay_text).
    1. Fusionne les lectures d'un même passage (same_passage) — y compris le
       même véhicule filmé par les caméras voie 1 et voie 2 :
         ≤ 3 s  : ≤ 2 caractères quelconques d'écart
         ≤ 5 s  : partielle ≈ portion d'une plaque complète (1 caractère d'écart)
         ≤ 10 s : 1 édition, partielle contenue, ≤ 2 substitutions confusables
         ≤ 60 s entre deux lectures : mêmes 3 chiffres + un groupe de lettres
                  à 1 lettre près (véhicule lent / arrêté / garé)
       Plaque gardée : vote caractère par caractère entre les lectures SIV du
       groupe (sinon SIV > FNI > autre, moins de '*', plus fréquente), à l'heure
       du premier passage. Répété jusqu'à stabilité.
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
        merged = _merge_near_passages(rows)
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
