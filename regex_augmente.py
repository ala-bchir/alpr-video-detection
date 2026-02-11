"""
Module de reconnaissance et correction de plaques d'immatriculation multi-format.

Formats supportés (par ordre de priorité) :
1. SIV  (France, depuis 2009) : AA-123-AA
2. FNI  (France, avant 2009)  : 1234 AB 57
3. UK   (Royaume-Uni)         : AA12 AAA
4. CH   (Suisse)              : AA 123456  (2 lettres + 1 à 6 chiffres)
"""

import re
from dataclasses import dataclass
from typing import Literal


# ═══════════════════════════════════════════════════════════════════════════
# TYPES
# ═══════════════════════════════════════════════════════════════════════════

PlateFormat = Literal["SIV", "FNI", "UK", "CH", "UNKNOWN"]


# ═══════════════════════════════════════════════════════════════════════════
# REGEX PATTERNS
# ═══════════════════════════════════════════════════════════════════════════

# SIV français (2009+) : AA-123-AA — lettres sauf I, O, U
SIV_VALID = "ABCDEFGHJKLMNPQRSTVWXYZ"
SIV_REGEX = re.compile(
    rf'^([{SIV_VALID}]{{2}})'
    r'-?'
    r'(\d{3})'
    r'-?'
    rf'([{SIV_VALID}]{{2}})$'
)

# FNI français (avant 2009) : 1234 AB 57 (ou variantes sans espaces)
# 1 à 4 chiffres + 1 à 3 lettres + 2 chiffres (département)
FNI_REGEX = re.compile(
    r'^(\d{1,4})'
    r'[\s\-]?'
    r'([A-Z]{1,3})'
    r'[\s\-]?'
    r'(\d{2})$'
)

# UK : AA12 AAA (2 lettres région + 2 chiffres âge + 3 lettres aléatoires)
UK_REGEX = re.compile(
    r'^([A-Z]{2})'
    r'[\s\-]?'
    r'(\d{2})'
    r'[\s\-]?'
    r'([A-Z]{3})$'
)

# Suisse : AA 1-6 chiffres (canton 2 lettres + numéro)
CH_REGEX = re.compile(
    r'^([A-Z]{2})'
    r'[\s\-]?'
    r'(\d{1,6})$'
)

# Cantons suisses valides
CH_CANTONS = {
    'AG', 'AI', 'AR', 'BE', 'BL', 'BS', 'FR', 'GE', 'GL', 'GR',
    'JU', 'LU', 'NE', 'NW', 'OW', 'SG', 'SH', 'SO', 'SZ', 'TG',
    'TI', 'UR', 'VD', 'VS', 'ZG', 'ZH',
}


# ═══════════════════════════════════════════════════════════════════════════
# TABLES DE CORRECTION OCR
# ═══════════════════════════════════════════════════════════════════════════

# Lettres lues dans une zone de chiffres → correction en chiffre
OCR_LETTER_TO_DIGIT: dict[str, str] = {
    'O': '0', 'Q': '0', 'D': '0',
    'I': '1', '|': '1',
    'Z': '2',
    'S': '5',
    'B': '8',
    'G': '6',
    'T': '7',
    'A': '4',
}

# Chiffres lus dans une zone de lettres → correction en lettre
OCR_DIGIT_TO_LETTER: dict[str, str] = {
    '0': 'Q',
    '1': 'L',
    '2': 'Z',
    '3': 'B',
    '4': 'A',
    '5': 'S',
    '6': 'G',
    '7': 'T',
    '8': 'B',
    '9': 'P',
}

# Lettres interdites SIV → remplacement
SIV_FORBIDDEN_FIX: dict[str, str] = {
    'I': 'J',
    'O': 'Q',
    'U': 'V',
}


# ═══════════════════════════════════════════════════════════════════════════
# DATACLASS RÉSULTAT
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PlateResult:
    """Résultat de la reconnaissance d'une plaque."""
    raw_text: str
    plate: str | None
    format: PlateFormat
    confidence: float    # 0.0 à 1.0
    corrections: int     # nombre de corrections appliquées

    def __repr__(self) -> str:
        if self.plate:
            return (
                f"[{self.format}] {self.plate} "
                f"(confiance: {self.confidence:.0%}, corrections: {self.corrections})"
            )
        return f"[UNKNOWN] '{self.raw_text}' → non reconnu"


# ═══════════════════════════════════════════════════════════════════════════
# FONCTIONS UTILITAIRES
# ═══════════════════════════════════════════════════════════════════════════

def clean_ocr_text(raw_text: str) -> str:
    """Nettoie le texte brut de l'OCR, conserve lettres, chiffres, tirets, pipes."""
    text = raw_text.upper().strip()
    text = re.sub(r'[^A-Z0-9|\-]', '', text)
    return text


def _compute_confidence(corrections: int, total_chars: int) -> float:
    """Calcule un score de confiance proportionnel au ratio de caractères corrigés."""
    if total_chars == 0:
        return 0.0
    return max(0.2, 1.0 - (corrections / total_chars))


def _correct_zone(
    chars: list[str],
    correction_map: dict[str, str],
    expect_digits: bool,
) -> int | None:
    """
    Corrige une zone de caractères en place.

    Args:
        chars: liste de caractères à corriger (modifiée en place)
        correction_map: table de substitution à appliquer
        expect_digits: True → zone de chiffres (corrige lettres parasites),
                       False → zone de lettres (corrige chiffres parasites)

    Returns:
        Nombre de corrections effectuées, ou None si une correction est impossible.
    """
    corrections = 0
    for i, ch in enumerate(chars):
        needs_fix = ch.isalpha() if expect_digits else ch.isdigit()
        if needs_fix:
            if ch in correction_map:
                chars[i] = correction_map[ch]
                corrections += 1
            else:
                return None
    return corrections


# ═══════════════════════════════════════════════════════════════════════════
# CORRECTEURS PAR FORMAT
# ═══════════════════════════════════════════════════════════════════════════

def _try_siv(text: str) -> PlateResult | None:
    """Tente de matcher et corriger au format SIV : AA-123-AA."""
    text_clean = text.replace('-', '').replace('|', '1')
    if len(text_clean) != 7:
        return None

    corrections = 0
    part_l1 = list(text_clean[0:2])
    part_d  = list(text_clean[2:5])
    part_l2 = list(text_clean[5:7])

    # Corriger zone chiffres
    corr = _correct_zone(part_d, OCR_LETTER_TO_DIGIT, expect_digits=True)
    if corr is None:
        return None
    corrections += corr

    # Corriger zones lettres
    for part in [part_l1, part_l2]:
        corr = _correct_zone(part, OCR_DIGIT_TO_LETTER, expect_digits=False)
        if corr is None:
            return None
        corrections += corr
        # Remplacer les lettres interdites SIV (I, O, U)
        for i, ch in enumerate(part):
            if ch in SIV_FORBIDDEN_FIX:
                part[i] = SIV_FORBIDDEN_FIX[ch]
                corrections += 1

    plate = f"{''.join(part_l1)}-{''.join(part_d)}-{''.join(part_l2)}"

    if SIV_REGEX.match(plate):
        return PlateResult(
            raw_text=text, plate=plate, format="SIV",
            confidence=_compute_confidence(corrections, 7),
            corrections=corrections,
        )
    return None


def _try_fni(text: str) -> PlateResult | None:
    """Tente de matcher au format FNI : 1234 AB 57."""
    text_clean = re.sub(r'[\s\-]', '', text)

    # Le FNI a entre 4 et 9 caractères (ex: "1A01" à "9999ABC99")
    if len(text_clean) < 4 or len(text_clean) > 9:
        return None

    best_match: tuple[str, int] | None = None

    # Stratégie : les 2 derniers caractères sont le département (chiffres)
    # Avant ça, il y a 1-3 lettres, et avant ça 1-4 chiffres
    for dept_start in range(len(text_clean) - 2, 1, -1):
        dept = text_clean[dept_start:]
        if len(dept) != 2:
            continue
        remaining = text_clean[:dept_start]

        # Trouver la séparation lettres/chiffres dans remaining
        letter_start = -1
        for j in range(len(remaining) - 1, -1, -1):
            if remaining[j].isdigit():
                letter_start = j + 1
                break

        if letter_start < 1:  # au moins 1 chiffre avant
            continue

        num_part = remaining[:letter_start]
        letter_part = remaining[letter_start:]

        if not (1 <= len(num_part) <= 4):
            continue
        if not (1 <= len(letter_part) <= 3):
            continue

        # Corriger chaque zone avec _correct_zone
        num_chars = list(num_part)
        letter_chars = list(letter_part)
        dept_chars = list(dept)

        corr_num = _correct_zone(num_chars, OCR_LETTER_TO_DIGIT, expect_digits=True)
        if corr_num is None:
            continue

        corr_let = _correct_zone(letter_chars, OCR_DIGIT_TO_LETTER, expect_digits=False)
        if corr_let is None:
            continue

        corr_dept = _correct_zone(dept_chars, OCR_LETTER_TO_DIGIT, expect_digits=True)
        if corr_dept is None:
            continue

        total_corr = corr_num + corr_let + corr_dept
        candidate = f"{''.join(num_chars)} {''.join(letter_chars)} {''.join(dept_chars)}"

        if FNI_REGEX.match(candidate):
            if best_match is None or total_corr < best_match[1]:
                best_match = (candidate, total_corr)

    if best_match:
        plate, corrections = best_match
        total_chars = len(plate.replace(' ', ''))
        return PlateResult(
            raw_text=text, plate=plate, format="FNI",
            confidence=_compute_confidence(corrections, total_chars),
            corrections=corrections,
        )
    return None


def _try_uk(text: str) -> PlateResult | None:
    """Tente de matcher au format UK : AA12 AAA."""
    text_clean = re.sub(r'[\s\-]', '', text)
    if len(text_clean) != 7:
        return None

    corrections = 0
    part_l1 = list(text_clean[0:2])   # 2 lettres
    part_d  = list(text_clean[2:4])   # 2 chiffres
    part_l2 = list(text_clean[4:7])   # 3 lettres

    # Corriger zone chiffres
    corr = _correct_zone(part_d, OCR_LETTER_TO_DIGIT, expect_digits=True)
    if corr is None:
        return None
    corrections += corr

    # Corriger zones lettres
    for part in [part_l1, part_l2]:
        corr = _correct_zone(part, OCR_DIGIT_TO_LETTER, expect_digits=False)
        if corr is None:
            return None
        corrections += corr

    plate = f"{''.join(part_l1)}{''.join(part_d)} {''.join(part_l2)}"

    if UK_REGEX.match(plate):
        return PlateResult(
            raw_text=text, plate=plate, format="UK",
            confidence=_compute_confidence(corrections, 7),
            corrections=corrections,
        )
    return None


def _try_ch(text: str) -> PlateResult | None:
    """Tente de matcher au format suisse : AA 123456."""
    text_clean = re.sub(r'[\s\-]', '', text)
    if len(text_clean) < 3 or len(text_clean) > 8:
        return None

    corrections = 0
    canton = list(text_clean[0:2])
    num = list(text_clean[2:])

    # Corriger canton (2 lettres)
    corr = _correct_zone(canton, OCR_DIGIT_TO_LETTER, expect_digits=False)
    if corr is None:
        return None
    corrections += corr

    # Corriger numéro (1-6 chiffres)
    corr = _correct_zone(num, OCR_LETTER_TO_DIGIT, expect_digits=True)
    if corr is None:
        return None
    corrections += corr

    canton_str = ''.join(canton)
    num_str = ''.join(num)

    # Vérifier que le canton est valide
    if canton_str not in CH_CANTONS:
        return None

    if not (1 <= len(num_str) <= 6) or not num_str.isdigit():
        return None

    plate = f"{canton_str} {num_str}"

    return PlateResult(
        raw_text=text, plate=plate, format="CH",
        confidence=_compute_confidence(corrections, len(text_clean)),
        corrections=corrections,
    )


# ═══════════════════════════════════════════════════════════════════════════
# FONCTION PRINCIPALE
# ═══════════════════════════════════════════════════════════════════════════

def recognize_plate(raw_text: str) -> PlateResult:
    """
    Reconnaît et corrige une plaque d'immatriculation.

    Essaie TOUS les formats et retourne le meilleur match (confiance la plus élevée).
    En cas d'égalité, l'ordre de priorité est : SIV > FNI > UK > CH.

    Formats supportés :
      1. SIV  (France, AA-123-AA)
      2. FNI  (France, 1234 AB 57)
      3. UK   (AA12 AAA)
      4. CH   (AA 123456)

    Args:
        raw_text: texte brut lu par l'OCR

    Returns:
        PlateResult avec la plaque corrigée ou format UNKNOWN
    """
    text = clean_ocr_text(raw_text)

    # Essayer TOUS les formats et collecter les résultats
    candidates: list[PlateResult] = []
    for try_fn in [_try_siv, _try_fni, _try_uk, _try_ch]:
        result = try_fn(text)
        if result is not None:
            result.raw_text = raw_text
            candidates.append(result)

    if candidates:
        # Retourner le candidat avec la meilleure confiance
        # En cas d'égalité, le premier dans la liste gagne (priorité SIV > FNI > UK > CH)
        return max(candidates, key=lambda r: r.confidence)

    return PlateResult(
        raw_text=raw_text, plate=None, format="UNKNOWN",
        confidence=0.0, corrections=0,
    )


# ═══════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    test_cases: list[tuple[str, PlateFormat]] = [
        # --- SIV (France) ---
        ("AB-123-CD",    "SIV"),
        ("AB123CD",      "SIV"),
        ("A8-I23-C0",    "SIV"),
        ("ab-123-cd",    "SIV"),
        ("XY-987-ZZ",    "SIV"),
        ("48-123-CD",    "SIV"),     # chiffre dans zone lettre 1

        # --- FNI (France ancien) ---
        ("1234 AB 75",   "FNI"),
        ("1234AB75",     "FNI"),
        ("567 XY 13",    "FNI"),
        ("12ABC99",      "FNI"),
        ("1 A 75",       "FNI"),     # cas court

        # --- UK ---
        ("AB12 CDE",     "UK"),
        ("AB12CDE",      "UK"),
        ("LB51 ABC",     "UK"),

        # --- Suisse ---
        ("ZH 123456",    "CH"),
        ("BE 1",         "CH"),
        ("GE12345",      "CH"),
        ("VD 999999",    "CH"),

        # --- Invalides / cas limites ---
        ("ABCDEFGH",     "UNKNOWN"),
        ("12345678",     "UNKNOWN"),
        ("X",            "UNKNOWN"),
        ("XX 123",       "UNKNOWN"),  # canton invalide
        ("",             "UNKNOWN"),  # chaîne vide
        ("###",          "UNKNOWN"),  # tout nettoyé → vide
        ("AB-@!#-CD",    "UNKNOWN"),  # caractères spéciaux → trop court
    ]

    print("=" * 65)
    print("  TEST DE RECONNAISSANCE MULTI-FORMAT DE PLAQUES")
    print("=" * 65)

    passed = 0
    failed = 0
    for raw, expected_format in test_cases:
        result = recognize_plate(raw)
        ok = result.format == expected_format
        icon = "✅" if ok else "❌"

        if ok:
            passed += 1
        else:
            failed += 1

        print(f"  {icon}  '{raw}' → {result}")
        if not ok:
            print(f"       ⚠ Attendu: {expected_format}, obtenu: {result.format}")

    print(f"\n{'=' * 65}")
    print(f"  Résultats: {passed}/{passed + failed} tests passés")
    print(f"{'=' * 65}")
