import re


def normalize_plate_text(text: str) -> str:
    """Keep plate searches stable across common OCR punctuation errors."""
    normalized = re.sub(r"[^A-Za-z0-9]", "", text or "").upper()
    replacements = {"|": "I", "O": "0"}
    return "".join(replacements.get(character, character) for character in normalized)


def safe_plate_filename(plate: str, fallback: str = "unknown") -> str:
    value = normalize_plate_text(plate) or fallback
    return re.sub(r"[^A-Z0-9_-]", "_", value)[:64]