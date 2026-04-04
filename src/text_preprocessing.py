import re


def clean_text(text: str) -> str:
    """Prepare review text for a TF-IDF baseline."""
    text = str(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
