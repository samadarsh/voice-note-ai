"""Optional Tamil script → ASCII romanization after ASR."""

from core.tamil_romanizer import tamil_to_ascii
from core.text_utils import contains_tamil_script


def maybe_transliterate(transcript: str) -> str | None:
    if not transcript or not transcript.strip():
        return None
    if not contains_tamil_script(transcript):
        return None
    return tamil_to_ascii(transcript).strip() or None
