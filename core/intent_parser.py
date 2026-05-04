from typing import Any

from core.text_utils import contains_indic_text
INTENT_SCHEMA_KEYS = [
    "intent",
    "subject",
    "content_type",
    "language_detected",
    "context",
    "confidence",
    "raw_transcript",
    "cleaned_transcript",
]


def normalize_intent_result(
    data: dict[str, Any],
    transcript: str,
    normalized_transcript: str | None = None,
) -> dict[str, Any]:
    normalized = {key: data.get(key) for key in INTENT_SCHEMA_KEYS}
    normalized["raw_transcript"] = transcript
    normalized["cleaned_transcript"] = normalized["cleaned_transcript"] or transcript.strip()
    
    if not isinstance(normalized["context"], dict):
        normalized["context"] = {}
        
    if not normalized["content_type"]:
        normalized["content_type"] = "unknown"
    if not normalized["language_detected"]:
        normalized["language_detected"] = "unknown"
    if normalized["confidence"] not in {"high", "medium", "low"}:
        normalized["confidence"] = "medium"
        
    return normalized
