from typing import Any

from core.text_utils import (
    contains_indic_text,
    extract_personal_intro_context,
    is_personal_introduction,
    normalize_known_terms,
    normalize_known_terms_in_value,
)


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


def _detect_clear_tamil_emotion(transcript: str) -> str | None:
    if "கோவ" in transcript or "கோப" in transcript:
        return "angry"
    if "சோகம்" in transcript or "வருத்தமா" in transcript:
        return "sad"
    if "சந்தோஷ" in transcript:
        return "happy"
    return None


def normalize_intent_result(
    data: dict[str, Any],
    transcript: str,
    normalized_transcript: str | None = None,
) -> dict[str, Any]:
    cleaned_transcript = normalize_known_terms(normalized_transcript or transcript.strip())
    normalized = {key: normalize_known_terms_in_value(data.get(key)) for key in INTENT_SCHEMA_KEYS}
    normalized["raw_transcript"] = transcript
    if contains_indic_text(transcript) and normalized_transcript:
        normalized["cleaned_transcript"] = cleaned_transcript
    else:
        normalized["cleaned_transcript"] = normalize_known_terms(
            normalized["cleaned_transcript"] or transcript.strip()
        )
    if not isinstance(normalized["context"], dict):
        normalized["context"] = {}
    normalized["context"] = normalize_known_terms_in_value(normalized["context"])
    if contains_indic_text(transcript) and cleaned_transcript != transcript:
        normalized["context"].setdefault("language_style", "tanglish")
        normalized["context"].setdefault("transcript_quality", "noisy_phonetic")
    if is_personal_introduction(transcript):
        normalized["intent"] = "personal_note"
        normalized["subject"] = "personal introduction"
        normalized["content_type"] = "note"
        normalized["confidence"] = "high"
        normalized["context"] = extract_personal_intro_context(transcript, normalized["context"])
    tamil_emotion = _detect_clear_tamil_emotion(transcript)
    if tamil_emotion:
        normalized["intent"] = normalized["intent"] or "personal_note"
        normalized["subject"] = normalized["subject"] or f"feeling {tamil_emotion} now"
        normalized["context"].setdefault("current_state", tamil_emotion)
        normalized["confidence"] = "high"
    if not normalized["content_type"]:
        normalized["content_type"] = "unknown"
    if not normalized["language_detected"]:
        normalized["language_detected"] = "unknown"
    if normalized["confidence"] not in {"high", "medium", "low"}:
        normalized["confidence"] = "medium" if cleaned_transcript != transcript else "low"
    return normalized
