from typing import Any

from core.text_utils import (
    extract_personal_intro_context,
    is_personal_introduction,
    normalize_known_terms,
    normalize_known_terms_in_value,
)


SUMMARY_SCHEMA_KEYS = [
    "cleaned_transcript",
    "short_summary",
    "key_points",
    "action_items",
    "important_entities",
    "language_detected",
    "suggested_title",
]


def normalize_summary_result(data: dict[str, Any], transcript: str) -> dict[str, Any]:
    normalized = {key: normalize_known_terms_in_value(data.get(key)) for key in SUMMARY_SCHEMA_KEYS}
    normalized["cleaned_transcript"] = normalize_known_terms(
        normalized["cleaned_transcript"] or transcript.strip()
    )
    normalized["short_summary"] = normalize_known_terms(
        normalized["short_summary"] or "No clear summary available."
    )
    for key in ("key_points", "action_items", "important_entities"):
        if not isinstance(normalized[key], list):
            normalized[key] = []
        normalized[key] = normalize_known_terms_in_value(normalized[key])
    normalized["language_detected"] = normalized["language_detected"] or "unknown"
    normalized["suggested_title"] = normalized["suggested_title"] or "Untitled Voice Note"
    if is_personal_introduction(transcript):
        context = extract_personal_intro_context(transcript, {})
        name = context.get("name")
        role = context.get("role")
        interests = context.get("interests", [])
        if name and role and interests:
            normalized["language_detected"] = "multilingual"
            normalized["suggested_title"] = "Personal Introduction"
    return normalized
