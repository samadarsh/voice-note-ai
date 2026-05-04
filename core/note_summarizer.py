from typing import Any




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
    normalized = {key: data.get(key) for key in SUMMARY_SCHEMA_KEYS}
    
    normalized["cleaned_transcript"] = normalized["cleaned_transcript"] or transcript.strip()
    normalized["short_summary"] = normalized["short_summary"] or "No clear summary available."
    
    for key in ("key_points", "action_items", "important_entities"):
        if not isinstance(normalized[key], list):
            normalized[key] = []
            
    normalized["language_detected"] = normalized["language_detected"] or "unknown"
    normalized["suggested_title"] = normalized["suggested_title"] or "Untitled Voice Note"
    
    return normalized
