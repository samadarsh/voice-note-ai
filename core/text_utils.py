import json
import re
from typing import Any


def extract_json(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(content[start : end + 1])


def contains_indic_text(text: str) -> bool:
    return any(
        "\u0900" <= char <= "\u097f"
        or "\u0980" <= char <= "\u09ff"
        or "\u0a00" <= char <= "\u0a7f"
        or "\u0a80" <= char <= "\u0aff"
        or "\u0b00" <= char <= "\u0b7f"
        or "\u0b80" <= char <= "\u0bff"
        or "\u0c00" <= char <= "\u0c7f"
        or "\u0c80" <= char <= "\u0cff"
        or "\u0d00" <= char <= "\u0d7f"
        for char in text
    )


def normalize_known_terms(text: str) -> str:
    replacements = [
        (r"சுடோக்கு", "Sudoku"),
        (r"சூடோக்கு", "Sudoku"),
        (r"\bsudokku\b", "Sudoku"),
        (r"\bstooges\b", "Sudoku"),
        (r"\blink-donல்\b", "LinkedInல்"),
        (r"\blink[\s-]?don\b", "LinkedIn"),
        (r"\blinkedin\b", "LinkedIn"),
        (r"\bM\.?\s*Tech\s+AI\b", "M.Tech AI"),
    ]
    normalized = text
    for pattern, replacement in replacements:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    return normalized


def normalize_known_terms_in_value(value: Any) -> Any:
    if isinstance(value, str):
        return normalize_known_terms(value)
    if isinstance(value, list):
        return [normalize_known_terms_in_value(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize_known_terms_in_value(item) for key, item in value.items()}
    return value


def is_personal_introduction(transcript: str) -> bool:
    lowered = transcript.lower()
    has_intro = "my name" in lowered or "என் பெய" in transcript or "பெயரு" in transcript
    has_role = "student" in lowered or "m.tech" in lowered or "மாணவர்" in transcript
    has_interests = (
        "பிடிச்ச" in transcript
        or "interested" in lowered
        or "like" in lowered
        or "hobby" in lowered
    )
    return has_intro and (has_role or has_interests)


def extract_personal_intro_context(transcript: str, existing_context: dict[str, Any]) -> dict[str, Any]:
    # Let the LLM handle context extraction dynamically based on ANALYSIS_PROMPT
    return existing_context
