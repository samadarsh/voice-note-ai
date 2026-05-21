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


def contains_tamil_script(text: str) -> bool:
    return any("\u0b80" <= char <= "\u0bff" for char in text)



