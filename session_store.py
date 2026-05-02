import json
from datetime import datetime
from pathlib import Path
from typing import Any


def get_next_session_id(outputs_dir: Path) -> str:
    outputs_dir.mkdir(exist_ok=True)
    session_numbers = []
    for path in outputs_dir.glob("session_*.json"):
        try:
            session_numbers.append(int(path.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue

    next_number = max(session_numbers, default=0) + 1
    return f"session_{next_number:03d}"


def build_session_note(
    session_id: str,
    audio_file: Path,
    raw_transcript: str,
    intent: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "audio_file": str(audio_file),
        "raw_transcript": raw_transcript,
        "cleaned_transcript": summary["cleaned_transcript"],
        "intent": intent,
        "summary": summary["short_summary"],
        "key_points": summary["key_points"],
        "action_items": summary["action_items"],
        "important_entities": summary["important_entities"],
        "language_detected": summary["language_detected"],
        "suggested_title": summary["suggested_title"],
    }


def save_session_note(outputs_dir: Path, session_note: dict[str, Any]) -> Path:
    outputs_dir.mkdir(exist_ok=True)
    output_path = outputs_dir / f"{session_note['session_id']}.json"
    output_path.write_text(
        json.dumps(session_note, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path
