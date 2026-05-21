import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from config import SESSION_MAX_FILES


def create_session_id(outputs_dir: Path) -> str:
    outputs_dir.mkdir(exist_ok=True)
    return f"session_{uuid4().hex[:8]}"


def build_session_note(
    session_id: str,
    audio_file: Path,
    raw_transcript: str,
    intent: dict[str, Any],
    summary: dict[str, Any],
    transliteration: str | None = None,
    asr_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    note: dict[str, Any] = {
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
    if transliteration:
        note["transliteration"] = transliteration
    if asr_meta:
        note["asr_meta"] = asr_meta
    return note


def list_session_files(outputs_dir: Path) -> list[Path]:
    if not outputs_dir.exists():
        return []
    return sorted(
        outputs_dir.glob("session_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def load_session_note(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def cleanup_old_sessions(outputs_dir: Path, max_files: int = 20) -> None:
    if not outputs_dir.exists():
        return
    json_files = list(outputs_dir.glob("session_*.json"))
    if len(json_files) > max_files:
        json_files.sort(key=lambda p: p.stat().st_mtime)
        for old_file in json_files[:-max_files]:
            old_file.unlink(missing_ok=True)


def save_session_note(outputs_dir: Path, session_note: dict[str, Any]) -> Path:
    outputs_dir.mkdir(exist_ok=True)
    output_path = outputs_dir / f"{session_note['session_id']}.json"
    output_path.write_text(
        json.dumps(session_note, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    cleanup_old_sessions(outputs_dir, max_files=SESSION_MAX_FILES)
    return output_path
