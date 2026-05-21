"""End-to-end voice note processing pipeline."""

from pathlib import Path
from typing import Any

from config import OUTPUTS_DIR, resolve_whisper_model
from core.transcriber import DEFAULT_INITIAL_PROMPT
from core.transliteration import maybe_transliterate
from core.whisper_service import transcribe_audio_full
from core.voice_note_analyzer import analyze_note
from storage.session_store import build_session_note, create_session_id, save_session_note


def transcribe_step(
    audio_path: Path,
    whisper_model: str,
    language: str | None,
    chunked: bool | None = None,
) -> dict[str, Any]:
    model = resolve_whisper_model(whisper_model, language)
    prompt = DEFAULT_INITIAL_PROMPT if language is None else None
    result = transcribe_audio_full(
        audio_path,
        model,
        language,
        prompt,
        chunked=chunked,
    )
    transliteration = maybe_transliterate(result.transcript)
    return {
        "transcript": result.transcript,
        "transliteration": transliteration,
        "asr_meta": result.asr_meta,
    }


def analyze_step(
    transcript: str,
    groq_model: str | None,
    transliteration: str | None = None,
) -> dict[str, Any]:
    return analyze_note(transcript, groq_model, transliteration=transliteration)


def save_step(
    audio_path: Path,
    transcript: str,
    intent: dict[str, Any],
    summary: dict[str, Any],
    transliteration: str | None = None,
    asr_meta: dict[str, Any] | None = None,
    outputs_dir: Path | None = None,
) -> Path:
    out = Path(outputs_dir or OUTPUTS_DIR)
    session_id = create_session_id(out)
    session_note = build_session_note(
        session_id=session_id,
        audio_file=audio_path,
        raw_transcript=transcript,
        intent=intent,
        summary=summary,
        transliteration=transliteration,
        asr_meta=asr_meta,
    )
    return save_session_note(out, session_note)
