from pathlib import Path

from core.whisper_service import (
    TranscribeResult,
    TranscriptionError,
    transcribe_audio_full,
)

DEFAULT_INITIAL_PROMPT = (
    "This audio may contain English, Tamil, Tanglish, or mixed-language voice notes. "
    "Common phrases include kobama varuthu, pasikkuthu, santhosham, budget-friendly, "
    "nearby, reminder, meeting note, and action items."
)


def transcribe_audio(
    audio_path: Path,
    model_name: str,
    language: str | None = None,
    initial_prompt: str | None = DEFAULT_INITIAL_PROMPT,
    chunked: bool | None = None,
) -> str:
    return transcribe_audio_full(
        audio_path,
        model_name,
        language,
        initial_prompt,
        chunked=chunked,
    ).transcript
