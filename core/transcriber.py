from pathlib import Path
import whisper

DEFAULT_INITIAL_PROMPT = (
    "This audio may contain English, Tamil, Tanglish, or mixed-language voice notes. "
    "Common phrases include கோபமா வருது, பசிக்குது, சந்தோஷம், budget-friendly, "
    "nearby, reminder, meeting note, and action items."
)

class TranscriptionError(RuntimeError):
    """Raised when Whisper cannot load a model or transcribe audio."""

def transcribe_audio(
    audio_path: Path,
    model_name: str,
    language: str | None = None,
    initial_prompt: str | None = DEFAULT_INITIAL_PROMPT,
) -> str:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        model = whisper.load_model(model_name)
    except Exception as exc:
        raise TranscriptionError(f"Could not load Whisper model '{model_name}': {exc}") from exc

    options = {}
    if language:
        options["language"] = language
    if initial_prompt:
        options["initial_prompt"] = initial_prompt

    try:
        result = model.transcribe(str(audio_path), **options)
    except Exception as exc:
        raise TranscriptionError(f"Whisper could not transcribe '{audio_path}': {exc}") from exc
    return result["text"].strip()
