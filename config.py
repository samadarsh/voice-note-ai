"""Central configuration — override via environment variables."""

import os

WHISPER_MODEL_DEFAULT = os.getenv("WHISPER_MODEL", "small")
WHISPER_MODEL_TAMIL = os.getenv("WHISPER_MODEL_TAMIL", "medium")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "")  # empty → auto-detect in whisper_service
WHISPER_FP16 = os.getenv("WHISPER_FP16", "").lower() in {"1", "true", "yes"}

CHUNK_DURATION_SEC = int(os.getenv("CHUNK_DURATION_SEC", "30"))
BUFFER_MAX_QUEUE = int(os.getenv("BUFFER_MAX_QUEUE", "10"))
CHUNK_SAMPLE_RATE = int(os.getenv("CHUNK_SAMPLE_RATE", "16000"))

GROQ_MODEL_DEFAULT = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
SESSION_MAX_FILES = int(os.getenv("SESSION_MAX_FILES", "20"))
OUTPUTS_DIR = os.getenv("OUTPUTS_DIR", "outputs")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

WHISPER_MODEL_CHOICES = ("tiny", "base", "small", "medium", "large", "turbo")


def resolve_whisper_model(selected: str, language: str | None) -> str:
    """Use Tamil-tuned default when user keeps the generic default on a Tamil hint."""
    if language == "ta" and selected == WHISPER_MODEL_DEFAULT:
        return WHISPER_MODEL_TAMIL
    return selected
