# models/model_config.py
# Central configuration for all models and system settings.
# Values can be overridden via environment variables so the same image
# runs locally, in Docker, and on hosted platforms (Hugging Face Spaces, etc).

import os


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


ASR_CONFIG = {
    "model_id":   os.getenv("WHISPER_MODEL_ID", "openai/whisper-medium"),
    "language":   os.getenv("ASR_LANGUAGE", "ta"),   # Tamil
    "task":       "transcribe",
    "device":     os.getenv("ASR_DEVICE", "cpu"),    # set "cuda" if GPU available
}

TRANSLITERATION_CONFIG = {
    "src_script": "tamil",
    "tgt_script": "ascii_latin",   # Tamil-aware Latin (ASCII) Romanization
}

BUFFER_CONFIG = {
    "max_queue_size": int(os.getenv("BUFFER_MAX_QUEUE", "10")),   # max audio chunks in queue
    "chunk_duration": int(os.getenv("CHUNK_DURATION_SEC", "30")), # seconds per chunk
    "sample_rate":    int(os.getenv("CHUNK_SAMPLE_RATE", "16000")), # whisper requires 16kHz
}

APP_CONFIG = {
    # GRADIO_SERVER_* are set by Docker/HF Spaces; PORT is set by Render/Railway.
    "host":                 os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
    "port":                 int(os.getenv("GRADIO_SERVER_PORT", os.getenv("PORT", "7860"))),
    "share":                _env_bool("GRADIO_SHARE", False),
    "output_dir":           os.getenv("OUTPUTS_DIR", "outputs"),
    "transcripts_dir":      "outputs/transcripts",
    "transliterations_dir": "outputs/transliterations",
}
