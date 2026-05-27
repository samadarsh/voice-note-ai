# models/model_config.py
# Central configuration for all models and system settings

ASR_CONFIG = {
    "model_id":   "openai/whisper-medium",
    "language":   "ta",        # Tamil
    "task":       "transcribe",
    "device":     "cpu",       # change to "cuda" if GPU available
}

TRANSLITERATION_CONFIG = {
    "src_script": "tamil",
    "tgt_script": "ascii_latin",   # Tamil-aware Latin (ASCII) Romanization
}

BUFFER_CONFIG = {
    "max_queue_size": 10,      # max audio chunks in queue
    "chunk_duration": 30,      # seconds per chunk
    "sample_rate":    16000,   # whisper requires 16kHz
}

APP_CONFIG = {
    "host":                 "0.0.0.0",
    "port":                 7860,
    "share":                False,
    "output_dir":           "outputs",
    "transcripts_dir":      "outputs/transcripts",
    "transliterations_dir": "outputs/transliterations",
}
