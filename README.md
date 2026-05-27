# VoiceNote AI — Transliteration

Tamil voice-note pipeline: Whisper ASR + custom Tamil-to-ASCII romanization (Gradio UI).

This repository runs the same ASR + transliteration flow as your
`task2_asr_transliteration` project:

- Whisper ASR (`openai/whisper-medium`)
- Tamil script to ASCII romanization (custom mapper)
- Gradio UI
- No LLM analysis

## Run locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app/main.py
```

Open: `http://localhost:7860`

## Docker

```bash
docker compose up --build
```

## Structure

- `app/` - Task 2 style app code (ASR, transliteration, interface, main)
- `models/` - central config
- `sample_inputs/` - input sample audio
- `outputs/transcripts`, `outputs/transliterations` - generated outputs
- `tests/test_pipeline.py` - pipeline tests
