# Voice Note AI

Voice Note AI is a multilingual voice assistant that records speech, transcribes it using Whisper, extracts structured intent using an LLM, and generates concise summaries with key points and action items.

## Features

- Record a voice note in Streamlit or upload an audio file.
- Transcribe speech with Whisper.
- Extract structured intent with Groq-hosted LLMs.
- Generate a short summary, key points, and action items.
- Save session output as JSON in `outputs/`.

## Project Structure

```text
app.py                    Streamlit app
record_and_transcribe.py  CLI recorder
transcribe_file.py        Audio transcription helper
intent_parser.py          Intent extraction layer
note_summarizer.py        Summary generation layer
session_store.py          Session JSON storage helpers
outputs/                  Sample saved sessions
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file for local use:

```bash
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.1-8b-instant
```

## Run the Streamlit App

```bash
streamlit run app.py
```

For Streamlit Cloud, add `GROQ_API_KEY` in app secrets.

## Run the CLI

Record and process one voice note:

```bash
python record_and_transcribe.py --once
```

Transcribe an existing audio file:

```bash
python transcribe_file.py path/to/audio.wav --parse-intent --summarize --save
```
