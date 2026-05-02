# 🎙️ Voice Note AI

Voice Note AI is a multilingual AI voice-note assistant that records or accepts audio, transcribes speech using Whisper, extracts structured intent using Groq LLMs, summarizes the note, and saves the complete session as JSON.

It is designed for English, Tamil, Tanglish, and multilingual voice notes.

---

## 🚀 What It Does

Voice Note AI converts raw speech into structured, useful notes.

Example:

You speak:

> எனக்கு இப்ப மிகவும் கோபமாக வருகிறது

The system produces:

- Transcript
- Detected intent
- Cleaned transcript
- Short summary
- Key points
- Action items
- Important entities
- Saved JSON output

---

## ✨ Features

- 🎙️ Record voice notes directly in Streamlit
- 📁 Upload audio files such as WAV, MP3, M4A, OGG, and FLAC
- 🧠 Transcribe speech using OpenAI Whisper
- ⚡ Extract structured intent using Groq-hosted LLMs
- 📝 Generate summaries, key points, and action items
- 🌐 Supports English, Tamil, Tanglish, and mixed-language speech
- 💾 Saves every session as JSON inside the `outputs/` folder
- 🖥️ Includes both Streamlit app and CLI workflow

---

## 🧠 Core Idea

Most note-taking apps only store audio or text.

Voice Note AI goes one step further.

It understands what the user said and converts it into a structured note that can later be used for:

- Personal note tracking
- Voice journaling
- Task extraction
- Meeting note summarization
- Multilingual speech understanding
- AI assistant workflows

---

## 🏗️ System Architecture

```mermaid
flowchart LR
    A[Audio] --> B[Whisper]
    B --> C[Transcript]
    C --> D[Intent]
    D --> E[Summary]
    E --> F[JSON]
```

---

## 📦 Repository Structure

```text
voice-note-ai/
├── app.py                    # Streamlit web app
├── record_and_transcribe.py  # CLI recorder workflow
├── transcribe_file.py        # Transcribe and process existing audio files
├── intent_parser.py          # Groq-powered structured intent extraction
├── note_summarizer.py        # Groq-powered summaries and action items
├── session_store.py          # Session ID and JSON saving helpers
├── parse_intent.py           # Small CLI for parsing text intent
├── requirements.txt          # Python dependencies
├── .env.example              # Example environment variables
└── outputs/                  # Sample saved session JSON files
```

---

## ✅ Prerequisites

- Python 3.10 or newer
- FFmpeg, required by Whisper for audio processing
- A Groq API key

Install FFmpeg:

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt update
sudo apt install ffmpeg
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/samadarsh/voice-note-ai.git
cd voice-note-ai
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Variables

For local development, create a `.env` file in the project root:

```bash
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

For Streamlit Cloud, add the same key in app secrets:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
GROQ_MODEL = "llama-3.1-8b-instant"
```

---

## ▶️ Usage

Run the Streamlit web app:

```bash
streamlit run app.py
```

Record and process a voice note from the CLI:

```bash
python record_and_transcribe.py --once
```

Transcribe and process an existing audio file:

```bash
python transcribe_file.py path/to/audio.wav --parse-intent --summarize --save
```
