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

```text
User Voice / Audio File
        ↓
Whisper Transcription
        ↓
Raw Transcript
        ↓
Groq LLM Intent Parser
        ↓
Structured Intent JSON
        ↓
Groq LLM Summarizer
        ↓
Summary + Key Points + Action Items
        ↓
Saved Session JSON
```
