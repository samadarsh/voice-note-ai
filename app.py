import os
import tempfile
from pathlib import Path

import streamlit as st

from intent_parser import parse_intent
from note_summarizer import summarize_note
from session_store import build_session_note, get_next_session_id, save_session_note
from transcribe_file import transcribe_audio


def configure_api_key() -> None:
    try:
        api_key = st.secrets.get("GROQ_API_KEY", None)
    except Exception:
        api_key = None
    if api_key and not os.getenv("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = api_key


def save_uploaded_audio(uploaded_audio) -> Path:
    suffix = Path(uploaded_audio.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_audio.getbuffer())
        return Path(temp_file.name)


def process_audio(audio_path: Path, whisper_model: str, groq_model: str | None) -> dict:
    transcript = transcribe_audio(audio_path, whisper_model)
    intent = parse_intent(transcript, groq_model)
    summary = summarize_note(transcript, intent, groq_model)

    outputs_dir = Path("outputs")
    session_id = get_next_session_id(outputs_dir)
    session_note = build_session_note(
        session_id=session_id,
        audio_file=audio_path,
        raw_transcript=transcript,
        intent=intent,
        summary=summary,
    )
    saved_path = save_session_note(outputs_dir, session_note)

    return {
        "transcript": transcript,
        "intent": intent,
        "summary": summary,
        "saved_path": saved_path,
    }


def render_list(items: list[str], empty_text: str) -> None:
    if items:
        for item in items:
            st.markdown(f"- {item}")
    else:
        st.caption(empty_text)


st.set_page_config(page_title="Voice Note AI", page_icon="VN", layout="centered")
configure_api_key()

st.title("Voice Note AI")

with st.sidebar:
    st.header("Settings")
    whisper_model = st.selectbox("Whisper model", ["tiny", "base", "small", "medium"], index=0)
    groq_model = st.text_input("Groq model", value=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"))

audio_input = st.audio_input("Record a voice note")
uploaded_file = st.file_uploader("Or upload an audio file", type=["wav", "mp3", "m4a", "ogg", "flac"])
audio_source = audio_input or uploaded_file

if audio_source:
    st.audio(audio_source)

if st.button("Process Voice Note", type="primary", disabled=audio_source is None):
    if not os.getenv("GROQ_API_KEY"):
        st.error("Missing GROQ_API_KEY. Add it in Streamlit secrets before deploying.")
        st.stop()

    with st.spinner("Transcribing and analyzing your voice note..."):
        try:
            audio_path = save_uploaded_audio(audio_source)
            result = process_audio(audio_path, whisper_model, groq_model or None)
        except Exception as exc:
            st.error(f"Could not process this audio: {exc}")
            st.stop()

    st.subheader("Transcript")
    st.write(result["transcript"])

    st.subheader("Intent")
    st.json(result["intent"])

    st.subheader("Summary")
    st.write(result["summary"]["short_summary"])

    st.subheader("Key Points")
    render_list(result["summary"]["key_points"], "No key points found.")

    st.subheader("Action Items")
    render_list(result["summary"]["action_items"], "No action items found.")

    st.caption(f"Saved to {result['saved_path']}")
