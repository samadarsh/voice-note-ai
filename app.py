import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from config import (
    GROQ_MODEL_DEFAULT,
    OUTPUTS_DIR,
    WHISPER_MODEL_CHOICES,
    WHISPER_MODEL_DEFAULT,
)
from core.logging_config import setup_logging
from core.pipeline import analyze_step, save_step, transcribe_step
from core.groq_client import groq_api_key_status, is_configured_groq_api_key
from core.whisper_service import clear_model_cache
from storage.session_store import list_session_files, load_session_note


def configure_api_key() -> None:
    load_dotenv()
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


def render_list(items: list[str], empty_text: str) -> None:
    if items:
        for item in items:
            st.markdown(f"- {item}")
    else:
        st.caption(empty_text)


def build_export_text(result: dict) -> str:
    lines = [
        f"Title: {result['summary'].get('suggested_title', 'Untitled')}",
        "",
        "Transcript:",
        result["transcript"],
    ]
    if result.get("transliteration"):
        lines.extend(["", "Transliteration (ASCII):", result["transliteration"]])
    lines.extend(
        [
            "",
            "Summary:",
            result["summary"].get("short_summary", ""),
            "",
            "Key Points:",
        ]
    )
    lines.extend(f"- {kp}" for kp in result["summary"].get("key_points", []) or ["None"])
    lines.extend(["", "Action Items:"])
    lines.extend(f"- {ai}" for ai in result["summary"].get("action_items", []) or ["None"])
    return "\n".join(lines)


def render_session_history(outputs_dir: Path) -> None:
    sessions = list_session_files(outputs_dir)
    if not sessions:
        st.caption("No saved sessions yet.")
        return

    labels = [p.name for p in sessions]
    choice = st.selectbox("Past sessions", labels, index=0)
    if not choice:
        return

    note = load_session_note(outputs_dir / choice)
    st.markdown(f"**{note.get('suggested_title', note['session_id'])}**")
    st.caption(note.get("timestamp", ""))
    st.write(note.get("raw_transcript", ""))
    if note.get("transliteration"):
        with st.expander("Romanized (ASCII)"):
            st.code(note["transliteration"])
    st.write(note.get("summary", ""))


setup_logging()
st.set_page_config(page_title="VoiceNote AI", page_icon="🎙️", layout="centered")
configure_api_key()

st.title("🎙️ VoiceNote AI")
outputs_dir = Path(OUTPUTS_DIR)

with st.sidebar:
    st.header("Settings")
    default_index = (
        WHISPER_MODEL_CHOICES.index(WHISPER_MODEL_DEFAULT)
        if WHISPER_MODEL_DEFAULT in WHISPER_MODEL_CHOICES
        else 2
    )
    whisper_model = st.selectbox(
        "Whisper model",
        list(WHISPER_MODEL_CHOICES),
        index=default_index,
    )
    language_label = st.selectbox(
        "Language hint", ["Auto-detect", "Tamil", "English"], index=0
    )
    language = {"Auto-detect": None, "Tamil": "ta", "English": "en"}[language_label]
    force_chunked = st.checkbox("Force chunked ASR (long audio)", value=False)
    groq_model = st.text_input("Groq model", value=os.getenv("GROQ_MODEL", GROQ_MODEL_DEFAULT))

    groq_ok, groq_status = groq_api_key_status()
    if groq_ok:
        st.caption(f"Groq: {groq_status}")
    else:
        st.warning(f"Groq: {groq_status}")

    if st.button("Clear cached Whisper model"):
        clear_model_cache()
        st.success("Model cache cleared.")

    st.divider()
    st.subheader("History")
    render_session_history(outputs_dir)

audio_input = st.audio_input("Record a voice note")
uploaded_file = st.file_uploader(
    "Or upload an audio file", type=["wav", "mp3", "m4a", "ogg", "flac"]
)
audio_source = audio_input or uploaded_file

if audio_source:
    st.audio(audio_source)

if st.button("Process Voice Note", type="primary", disabled=audio_source is None):
    if not is_configured_groq_api_key():
        st.error(
            "Groq API key is missing or invalid. Create `.env` in the project root with:\n\n"
            "`GROQ_API_KEY=gsk_your_real_key`\n\n"
            "Get a key at https://console.groq.com/keys — do not leave the "
            "`.env.example` placeholder. Restart Streamlit after saving."
        )
        st.stop()

    audio_path: Path | None = None
    try:
        audio_path = save_uploaded_audio(audio_source)
        chunked = True if force_chunked else None

        with st.spinner("Transcribing audio…"):
            tx = transcribe_step(audio_path, whisper_model, language, chunked=chunked)

        st.subheader("Transcript")
        st.write(tx["transcript"])
        if tx.get("transliteration"):
            with st.expander("Romanized (ASCII)"):
                st.code(tx["transliteration"])
        if tx.get("asr_meta"):
            st.caption(
                f"ASR: model={tx['asr_meta'].get('model')} · "
                f"chunked={tx['asr_meta'].get('chunked')} · "
                f"chunks={tx['asr_meta'].get('chunk_count')}"
            )

        with st.spinner("Analyzing intent and summary…"):
            analysis = analyze_step(
                tx["transcript"],
                groq_model or None,
                transliteration=tx.get("transliteration"),
            )

        intent = analysis["intent"]
        summary = analysis["summary"]
        saved_path = save_step(
            audio_path,
            tx["transcript"],
            intent,
            summary,
            transliteration=tx.get("transliteration"),
            asr_meta=tx.get("asr_meta"),
            outputs_dir=outputs_dir,
        )

        result = {
            "transcript": tx["transcript"],
            "transliteration": tx.get("transliteration"),
            "intent": intent,
            "summary": summary,
            "saved_path": saved_path,
        }

        st.subheader("Intent")
        st.json(result["intent"])

        st.subheader("Summary")
        st.write(result["summary"]["short_summary"])

        st.subheader("Key Points")
        render_list(result["summary"]["key_points"], "No key points found.")

        st.subheader("Action Items")
        render_list(result["summary"]["action_items"], "No action items found.")

        st.caption(f"Saved to {result['saved_path']}")

        st.download_button(
            label="⬇️ Download Note as Text",
            data=build_export_text(result),
            file_name="voice_note_summary.txt",
            mime="text/plain",
        )
    except Exception as exc:
        st.error(str(exc))
        st.caption(
            "If transcription succeeded above, your Whisper step is fine — fix the Groq key "
            "in `.env` and try again."
        )
    finally:
        if audio_path is not None:
            audio_path.unlink(missing_ok=True)
