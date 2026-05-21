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

SAMPLE_INPUTS_DIR = Path(__file__).resolve().parent / "sample_inputs"
SAMPLE_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}
SAMPLE_AUDIO_MIME = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
}


def list_sample_input_files() -> list[Path]:
    if not SAMPLE_INPUTS_DIR.is_dir():
        return []
    return sorted(
        p
        for p in SAMPLE_INPUTS_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SAMPLE_AUDIO_EXTENSIONS
    )


def sample_label(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ").strip().title() or path.name


def sample_mime(path: Path) -> str:
    return SAMPLE_AUDIO_MIME.get(path.suffix.lower(), "audio/wav")


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

sample_files = list_sample_input_files()
if sample_files:
    st.subheader("Try a sample")
    sample_cols = st.columns(len(sample_files))
    for col, sample_path in zip(sample_cols, sample_files):
        with col:
            st.caption(sample_label(sample_path))
            st.audio(sample_path.read_bytes(), format=sample_mime(sample_path))
            if st.button(
                "Use this sample",
                key=f"sample_{sample_path.name}",
                use_container_width=True,
            ):
                st.session_state["active_sample"] = sample_path.name
                st.rerun()
else:
    st.caption("Add audio files to the `sample_inputs/` folder to show samples here.")

if st.session_state.get("active_sample"):
    active_name = st.session_state["active_sample"]
    st.info(f"Sample selected: **{sample_label(SAMPLE_INPUTS_DIR / active_name)}**")
    if st.button("Clear sample"):
        st.session_state.pop("active_sample", None)
        st.rerun()

audio_input = st.audio_input("Record a voice note")
uploaded_file = st.file_uploader(
    "Or upload an audio file", type=["wav", "mp3", "m4a", "ogg", "flac"]
)

live_input = audio_input or uploaded_file
if live_input:
    st.session_state.pop("active_sample", None)
    st.audio(live_input)
elif active_sample_file := st.session_state.get("active_sample"):
    preview = SAMPLE_INPUTS_DIR / active_sample_file
    if preview.exists():
        st.audio(preview.read_bytes(), format=sample_mime(preview))

active_sample_file = st.session_state.get("active_sample")
has_audio = bool(live_input or active_sample_file)

if st.button("Process Voice Note", type="primary", disabled=not has_audio):
    if not is_configured_groq_api_key():
        st.error(
            "Groq API key is missing or invalid. Create `.env` in the project root with:\n\n"
            "`GROQ_API_KEY=gsk_your_real_key`\n\n"
            "Get a key at https://console.groq.com/keys — do not leave the "
            "`.env.example` placeholder. Restart Streamlit after saving."
        )
        st.stop()

    audio_path: Path | None = None
    delete_audio_after = False
    try:
        if live_input:
            audio_path = save_uploaded_audio(live_input)
            delete_audio_after = True
        elif active_sample_file:
            audio_path = SAMPLE_INPUTS_DIR / active_sample_file
            if not audio_path.exists():
                st.error(f"Sample file not found: {audio_path.name}")
                st.stop()
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
        if delete_audio_after and audio_path is not None:
            audio_path.unlink(missing_ok=True)
