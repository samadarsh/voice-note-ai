import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from session_store import build_session_note, get_next_session_id, save_session_note
from transcribe_file import transcribe_audio
from voice_note_analyzer import analyze_note


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


def process_audio(
    audio_path: Path,
    whisper_model: str,
    groq_model: str | None,
    language: str | None,
) -> dict:
    try:
        transcript = transcribe_audio(audio_path, whisper_model, language)
        analysis = analyze_note(transcript, groq_model)
        intent = analysis["intent"]
        summary = analysis["summary"]

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
    finally:
        audio_path.unlink(missing_ok=True)


def render_list(items: list[str], empty_text: str) -> None:
    if items:
        for item in items:
            st.markdown(f"- {item}")
    else:
        st.caption(empty_text)


st.set_page_config(page_title="VoiceNote AI", page_icon="🎙️", layout="wide")
configure_api_key()

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1120px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .vn-hero {
        border: 1px solid rgba(148, 163, 184, 0.28);
        border-radius: 14px;
        padding: 1.35rem 1.5rem;
        background: linear-gradient(135deg, rgba(20, 184, 166, 0.10), rgba(59, 130, 246, 0.08));
    }
    .vn-title {
        font-size: 2.45rem;
        font-weight: 760;
        line-height: 1.1;
        margin: 0;
    }
    .vn-subtitle {
        color: rgba(100, 116, 139, 0.98);
        font-size: 1.02rem;
        margin-top: 0.45rem;
        max-width: 760px;
    }
    .vn-status {
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 12px;
        padding: 0.85rem 1rem;
        background: rgba(248, 250, 252, 0.72);
    }
    .vn-status strong {
        display: block;
        font-size: 0.86rem;
        color: rgb(15, 23, 42);
        margin-bottom: 0.2rem;
    }
    .vn-status span {
        color: rgb(71, 85, 105);
        font-size: 0.9rem;
    }
    div.stButton > button {
        min-height: 3rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="vn-hero">
        <p class="vn-title">🎙️ VoiceNote AI</p>
        <p class="vn-subtitle">
            Record or upload multilingual audio, then turn it into a transcript,
            intent, summary, key points, and action items.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)
st.write("")

with st.sidebar:
    st.header("⚙️ Settings")
    whisper_model = st.selectbox("Whisper model", ["tiny", "base", "small", "medium"], index=0)
    language_label = st.selectbox("Language hint", ["Auto-detect", "Tamil", "English"], index=0)
    groq_model = st.text_input("Groq model", value=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"))
    language = {"Auto-detect": None, "Tamil": "ta", "English": "en"}[language_label]
    st.divider()
    if os.getenv("GROQ_API_KEY"):
        st.success("Groq API key loaded")
    else:
        st.warning("Groq API key missing")

left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.subheader("Add Audio")
    audio_input = st.audio_input("Record a voice note")
    uploaded_file = st.file_uploader("Or upload an audio file", type=["wav", "mp3", "m4a", "ogg", "flac"])

with right:
    st.subheader("Session")
    st.markdown(
        f"""
        <div class="vn-status">
            <strong>Whisper model</strong><span>{whisper_model}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="vn-status">
            <strong>Language hint</strong><span>{language_label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="vn-status">
            <strong>Output</strong><span>Transcript, intent, summary, key points, action items</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
audio_source = audio_input or uploaded_file

if audio_source:
    st.write("")
    st.audio(audio_source)

process_clicked = st.button("Analyze Audio", type="primary", disabled=audio_source is None, use_container_width=True)

if process_clicked:
    if not os.getenv("GROQ_API_KEY"):
        st.error("Missing GROQ_API_KEY. Add it to your local .env file or Streamlit Cloud secrets.")
        st.stop()

    with st.spinner("Transcribing and analyzing your voice note..."):
        try:
            audio_path = save_uploaded_audio(audio_source)
            result = process_audio(audio_path, whisper_model, groq_model or None, language)
        except Exception as exc:
            st.error(f"Could not process this audio: {exc}")
            st.stop()

    st.success("Analysis complete")
    st.write("")

    summary = result["summary"]
    intent = result["intent"]

    metric_cols = st.columns(3)
    metric_cols[0].metric("Intent", intent.get("intent", "unknown"))
    metric_cols[1].metric("Language", summary.get("language_detected", "unknown"))
    metric_cols[2].metric("Confidence", intent.get("confidence", "unknown"))

    tab_summary, tab_transcript, tab_intent = st.tabs(["Summary", "Transcript", "Intent JSON"])

    with tab_summary:
        st.subheader(summary.get("suggested_title", "Untitled Note"))
        st.write(summary["short_summary"])

        col_a, col_b = st.columns(2, gap="large")
        with col_a:
            st.markdown("#### Key Points")
            render_list(summary["key_points"], "No key points found.")
        with col_b:
            st.markdown("#### Action Items")
            render_list(summary["action_items"], "No action items found.")

        st.markdown("#### Important Entities")
        render_list(summary.get("important_entities", []), "No important entities found.")

    with tab_transcript:
        st.write(result["transcript"])

    with tab_intent:
        st.json(intent)

    st.caption(f"Saved to {result['saved_path']}")
