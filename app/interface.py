# app/interface.py
# Gradio UI definition for the ASR + Transliteration system

import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr

from app.asr_pipeline import ASRPipeline
from app.transliteration import TransliterationPipeline
from app.utils import format_output, ensure_dirs

logger = logging.getLogger(__name__)

asr = ASRPipeline()
trans = TransliterationPipeline()


def process_audio(audio_path: str) -> tuple:
    if audio_path is None:
        return "Please upload an audio file.", ""

    try:
        logger.info(f"Processing audio: {audio_path}")
        asr.load_model()
        transcript, _ = asr.transcribe_and_save(audio_path)
        if not transcript:
            return "Could not transcribe audio. Please try again.", ""
        transliteration, _ = trans.transliterate_and_save(transcript)
        output = format_output(transcript, transliteration)
        logger.info(f"Processing complete: {output}")
        return transcript, transliteration
    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        return f"File error: {str(e)}", ""
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return f"Error: {str(e)}", ""


def build_interface() -> gr.Blocks:
    ensure_dirs()

    with gr.Blocks(
        title="VoiceNote AI — Transliteration",
        theme=gr.themes.Soft(primary_hue="violet", secondary_hue="purple"),
    ) as demo:
        gr.Markdown(
            """
            # VoiceNote AI — Transliteration
            Upload or record Tamil audio to get a **Tamil transcript** and **ASCII romanization** (script change, not translation).
            ---
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                audio_input = gr.Audio(
                    label="Upload Tamil Audio",
                    type="filepath",
                    sources=["upload", "microphone"],
                )

                gr.Examples(
                    examples=[["sample_inputs/sample.wav"]],
                    inputs=audio_input,
                    label="Try with sample audio",
                )

                submit_btn = gr.Button(
                    "Transcribe & Transliterate",
                    variant="primary",
                    size="lg",
                )

            with gr.Column(scale=3):
                transcript_output = gr.Textbox(
                    label="Tamil Transcript",
                    placeholder="Tamil transcript will appear here...",
                    lines=8,
                    interactive=False,
                )
                transliteration_output = gr.Textbox(
                    label="Romanized Tamil (ASCII)",
                    placeholder="Romanized text will appear here...",
                    lines=8,
                    interactive=False,
                )

        submit_btn.click(
            fn=process_audio,
            inputs=[audio_input],
            outputs=[transcript_output, transliteration_output],
        )

    return demo
