from argparse import ArgumentParser
from pathlib import Path

import whisper
from dotenv import load_dotenv

from intent_parser import parse_intent, print_intent
from note_summarizer import print_summary, summarize_note
from session_store import build_session_note, create_session_id, save_session_note


DEFAULT_INITIAL_PROMPT = (
    "This audio may contain English, Tamil, Tanglish, or mixed-language voice notes. "
    "Common phrases include கோபமா வருது, பசிக்குது, சந்தோஷம், budget-friendly, "
    "nearby, reminder, meeting note, and action items."
)


class TranscriptionError(RuntimeError):
    """Raised when Whisper cannot load a model or transcribe audio."""


def transcribe_audio(
    audio_path: Path,
    model_name: str,
    language: str | None = None,
    initial_prompt: str | None = DEFAULT_INITIAL_PROMPT,
) -> str:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        model = whisper.load_model(model_name)
    except Exception as exc:
        raise TranscriptionError(f"Could not load Whisper model '{model_name}': {exc}") from exc

    options = {}
    if language:
        options["language"] = language
    if initial_prompt:
        options["initial_prompt"] = initial_prompt

    try:
        result = model.transcribe(str(audio_path), **options)
    except Exception as exc:
        raise TranscriptionError(f"Whisper could not transcribe '{audio_path}': {exc}") from exc
    return result["text"].strip()


def main() -> None:
    parser = ArgumentParser(description="Transcribe an audio file with Whisper.")
    parser.add_argument("audio_file", help="Path to the audio file to transcribe.")
    parser.add_argument(
        "--model",
        default="tiny",
        help="Whisper model to use. Default: tiny.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional Whisper language code, such as en or ta. Default: auto-detect.",
    )
    parser.add_argument(
        "--initial-prompt",
        default=DEFAULT_INITIAL_PROMPT,
        help="Optional Whisper prompt used to bias transcription for mixed-language voice notes.",
    )
    parser.add_argument(
        "--parse-intent",
        action="store_true",
        help="Send the transcript to Groq and print the universal intent object.",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Create a structured summary note with Groq.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save intent and summary output to outputs/session_<id>.json.",
    )
    parser.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Folder for saved session JSON files. Default: outputs.",
    )
    parser.add_argument(
        "--groq-model",
        default=None,
        help="Groq model to use for intent parsing. Defaults to GROQ_MODEL or llama-3.1-8b-instant.",
    )
    args = parser.parse_args()
    load_dotenv()

    text = transcribe_audio(Path(args.audio_file), args.model, args.language, args.initial_prompt)
    print(text)

    if args.parse_intent or args.summarize or args.save:
        intent_result = parse_intent(text, args.groq_model)
        print("Intent:")
        print_intent(intent_result)

        if args.summarize or args.save:
            summary_result = summarize_note(text, intent_result, args.groq_model)
            print("Summary:")
            print_summary(summary_result)

            if args.save:
                outputs_dir = Path(args.outputs_dir)
                session_id = create_session_id(outputs_dir)
                session_note = build_session_note(
                    session_id=session_id,
                    audio_file=Path(args.audio_file),
                    raw_transcript=text,
                    intent=intent_result,
                    summary=summary_result,
                )
                saved_path = save_session_note(outputs_dir, session_note)
                print("Saved to:")
                print(saved_path)


if __name__ == "__main__":
    main()
