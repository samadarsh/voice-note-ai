from argparse import ArgumentParser
from pathlib import Path

import whisper

from intent_parser import parse_intent, print_intent
from note_summarizer import print_summary, summarize_note
from session_store import build_session_note, get_next_session_id, save_session_note


def transcribe_audio(audio_path: Path, model_name: str) -> str:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = whisper.load_model(model_name)
    result = model.transcribe(str(audio_path))
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
        help="Save intent and summary output to outputs/session_###.json.",
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

    text = transcribe_audio(Path(args.audio_file), args.model)
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
                session_id = get_next_session_id(outputs_dir)
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
