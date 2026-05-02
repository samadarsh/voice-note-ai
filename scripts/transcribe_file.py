from argparse import ArgumentParser
from pathlib import Path

import whisper
from dotenv import load_dotenv

import sys
sys.path.append(str(Path(__file__).parent.parent))

from storage.session_store import build_session_note, create_session_id, save_session_note
from core.voice_note_analyzer import analyze_note
from core.transcriber import DEFAULT_INITIAL_PROMPT, transcribe_audio


def print_json(data: dict) -> None:
    import json

    print(json.dumps(data, indent=2, ensure_ascii=False))


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
    parser.add_argument("--analyze", action="store_true", help="Analyze transcript with Groq.")
    parser.add_argument(
        "--save",
        action="store_true",
        help="Analyze and save output to outputs/session_<id>.json.",
    )
    parser.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Folder for saved session JSON files. Default: outputs.",
    )
    parser.add_argument(
        "--groq-model",
        default=None,
        help="Groq model to use for analysis. Defaults to GROQ_MODEL or llama-3.1-8b-instant.",
    )
    args = parser.parse_args()
    load_dotenv()

    text = transcribe_audio(Path(args.audio_file), args.model, args.language, args.initial_prompt)
    print(text)

    if args.analyze or args.save:
        analysis = analyze_note(text, args.groq_model)
        intent_result = analysis["intent"]
        summary_result = analysis["summary"]

        print("Intent:")
        print_json(intent_result)
        print("Summary:")
        print_json(summary_result)

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
