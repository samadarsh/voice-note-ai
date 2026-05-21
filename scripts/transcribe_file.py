from argparse import ArgumentParser
from pathlib import Path

from dotenv import load_dotenv

import sys

sys.path.append(str(Path(__file__).parent.parent))

from config import GROQ_MODEL_DEFAULT, OUTPUTS_DIR, WHISPER_MODEL_DEFAULT
from core.logging_config import setup_logging
from core.pipeline import analyze_step, save_step, transcribe_step
from core.transcriber import DEFAULT_INITIAL_PROMPT


def print_json(data: dict) -> None:
    import json

    print(json.dumps(data, indent=2, ensure_ascii=False))


def main() -> None:
    parser = ArgumentParser(description="Transcribe an audio file with Whisper.")
    parser.add_argument("audio_file", help="Path to the audio file to transcribe.")
    parser.add_argument(
        "--model",
        default=WHISPER_MODEL_DEFAULT,
        help=f"Whisper model to use. Default: {WHISPER_MODEL_DEFAULT}.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional Whisper language code, such as en or ta. Default: auto-detect.",
    )
    parser.add_argument(
        "--initial-prompt",
        default=DEFAULT_INITIAL_PROMPT,
        help="Optional Whisper prompt for mixed-language voice notes.",
    )
    parser.add_argument("--analyze", action="store_true", help="Analyze transcript with Groq.")
    parser.add_argument(
        "--save",
        action="store_true",
        help="Analyze and save output to outputs/session_<id>.json.",
    )
    parser.add_argument(
        "--transliterate-only",
        action="store_true",
        help="Transcribe (+ romanize if Tamil) only; skip Groq.",
    )
    parser.add_argument(
        "--chunked",
        action="store_true",
        help="Force chunked ASR for long audio.",
    )
    parser.add_argument(
        "--no-chunked",
        action="store_true",
        help="Disable chunked ASR (single Whisper call).",
    )
    parser.add_argument(
        "--outputs-dir",
        default=OUTPUTS_DIR,
        help="Folder for saved session JSON files.",
    )
    parser.add_argument(
        "--groq-model",
        default=None,
        help=f"Groq model for analysis. Default: {GROQ_MODEL_DEFAULT}.",
    )
    args = parser.parse_args()
    setup_logging()
    load_dotenv()

    chunked = None
    if args.chunked:
        chunked = True
    elif args.no_chunked:
        chunked = False

    audio_path = Path(args.audio_file)
    tx = transcribe_step(audio_path, args.model, args.language, chunked=chunked)

    print(tx["transcript"])
    if tx.get("transliteration"):
        print("\nTransliteration (ASCII):")
        print(tx["transliteration"])
    if tx.get("asr_meta"):
        print("\nASR meta:", tx["asr_meta"])

    if args.transliterate_only:
        return

    if args.analyze or args.save:
        analysis = analyze_step(
            tx["transcript"],
            args.groq_model,
            transliteration=tx.get("transliteration"),
        )
        intent_result = analysis["intent"]
        summary_result = analysis["summary"]

        print("\nIntent:")
        print_json(intent_result)
        print("\nSummary:")
        print_json(summary_result)

        if args.save:
            saved_path = save_step(
                audio_path,
                tx["transcript"],
                intent_result,
                summary_result,
                transliteration=tx.get("transliteration"),
                asr_meta=tx.get("asr_meta"),
                outputs_dir=Path(args.outputs_dir),
            )
            print("\nSaved to:")
            print(saved_path)


if __name__ == "__main__":
    main()
