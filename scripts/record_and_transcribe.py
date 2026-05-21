from argparse import ArgumentParser
from pathlib import Path

import sounddevice as sd
from dotenv import load_dotenv
from scipy.io.wavfile import write

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import OUTPUTS_DIR, WHISPER_MODEL_DEFAULT
from core.logging_config import setup_logging
from core.pipeline import analyze_step, save_step, transcribe_step
from storage.session_store import create_session_id


def record_audio(output_path: Path, duration: float, sample_rate: int) -> None:
    print("Press Enter to start recording.")
    input()

    print(f"Recording for {duration:g} seconds...")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()

    write(output_path, sample_rate, audio)
    print(f"Saved audio to {output_path}")


def print_note_output(
    audio_path: Path,
    transcript: str,
    intent_result: dict,
    summary_result: dict,
    saved_path: Path,
) -> None:
    print()
    print(f"Recording saved: {audio_path}")
    print()
    print("Transcript:")
    print(transcript)
    print()
    print("Detected Intent:")
    print_json(intent_result)
    print()
    print("Summary:")
    print(summary_result["short_summary"])
    print()
    print("Key Points:")
    if summary_result["key_points"]:
        for point in summary_result["key_points"]:
            print(f"- {point}")
    else:
        print("- None")
    print()
    print("Action Items:")
    if summary_result["action_items"]:
        for item in summary_result["action_items"]:
            print(f"- {item}")
    else:
        print("- None")
    print()
    print("Saved to:")
    print(saved_path)


def print_json(data: dict) -> None:
    import json

    print(json.dumps(data, indent=2, ensure_ascii=False))


def process_recording(args: object) -> None:
    outputs_dir = Path(args.outputs_dir)
    session_id = create_session_id(outputs_dir)
    audio_path = outputs_dir / f"{session_id}.wav"

    record_audio(audio_path, args.duration, args.sample_rate)

    print("Transcribing...")
    tx = transcribe_step(audio_path, args.model, args.language)
    transcript = tx["transcript"]

    if tx.get("transliteration"):
        print("\nTransliteration (ASCII):")
        print(tx["transliteration"])

    if not args.analyze:
        print("Transcript:")
        print(transcript)
        return

    print("Analyzing...")
    analysis = analyze_step(
        transcript,
        args.groq_model,
        transliteration=tx.get("transliteration"),
    )
    intent_result = analysis["intent"]
    summary_result = analysis["summary"]

    saved_path = save_step(
        audio_path,
        transcript,
        intent_result,
        summary_result,
        transliteration=tx.get("transliteration"),
        asr_meta=tx.get("asr_meta"),
        outputs_dir=outputs_dir,
    )

    print_note_output(audio_path, transcript, intent_result, summary_result, saved_path)


def main() -> None:
    parser = ArgumentParser(description="Record voice notes, summarize them, and save structured output.")
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Recording duration in seconds. Default: 10.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Deprecated. Audio is saved per session inside outputs/.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Recording sample rate in Hz. Default: 16000.",
    )
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
        "--groq-model",
        default=None,
        help="Groq model to use for analysis. Defaults to GROQ_MODEL or llama-3.1-8b-instant.",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        default=True,
        help="Analyze transcript with Groq and save structured output. Enabled by default.",
    )
    parser.add_argument(
        "--no-analyze",
        action="store_false",
        dest="analyze",
        help="Only record and transcribe audio; do not call Groq.",
    )
    parser.add_argument(
        "--skip-intent",
        action="store_true",
        help="Deprecated alias for --no-analyze.",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Deprecated alias for --no-analyze.",
    )
    parser.add_argument(
        "--outputs-dir",
        default=OUTPUTS_DIR,
        help=f"Folder for session WAV and JSON files. Default: {OUTPUTS_DIR}.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process one recording and exit without loop prompt.",
    )
    args = parser.parse_args()
    setup_logging()
    load_dotenv()
    if args.skip_llm or args.skip_intent:
        args.analyze = False

    while True:
        process_recording(args)
        if args.once:
            break

        choice = input("\nPress Enter to record again, or type q to quit: ").strip().lower()
        if choice == "q":
            break


if __name__ == "__main__":
    main()
