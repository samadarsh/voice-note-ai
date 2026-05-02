from argparse import ArgumentParser
from pathlib import Path

import sounddevice as sd
from dotenv import load_dotenv
from scipy.io.wavfile import write

from intent_parser import parse_intent, print_intent
from note_summarizer import summarize_note
from session_store import build_session_note, create_session_id, save_session_note
from transcribe_file import DEFAULT_INITIAL_PROMPT, transcribe_audio


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
    print_intent(intent_result)
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


def process_recording(args: object) -> None:
    outputs_dir = Path(args.outputs_dir)
    session_id = create_session_id(outputs_dir)
    audio_path = outputs_dir / f"{session_id}.wav"

    record_audio(audio_path, args.duration, args.sample_rate)

    print("Transcribing...")
    transcript = transcribe_audio(audio_path, args.model, args.language, args.initial_prompt)

    if args.skip_llm:
        print("Transcript:")
        print(transcript)
        return

    print("Extracting intent...")
    intent_result = parse_intent(transcript, args.groq_model)

    print("Summarizing...")
    summary_result = summarize_note(transcript, intent_result, args.groq_model)

    session_note = build_session_note(
        session_id=session_id,
        audio_file=audio_path,
        raw_transcript=transcript,
        intent=intent_result,
        summary=summary_result,
    )
    saved_path = save_session_note(outputs_dir, session_note)

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
        "--groq-model",
        default=None,
        help="Groq model to use for intent parsing. Defaults to GROQ_MODEL or llama-3.1-8b-instant.",
    )
    parser.add_argument(
        "--skip-intent",
        action="store_true",
        help="Deprecated alias for --skip-llm.",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Only record and transcribe audio; do not call Groq.",
    )
    parser.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Folder for session WAV and JSON files. Default: outputs.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process one recording and exit without loop prompt.",
    )
    args = parser.parse_args()
    load_dotenv()
    args.skip_llm = args.skip_llm or args.skip_intent

    while True:
        process_recording(args)
        if args.once:
            break

        choice = input("\nPress Enter to record again, or type q to quit: ").strip().lower()
        if choice == "q":
            break


if __name__ == "__main__":
    main()
