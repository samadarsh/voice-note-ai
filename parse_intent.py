from argparse import ArgumentParser

from intent_parser import parse_intent, print_intent


def main() -> None:
    parser = ArgumentParser(description="Parse transcribed text into a universal intent object.")
    parser.add_argument("transcript", help="Text transcript to parse.")
    parser.add_argument(
        "--groq-model",
        default=None,
        help="Groq model to use. Defaults to GROQ_MODEL or llama-3.1-8b-instant.",
    )
    args = parser.parse_args()

    intent_result = parse_intent(args.transcript, args.groq_model)
    print_intent(intent_result)


if __name__ == "__main__":
    main()
