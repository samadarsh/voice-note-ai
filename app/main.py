# app/main.py
# Entry point — starts the ASR + Transliteration Gradio app

import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.interface import build_interface
from app.utils import ensure_dirs
from models.model_config import APP_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 50)
    logger.info("  VoiceNote AI — Transliteration")
    logger.info("  Model  : openai/whisper-medium")
    logger.info("  Output : Tamil transcript + ASCII romanization")
    logger.info("=" * 50)

    ensure_dirs()
    logger.info("Building Gradio interface...")
    demo = build_interface()

    logger.info(
        f"Launching app on "
        f"http://{APP_CONFIG['host']}:{APP_CONFIG['port']}"
    )

    demo.launch(
        server_name=APP_CONFIG["host"],
        server_port=APP_CONFIG["port"],
        share=APP_CONFIG["share"],
    )


if __name__ == "__main__":
    main()
