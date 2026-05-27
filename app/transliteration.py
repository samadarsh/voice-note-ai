# app/transliteration.py
# Indic transliteration pipeline — Tamil script -> Romanized ASCII text.

import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.tamil_romanizer import tamil_to_ascii
from app.utils import save_transliteration
from models.model_config import TRANSLITERATION_CONFIG

logger = logging.getLogger(__name__)


class TransliterationPipeline:
    """
    Transliteration pipeline using a custom Tamil-aware Latin (ASCII)
    grapheme romanizer.
    """

    def __init__(self):
        self.src_script = TRANSLITERATION_CONFIG["src_script"]
        self.tgt_script = TRANSLITERATION_CONFIG["tgt_script"]
        logger.info(
            f"TransliterationPipeline initialized — "
            f"{self.src_script} → {self.tgt_script}"
        )

    def transliterate(self, text: str) -> str:
        if not text or not text.strip():
            logger.warning("Empty text received for transliteration")
            return ""

        logger.info(f"Transliterating: {text[:60]}...")
        try:
            result = tamil_to_ascii(text)
            logger.info(f"Transliteration result: {result[:60]}...")
            return result.strip()
        except Exception as e:
            logger.error(f"Transliteration failed: {e}")
            return text

    def transliterate_and_save(self, text: str) -> tuple:
        transliterated = self.transliterate(text)
        saved_path = save_transliteration(transliterated)
        return transliterated, saved_path

    def get_scheme_info(self) -> dict:
        return {
            "source_script": "Tamil (Unicode)",
            "target_scheme": "Tamil-Aware Latin (ASCII) Romanization",
            "library": "in-house grapheme mapping (app.tamil_romanizer)",
            "example_input": "நான் சாப்பிட்டேன்",
            "example_output": "naan saappittaen",
        }
