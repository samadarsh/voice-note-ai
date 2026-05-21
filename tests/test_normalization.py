import unittest

from core.intent_parser import normalize_intent_result
from core.note_summarizer import normalize_summary_result


INTRO_TRANSCRIPT = (
    "என் பெயரு ஆதஸ் நான் ஒரு M.Tech AI student எனக்கு மிகவும் பிடிச்ச விஷயம் "
    "சுடோக்கு விளையாடுவது, செஸ் விளையாடுவது, link-donல் போஸ்ட் போடுவது"
)


class NormalizationTests(unittest.TestCase):
    def test_intent_preserves_fields_and_raw_transcript(self) -> None:
        result = normalize_intent_result(
            {
                "intent": "personal_note",
                "subject": "Personal Introduction",
                "content_type": "note",
                "language_detected": "Tamil",
                "context": {"name": "Adas"},
                "confidence": "high",
                "cleaned_transcript": "My name is Adas.",
            },
            INTRO_TRANSCRIPT,
        )

        self.assertEqual(result["intent"], "personal_note")
        self.assertEqual(result["raw_transcript"], INTRO_TRANSCRIPT)
        self.assertEqual(result["context"]["name"], "Adas")
        self.assertEqual(result["confidence"], "high")

    def test_intent_fills_missing_defaults(self) -> None:
        result = normalize_intent_result({}, "hello")
        self.assertEqual(result["raw_transcript"], "hello")
        self.assertEqual(result["content_type"], "unknown")
        self.assertEqual(result["language_detected"], "unknown")
        self.assertEqual(result["confidence"], "medium")

    def test_summary_fills_list_defaults(self) -> None:
        result = normalize_summary_result(
            {"short_summary": "A short note."},
            INTRO_TRANSCRIPT,
        )
        self.assertEqual(result["short_summary"], "A short note.")
        self.assertEqual(result["key_points"], [])
        self.assertEqual(result["action_items"], [])
        self.assertEqual(result["suggested_title"], "Untitled Voice Note")


if __name__ == "__main__":
    unittest.main()
