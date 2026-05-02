import unittest

from intent_parser import normalize_intent_result
from note_summarizer import normalize_summary_result


INTRO_TRANSCRIPT = (
    "என் பெயரு ஆதஸ் நான் ஒரு M.Tech AI student எனக்கு மிகவும் பிடிச்ச விஷயம் "
    "சுடோக்கு விளையாடுவது, செஸ் விளையாடுவது, link-donல் போஸ்ட் போடுவது"
)


class NormalizationTests(unittest.TestCase):
    def test_personal_intro_intent_is_corrected(self) -> None:
        result = normalize_intent_result(
            {
                "intent": "general_conversation",
                "subject": "Personal Introduction",
                "content_type": "personal_note",
                "language_detected": "Tamil",
                "context": {
                    "name": "ஆதஸ்",
                    "interests": "Chess, Link-don, Stooges",
                },
                "confidence": "low",
                "cleaned_transcript": (
                    "My name is Adas. I'm an M.Tech AI student. "
                    "I'm very interested in playing stooges, playing chess, "
                    "and posting on link-don."
                ),
            },
            INTRO_TRANSCRIPT,
        )

        self.assertEqual(result["intent"], "personal_note")
        self.assertEqual(result["subject"], "personal introduction")
        self.assertEqual(result["content_type"], "note")
        self.assertEqual(result["confidence"], "high")
        self.assertEqual(result["context"]["name"], "Adas")
        self.assertEqual(result["context"]["role"], "M.Tech AI student")
        self.assertEqual(result["context"]["interests"], ["Sudoku", "Chess", "Posting on LinkedIn"])
        self.assertEqual(
            result["cleaned_transcript"],
            "My name is Adas. I am an M.Tech AI student. "
            "I like playing Sudoku, playing chess, and posting on LinkedIn.",
        )

    def test_personal_intro_summary_is_corrected(self) -> None:
        result = normalize_summary_result(
            {
                "cleaned_transcript": (
                    "My name is Adas. I'm an M.Tech AI student. "
                    "I'm very interested in playing stooges, playing chess, "
                    "and posting on link-don."
                ),
                "short_summary": "Adas is an M.Tech AI student interested in stooges, chess, and link-don.",
                "key_points": [
                    "Adas is an M.Tech AI student",
                    "Interested in stooges, chess, and link-don",
                ],
                "action_items": [],
                "important_entities": ["Adas", "Stooges", "Chess", "Link-don"],
                "language_detected": "Tamil",
                "suggested_title": "Personal Introduction",
            },
            INTRO_TRANSCRIPT,
        )

        self.assertEqual(
            result["short_summary"],
            "The user introduces themselves as an M.Tech AI student named Adas "
            "and mentions interests in Sudoku, chess, and posting on LinkedIn.",
        )
        self.assertEqual(
            result["key_points"],
            [
                "Name: Adas",
                "Education: M.Tech AI student",
                "Interests: Sudoku, Chess, LinkedIn posting",
            ],
        )
        self.assertEqual(result["action_items"], [])
        self.assertEqual(result["important_entities"], ["Adas", "M.Tech AI", "Sudoku", "Chess", "LinkedIn"])


if __name__ == "__main__":
    unittest.main()
