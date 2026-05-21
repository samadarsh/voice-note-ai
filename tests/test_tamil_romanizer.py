import unittest

from core.tamil_romanizer import tamil_to_ascii
from core.text_utils import contains_tamil_script
from core.transliteration import maybe_transliterate


class TamilRomanizerTests(unittest.TestCase):
    def test_engineer_example(self) -> None:
        tamil = "நான் ஒரு மென்பொருள் பொறியாளர்"
        result = tamil_to_ascii(tamil)
        self.assertEqual(result, "naan oru menporuL poRiyaaLar")
        self.assertTrue(result.isascii())

    def test_contains_tamil_script(self) -> None:
        self.assertTrue(contains_tamil_script("நான்"))
        self.assertFalse(contains_tamil_script("hello world"))

    def test_maybe_transliterate_skips_english(self) -> None:
        self.assertIsNone(maybe_transliterate("Buy milk tomorrow"))

    def test_maybe_transliterate_tamil(self) -> None:
        value = maybe_transliterate("நான் சாப்பிட்டேன்")
        self.assertIsNotNone(value)
        assert value is not None
        self.assertEqual(value, "naan saappittaen")


if __name__ == "__main__":
    unittest.main()
