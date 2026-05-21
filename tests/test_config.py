import unittest

from config import resolve_whisper_model, WHISPER_MODEL_DEFAULT, WHISPER_MODEL_TAMIL


class ConfigTests(unittest.TestCase):
    def test_resolve_tamil_upgrades_default(self) -> None:
        self.assertEqual(
            resolve_whisper_model(WHISPER_MODEL_DEFAULT, "ta"),
            WHISPER_MODEL_TAMIL,
        )

    def test_resolve_explicit_model_unchanged(self) -> None:
        self.assertEqual(resolve_whisper_model("tiny", "ta"), "tiny")


if __name__ == "__main__":
    unittest.main()
