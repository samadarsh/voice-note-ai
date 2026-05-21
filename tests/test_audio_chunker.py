import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.audio_chunker import should_chunk, validate_audio_file


class AudioChunkerTests(unittest.TestCase):
    def test_validate_missing_file(self) -> None:
        self.assertFalse(validate_audio_file(Path("/nonexistent/file.wav")))

    @patch("core.audio_chunker.get_duration_sec", return_value=45.0)
    def test_should_chunk_long_audio(self, _mock: MagicMock) -> None:
        with patch("core.audio_chunker.validate_audio_file", return_value=True):
            self.assertTrue(should_chunk(Path("fake.wav"), 30))

    @patch("core.audio_chunker.get_duration_sec", return_value=10.0)
    def test_should_not_chunk_short_audio(self, _mock: MagicMock) -> None:
        with patch("core.audio_chunker.validate_audio_file", return_value=True):
            self.assertFalse(should_chunk(Path("fake.wav"), 30))


if __name__ == "__main__":
    unittest.main()
