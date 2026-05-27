# tests/test_pipeline.py

import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.buffer_manager import BufferManager
from app.transliteration import TransliterationPipeline
from app.utils import get_timestamp, validate_audio_file, format_output, ensure_dirs


class TestBufferManager(unittest.TestCase):
    def setUp(self):
        self.buffer = BufferManager(max_queue_size=5)

    def test_add_chunk(self):
        result = self.buffer.add_chunk("chunk_1")
        self.assertTrue(result)
        self.assertEqual(self.buffer.size(), 1)

    def test_get_chunk(self):
        self.buffer.add_chunk("chunk_1")
        chunk = self.buffer.get_chunk(timeout=1.0)
        self.assertEqual(chunk, "chunk_1")


class TestTransliterationPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = TransliterationPipeline()

    def test_transliterate_sentence_ascii(self):
        result = self.pipeline.transliterate("நான் சாப்பிட்டேன்")
        self.assertIsInstance(result, str)
        self.assertTrue(result.isascii())


class TestUtils(unittest.TestCase):
    def test_get_timestamp(self):
        ts = get_timestamp()
        self.assertIsInstance(ts, str)
        self.assertEqual(len(ts), 15)

    def test_validate_audio_file_missing(self):
        result = validate_audio_file("nonexistent.wav")
        self.assertFalse(result)

    def test_format_output(self):
        output = format_output("நான்", "naan")
        self.assertIn("transcript", output)
        self.assertIn("transliteration", output)
        self.assertIn("timestamp", output)

    def test_ensure_dirs(self):
        ensure_dirs()
        self.assertTrue(os.path.exists("outputs/transcripts"))
        self.assertTrue(os.path.exists("outputs/transliterations"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
