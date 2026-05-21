import unittest

from core.groq_client import is_configured_groq_api_key, is_groq_auth_error


class GroqClientTests(unittest.TestCase):
    def test_rejects_placeholder(self) -> None:
        self.assertFalse(is_configured_groq_api_key("your_groq_api_key_here"))

    def test_accepts_gsk_format(self) -> None:
        self.assertTrue(
            is_configured_groq_api_key("gsk_" + "a" * 40)
        )

    def test_detects_auth_error(self) -> None:
        err = Exception(
            "Error code: 401 - {'error': {'message': 'Invalid API Key', 'code': 'invalid_api_key'}}"
        )
        self.assertTrue(is_groq_auth_error(err))


if __name__ == "__main__":
    unittest.main()
