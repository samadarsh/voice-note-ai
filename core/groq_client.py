import os

from groq import Groq


_client: Groq | None = None
_client_api_key: str | None = None


def get_groq_client() -> Groq:
    global _client, _client_api_key

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY. Add it to a .env file or export it in your shell.")

    if _client is None or _client_api_key != api_key:
        _client = Groq(api_key=api_key)
        _client_api_key = api_key

    return _client
