import os
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from groq import Groq

_client: "Groq | None" = None
_client_api_key: str | None = None

_PLACEHOLDER_PATTERNS = (
    "your_groq_api_key",
    "changeme",
    "replace_me",
    "insert",
    "api_key_here",
)

INVALID_GROQ_KEY_HELP = (
    "Invalid Groq API key. Get a key at https://console.groq.com/keys and set it in "
    "a `.env` file as GROQ_API_KEY=gsk_... (do not use the placeholder from .env.example). "
    "Restart Streamlit after saving `.env`."
)


def get_groq_api_key() -> str | None:
    return os.getenv("GROQ_API_KEY")


def is_configured_groq_api_key(api_key: str | None = None) -> bool:
    key = (api_key or get_groq_api_key() or "").strip()
    if not key:
        return False
    lower = key.lower()
    if any(marker in lower for marker in _PLACEHOLDER_PATTERNS):
        return False
    # Groq keys are typically gsk_ + alphanumeric
    if not re.match(r"^gsk_[A-Za-z0-9]{20,}$", key):
        return False
    return True


def groq_api_key_status() -> tuple[bool, str]:
    key = get_groq_api_key()
    if not key or not key.strip():
        return False, "Missing GROQ_API_KEY"
    if not is_configured_groq_api_key(key):
        return False, "GROQ_API_KEY looks like a placeholder or invalid format (expected gsk_...)"
    return True, f"Configured ({key[:8]}…{key[-4:]})"


def get_groq_client() -> "Groq":
    global _client, _client_api_key
    from groq import Groq

    api_key = get_groq_api_key()
    if not api_key or not api_key.strip():
        raise RuntimeError(
            "Missing GROQ_API_KEY. Add it to a `.env` file in the project root or Streamlit secrets."
        )
    if not is_configured_groq_api_key(api_key):
        raise RuntimeError(INVALID_GROQ_KEY_HELP)

    api_key = api_key.strip()
    if _client is None or _client_api_key != api_key:
        _client = Groq(api_key=api_key)
        _client_api_key = api_key

    return _client


def is_groq_auth_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        "401" in text
        or "invalid_api_key" in text
        or "invalid api key" in text
        or "authentication" in text
    )
