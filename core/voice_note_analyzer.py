import json
import os
from typing import Any

from core.groq_client import get_groq_client
from core.intent_parser import INTENT_SCHEMA_KEYS, normalize_intent_result
from core.note_summarizer import SUMMARY_SCHEMA_KEYS, normalize_summary_result
from core.text_utils import extract_json


class VoiceNoteAnalysisError(RuntimeError):
    """Raised when Groq cannot return a complete voice-note analysis."""


ANALYSIS_PROMPT = f"""
You analyze multilingual voice notes for Voice Note AI.

The user may speak in English, Tamil, Tanglish, or a mix. Return JSON only.
Do not include markdown or explanation.

Return exactly this top-level shape:
{{
  "intent": {{ ... }},
  "summary": {{ ... }}
}}

The "intent" object must include exactly these keys:
{", ".join(INTENT_SCHEMA_KEYS)}

The "summary" object must include exactly these keys:
{", ".join(SUMMARY_SCHEMA_KEYS)}

Intent values:
- information_query
- personal_note
- task_or_reminder
- idea
- meeting_note
- learning_note
- translation_request
- general_conversation
- unknown

Rules:
- Preserve raw_transcript exactly as provided.
- cleaned_transcript may lightly fix grammar, fillers, repeated words, and broken flow while preserving meaning.
- Cleaned transcript must preserve original meaning. If translating, ensure semantic accuracy.
- Correct phonetic or spoken variations to real terms when confidence is high (e.g., mispronounced proper nouns or tech terms).
- For Tanglish/code-mixed speech, normalize meaning using language understanding.
- Keep Tamil/Tanglish text in the original script when that is more faithful; if translating to English, preserve the exact meaning.
- Identify the emotions, intent, and overall context to appropriately set the `confidence`, `content_type`, and `subject` fields.
- Extract any relevant structural information from the transcript and place it in the `context` dictionary.
- The `short_summary` should be in English by default.
- Do not invent details or assume information not present in the transcript.
- Extract `action_items` only when the user mentions something to do or a next step.
- For missing information, include it as context metadata or a conditional action item.

Example:
Transcript: I need to buy milk tomorrow and also remind me to call mom
Output:
{{
  "intent": {{
    "intent": "task_or_reminder",
    "subject": "Buy milk and call mom",
    "content_type": "reminder",
    "language_detected": "English",
    "context": {{
      "timeframe": "tomorrow"
    }},
    "confidence": "high",
    "raw_transcript": "I need to buy milk tomorrow and also remind me to call mom",
    "cleaned_transcript": "I need to buy milk tomorrow. Also, remind me to call mom."
  }},
  "summary": {{
    "cleaned_transcript": "I need to buy milk tomorrow. Also, remind me to call mom.",
    "short_summary": "The user needs a reminder to buy milk tomorrow and call their mom.",
    "key_points": [
      "Buy milk tomorrow",
      "Call mom"
    ],
    "action_items": [
      "Buy milk",
      "Call mom"
    ],
    "important_entities": [
      "mom"
    ],
    "language_detected": "English",
    "suggested_title": "Chores and Reminders"
  }}
}}
"""


def analyze_note(transcript: str, model: str | None = None) -> dict[str, Any]:
    client = get_groq_client()
    model_name = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    user_content = {
        "raw_transcript": transcript,
    }

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": ANALYSIS_PROMPT},
                {"role": "user", "content": json.dumps(user_content, ensure_ascii=False)},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        data = extract_json(content)
        intent = normalize_intent_result(data.get("intent") or {}, transcript)
        summary = normalize_summary_result(data.get("summary") or {}, transcript)
        return {"intent": intent, "summary": summary}
    except json.JSONDecodeError as exc:
        raise VoiceNoteAnalysisError("Groq returned analysis output that was not valid JSON.") from exc
    except Exception as exc:
        raise VoiceNoteAnalysisError(f"Groq voice-note analysis failed: {exc}") from exc
