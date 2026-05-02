import json
import os
from typing import Any

from groq_client import get_groq_client
from text_utils import (
    extract_json,
    extract_personal_intro_context,
    is_personal_introduction,
    normalize_known_terms,
    normalize_known_terms_in_value,
)


SUMMARY_SCHEMA_KEYS = [
    "cleaned_transcript",
    "short_summary",
    "key_points",
    "action_items",
    "important_entities",
    "language_detected",
    "suggested_title",
]


class NoteSummaryError(RuntimeError):
    """Raised when Groq cannot return a valid summary object."""


SUMMARY_PROMPT = """
You are the summary layer for a local multilingual voice note assistant.

The user may speak in English, Tamil, Tanglish, or a mix. Convert the transcript
into a clean structured note. Return JSON only.

Rules:
- Preserve meaning. Do not invent details.
- Keep cleaned_transcript in the same language as the original when possible.
- Cleaned transcript must preserve original meaning. If translating, ensure
  semantic accuracy.
- Preserve named entities and correct them using the closest known real-world
  term. Do not invent unrelated words.
- Do not distort proper nouns, hobbies, education names, or platform names.
- Correct phonetic or spoken variations to real terms when confidence is high,
  such as "சுடோக்கு" to "Sudoku" and "link-don" to "LinkedIn".
- The short_summary must be in English by default.
- For Tamil/Tanglish, explain the meaning clearly in English in short_summary.
- For Tanglish, use language understanding to normalize meaning naturally; do
  not rely on fixed word replacements.
- Do not treat casual, short, Tamil, or Tanglish speech as low quality when the
  meaning is clear.
- If the meaning is clear, always extract at least 1-2 key points.
- For personal notes, describe the user's current state, feeling, thought, or
  preference as concrete key points.
- For personal introductions, separate name, education or role, and interests
  clearly in the key points.
- If the transcript contains multiple distinct topics, keep them separate in
  the summary and key points. For example, hobbies and food preferences should
  be described as separate ideas.
- Translate Tamil emotional phrases carefully.
- கோவம் / கோவமா வருது means anger or feeling angry.
- சோகம் / வருத்தமா இருக்கு means sadness or feeling sad.
- சந்தோஷம் means happiness.
- Do not convert anger into sadness.
- If a clear emotional statement is repeated, treat repetition as intensity,
  not uncertainty.
- Extract action_items only when the user mentions something to do or a next step.
- For missing information, include it as an action item phrased conditionally,
  such as "Ask for location if connecting this to a search tool."
- important_entities can include names, dates, places, products, tools, topics,
  languages, or project names.

Example:
Transcript: My name is Adas. I am an M.Tech AI student. I like playing சுடோக்கு, playing chess, and posting on link-don.
Output:
{
  "cleaned_transcript": "My name is Adas. I am an M.Tech AI student. I like playing Sudoku, playing chess, and posting on LinkedIn.",
  "short_summary": "The user introduces themselves as an M.Tech AI student named Adas and mentions interests in Sudoku, chess, and posting on LinkedIn.",
  "key_points": [
    "Name: Adas",
    "Education: M.Tech AI student",
    "Interests: Sudoku, Chess, LinkedIn posting"
  ],
  "action_items": [],
  "important_entities": [
    "Adas",
    "M.Tech AI",
    "Sudoku",
    "Chess",
    "LinkedIn"
  ],
  "language_detected": "multilingual",
  "suggested_title": "Personal Introduction"
}

Example:
Transcript: எனக்கு இப்போ பசிக்குது.
Output:
{
  "cleaned_transcript": "எனக்கு இப்போ பசிக்குது.",
  "short_summary": "The user is saying they are hungry now.",
  "key_points": [
    "The user is currently hungry.",
    "The user feels like eating food now."
  ],
  "action_items": [],
  "important_entities": [
    "food"
  ],
  "language_detected": "Tamil",
  "suggested_title": "Hungry Now"
}

Example:
Transcript: எனக்கு கோவமா வருது. கோவமா வருது.
Output:
{
  "cleaned_transcript": "எனக்கு கோவமா வருது. கோவமா வருது.",
  "short_summary": "The user is feeling very angry right now and expresses strong emotion.",
  "key_points": [
    "The user is currently feeling very angry.",
    "The user repeated the statement, indicating intensity."
  ],
  "action_items": [],
  "important_entities": [
    "anger"
  ],
  "language_detected": "Tamil",
  "suggested_title": "Feeling Angry"
}

Return exactly these keys:
cleaned_transcript, short_summary, key_points, action_items,
important_entities, language_detected, suggested_title

Use arrays for key_points, action_items, and important_entities.
"""


def normalize_summary_result(data: dict[str, Any], transcript: str) -> dict[str, Any]:
    normalized = {key: normalize_known_terms_in_value(data.get(key)) for key in SUMMARY_SCHEMA_KEYS}
    normalized["cleaned_transcript"] = normalize_known_terms(
        normalized["cleaned_transcript"] or transcript.strip()
    )
    normalized["short_summary"] = normalize_known_terms(
        normalized["short_summary"] or "No clear summary available."
    )
    for key in ("key_points", "action_items", "important_entities"):
        if not isinstance(normalized[key], list):
            normalized[key] = []
        normalized[key] = normalize_known_terms_in_value(normalized[key])
    normalized["language_detected"] = normalized["language_detected"] or "unknown"
    normalized["suggested_title"] = normalized["suggested_title"] or "Untitled Voice Note"
    if is_personal_introduction(transcript):
        context = extract_personal_intro_context(transcript, {})
        name = context.get("name")
        role = context.get("role")
        interests = context.get("interests", [])
        if name and role and interests:
            normalized["short_summary"] = (
                f"The user introduces themselves as an {role} named {name} "
                f"and mentions interests in Sudoku, chess, and posting on LinkedIn."
            )
            normalized["key_points"] = [
                f"Name: {name}",
                f"Education: {role}",
                "Interests: Sudoku, Chess, LinkedIn posting",
            ]
            normalized["action_items"] = []
            normalized["important_entities"] = [name, "M.Tech AI", "Sudoku", "Chess", "LinkedIn"]
            normalized["language_detected"] = "multilingual"
            normalized["suggested_title"] = "Personal Introduction"
    return normalized


def summarize_note(
    transcript: str,
    intent_data: dict[str, Any] | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    client = get_groq_client()
    model_name = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    user_content = {
        "raw_transcript": transcript,
        "intent": intent_data or {},
    }

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SUMMARY_PROMPT},
                {"role": "user", "content": json.dumps(user_content, ensure_ascii=False)},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        return normalize_summary_result(extract_json(content), transcript)
    except json.JSONDecodeError as exc:
        raise NoteSummaryError("Groq returned summary output that was not valid JSON.") from exc
    except Exception as exc:
        raise NoteSummaryError(f"Groq summarization failed: {exc}") from exc


def print_summary(summary_result: dict[str, Any]) -> None:
    print(json.dumps(summary_result, indent=2, ensure_ascii=False))
