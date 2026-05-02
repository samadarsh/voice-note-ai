import json
import os
from typing import Any

from dotenv import load_dotenv
from groq import Groq

from intent_parser import _extract_json, normalize_tanglish


SUMMARY_SCHEMA_KEYS = [
    "cleaned_transcript",
    "short_summary",
    "key_points",
    "action_items",
    "important_entities",
    "language_detected",
    "suggested_title",
]


SUMMARY_PROMPT = """
You are the summary layer for a local multilingual voice note assistant.

The user may speak in English, Tamil, Tanglish, or a mix. Convert the transcript
into a clean structured note. Return JSON only.

Rules:
- Preserve meaning. Do not invent details.
- Keep cleaned_transcript in the same language as the original when possible.
- The short_summary must be in English by default.
- For Tamil/Tanglish, explain the meaning clearly in English in short_summary.
- Do not treat casual, short, Tamil, or Tanglish speech as low quality when the
  meaning is clear.
- If the meaning is clear, always extract at least 1-2 key points.
- For personal notes, describe the user's current state, feeling, thought, or
  preference as concrete key points.
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


def _normalize_summary(data: dict[str, Any], transcript: str) -> dict[str, Any]:
    normalized = {key: data.get(key) for key in SUMMARY_SCHEMA_KEYS}
    normalized["cleaned_transcript"] = normalized["cleaned_transcript"] or transcript.strip()
    normalized["short_summary"] = normalized["short_summary"] or "No clear summary available."
    for key in ("key_points", "action_items", "important_entities"):
        if not isinstance(normalized[key], list):
            normalized[key] = []
    normalized["language_detected"] = normalized["language_detected"] or "unknown"
    normalized["suggested_title"] = normalized["suggested_title"] or "Untitled Voice Note"
    return normalized


def summarize_note(
    transcript: str,
    intent_data: dict[str, Any] | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY. Add it to a .env file or export it in your shell.")

    client = Groq(api_key=api_key)
    model_name = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    normalized_transcript = normalize_tanglish(transcript)

    user_content = {
        "raw_transcript": transcript,
        "normalized_hint": normalized_transcript if normalized_transcript != transcript else None,
        "intent": intent_data or {},
    }

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
    return _normalize_summary(_extract_json(content), transcript)


def print_summary(summary_result: dict[str, Any]) -> None:
    print(json.dumps(summary_result, indent=2, ensure_ascii=False))
