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
- cleaned_transcript may lightly fix grammar, fillers, repeated words, and
  broken flow while preserving meaning.
- Cleaned transcript must preserve original meaning. If translating, ensure
  semantic accuracy.
- Preserve named entities and correct them using the closest known real-world
  term. Do not invent unrelated words.
- Do not distort proper nouns, hobbies, education names, or platform names.
- Correct phonetic or spoken variations to real terms when confidence is high,
  such as "சுடோக்கு" to "Sudoku" and "link-don" to "LinkedIn".
- Use personal_note for personal introductions, including name, education,
  role, hobbies, interests, or platform activity.
- If the transcript is clear and structured, such as a personal introduction,
  set confidence to high.
- For personal introductions, structure context with name, role, and interests
  as an array when present.
- For personal introductions, separate name, education or role, and interests
  clearly in the key points.
- Keep Tamil/Tanglish text in the original script when that is more faithful;
  if translating to English, preserve the exact meaning.
- For Tanglish/code-mixed speech, normalize meaning using language
  understanding instead of fixed word replacements.
- The summary short_summary should be in English by default.
- Do not invent details.
- Extract action_items only when the user mentions something to do or a next step.
- For missing information, include it as context metadata or a conditional action item.

Tamil emotion guidance:
- கோவம் / கோவமா வருது means anger or feeling angry.
- சோகம் / வருத்தமா இருக்கு means sadness or feeling sad.
- சந்தோஷம் means happiness.
- Do not convert anger into sadness.

Example:
Transcript: என் பெயரு ஆதஸ் நான் ஒரு M.Tech AI student எனக்கு மிகவும் பிடிச்ச விஷயம் சுடோக்கு விளையாடுவது, செஸ் விளையாடுவது, link-donல் போஸ்ட் போடுவது
Output:
{{
  "intent": {{
    "intent": "personal_note",
    "subject": "personal introduction",
    "content_type": "note",
    "language_detected": "multilingual",
    "context": {{
      "name": "Adas",
      "role": "M.Tech AI student",
      "interests": [
        "Sudoku",
        "Chess",
        "Posting on LinkedIn"
      ]
    }},
    "confidence": "high",
    "raw_transcript": "என் பெயரு ஆதஸ் நான் ஒரு M.Tech AI student எனக்கு மிகவும் பிடிச்ச விஷயம் சுடோக்கு விளையாடுவது, செஸ் விளையாடுவது, link-donல் போஸ்ட் போடுவது",
    "cleaned_transcript": "My name is Adas. I am an M.Tech AI student. I like playing Sudoku, playing chess, and posting on LinkedIn."
  }},
  "summary": {{
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
