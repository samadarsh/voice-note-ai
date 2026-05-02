import json
import os
from typing import Any

from groq_client import get_groq_client
from text_utils import contains_indic_text, extract_json


INTENT_SCHEMA_KEYS = [
    "intent",
    "subject",
    "content_type",
    "language_detected",
    "context",
    "confidence",
    "raw_transcript",
    "cleaned_transcript",
]


class IntentParsingError(RuntimeError):
    """Raised when Groq cannot return a valid intent object."""


SYSTEM_PROMPT = """
You classify spoken voice notes for a Voice-to-Summary and Intent Note Assistant.

The system does not execute real-world actions. It only understands, cleans,
summarizes, and saves structured notes.

Return JSON only. Do not include markdown or explanation.

Allowed intent values:
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
- If Tamil or Tanglish is present, keep the original transcript exact.
- For Tamil/Tanglish, use language understanding to normalize clear code-mixed
  words in cleaned_transcript only when it improves readability and preserves
  meaning. Do not rely on literal word substitution.
- Do not include execution instructions, API routing, confirmation, or
  clarification fields.
- Location-dependent requests are still notes. Mention missing location only as
  metadata inside context, not as an action to perform.

Return exactly these keys:
intent, subject, content_type, language_detected, context, confidence,
raw_transcript, cleaned_transcript

Field guidance:
- intent: one of the allowed values.
- subject: short phrase describing what the speech is about.
- content_type: note, question, task, idea, meeting, learning, translation, chat, unknown.
- language_detected: English, Tamil, Tanglish, multilingual, or unknown.
- context: extra metadata such as missing_location, language_style, transcript_quality.
- confidence: high, medium, or low.

Intent guidance:
- Use personal_note when the user states facts about themselves, their
  background, interests, plans, preferences, feelings, or experience.
- Use information_query only when the user is asking for information, such as
  "what is", "show me", "tell me", "find", "search", or "explain".

Tamil emotion guidance:
- Translate Tamil emotional phrases carefully.
- கோவம் / கோவமா வருது means anger or feeling angry.
- சோகம் / வருத்தமா இருக்கு means sadness or feeling sad.
- சந்தோஷம் means happiness.
- Do not convert anger into sadness.

Confidence guidance:
- Use high when the transcript is clear and the user's meaning is understandable.
- Do not lower confidence just because the sentence is casual, short, Tamil, or Tanglish.
- Do not lower confidence because the sentence is repeated, emotional, or informal.
- Use medium for understandable but phonetically noisy transcripts.
- Use low only when the transcript is unclear or noisy, words are missing or
  incorrect, or the meaning is genuinely ambiguous.

Example:
Transcript: I want to build a small voice assistant project using Whisper and LLMs.
Output:
{
  "intent": "idea",
  "subject": "voice assistant project using Whisper and LLMs",
  "content_type": "idea",
  "language_detected": "English",
  "context": {},
  "confidence": "high",
  "raw_transcript": "I want to build a small voice assistant project using Whisper and LLMs.",
  "cleaned_transcript": "I want to build a small voice assistant project using Whisper and LLMs."
}

Example:
Transcript: I am a computer science student and I am interested in machine learning.
Output:
{
  "intent": "personal_note",
  "subject": "personal background and ML interests",
  "content_type": "note",
  "language_detected": "English",
  "context": {},
  "confidence": "high",
  "raw_transcript": "I am a computer science student and I am interested in machine learning.",
  "cleaned_transcript": "I am a computer science student and I am interested in machine learning."
}

Example:
Transcript: எனக்கு இப்போ பசிக்குது.
Output:
{
  "intent": "personal_note",
  "subject": "feeling hungry now",
  "content_type": "note",
  "language_detected": "Tamil",
  "context": {
    "current_state": "hungry"
  },
  "confidence": "high",
  "raw_transcript": "எனக்கு இப்போ பசிக்குது.",
  "cleaned_transcript": "எனக்கு இப்போ பசிக்குது."
}

Example:
Transcript: எனக்கு கோவமா வருது.
Output:
{
  "intent": "personal_note",
  "subject": "feeling angry now",
  "content_type": "note",
  "language_detected": "Tamil",
  "context": {
    "current_state": "angry"
  },
  "confidence": "high",
  "raw_transcript": "எனக்கு கோவமா வருது.",
  "cleaned_transcript": "எனக்கு கோவமா வருது."
}

Example:
Transcript: Remind me to call Arun tomorrow at 5 pm.
Output:
{
  "intent": "task_or_reminder",
  "subject": "call Arun tomorrow at 5 pm",
  "content_type": "task",
  "language_detected": "English",
  "context": {
    "time_reference": "tomorrow at 5 pm"
  },
  "confidence": "high",
  "raw_transcript": "Remind me to call Arun tomorrow at 5 pm.",
  "cleaned_transcript": "Remind me to call Arun tomorrow at 5 pm."
}

Example:
Transcript: எனக்கு நியராச்சியாக இருக்கும் பஜ்செட் ஃபிரெண்ட்லி ஐஸ்க்ரிம் சாப்பிட்டு சொல்ல முடியுமா?
Normalized hint: எனக்கு near-ah இருக்கும் budget-friendly ice cream shop சொல்ல முடியுமா?
Output:
{
  "intent": "information_query",
  "subject": "nearby budget-friendly ice cream shops",
  "content_type": "question",
  "language_detected": "Tanglish",
  "context": {
    "missing_location": true,
    "language_style": "tanglish",
    "transcript_quality": "noisy_phonetic"
  },
  "confidence": "medium",
  "raw_transcript": "எனக்கு நியராச்சியாக இருக்கும் பஜ்செட் ஃபிரெண்ட்லி ஐஸ்க்ரிம் சாப்பிட்டு சொல்ல முடியுமா?",
  "cleaned_transcript": "எனக்கு near-ah இருக்கும் budget-friendly ice cream shop சொல்ல முடியுமா?"
}
"""


def _detect_clear_tamil_emotion(transcript: str) -> str | None:
    if "கோவ" in transcript or "கோப" in transcript:
        return "angry"
    if "சோகம்" in transcript or "வருத்தமா" in transcript:
        return "sad"
    if "சந்தோஷ" in transcript:
        return "happy"
    return None


def normalize_intent_result(
    data: dict[str, Any],
    transcript: str,
    normalized_transcript: str | None = None,
) -> dict[str, Any]:
    cleaned_transcript = normalized_transcript or transcript.strip()
    normalized = {key: data.get(key) for key in INTENT_SCHEMA_KEYS}
    normalized["raw_transcript"] = transcript
    if contains_indic_text(transcript) and normalized_transcript:
        normalized["cleaned_transcript"] = cleaned_transcript
    else:
        normalized["cleaned_transcript"] = normalized["cleaned_transcript"] or transcript.strip()
    if not isinstance(normalized["context"], dict):
        normalized["context"] = {}
    if contains_indic_text(transcript) and cleaned_transcript != transcript:
        normalized["context"].setdefault("language_style", "tanglish")
        normalized["context"].setdefault("transcript_quality", "noisy_phonetic")
    tamil_emotion = _detect_clear_tamil_emotion(transcript)
    if tamil_emotion:
        normalized["intent"] = normalized["intent"] or "personal_note"
        normalized["subject"] = normalized["subject"] or f"feeling {tamil_emotion} now"
        normalized["context"].setdefault("current_state", tamil_emotion)
        normalized["confidence"] = "high"
    if not normalized["content_type"]:
        normalized["content_type"] = "unknown"
    if not normalized["language_detected"]:
        normalized["language_detected"] = "unknown"
    if normalized["confidence"] not in {"high", "medium", "low"}:
        normalized["confidence"] = "medium" if cleaned_transcript != transcript else "low"
    return normalized


def parse_intent(transcript: str, model: str | None = None) -> dict[str, Any]:
    client = get_groq_client()
    model_name = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    normalized_transcript = None
    user_content = f"Transcript: {transcript}"

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        return normalize_intent_result(extract_json(content), transcript)
    except json.JSONDecodeError as exc:
        raise IntentParsingError("Groq returned intent output that was not valid JSON.") from exc
    except Exception as exc:
        raise IntentParsingError(f"Groq intent parsing failed: {exc}") from exc


def print_intent(intent_result: dict[str, Any]) -> None:
    print(json.dumps(intent_result, indent=2, ensure_ascii=False))
