"""
Groq-powered Speech Analyzer — the LLM layer of the TTS pipeline.

Purpose:
  Instead of treating text as static content, this module simulates how a human
  would THINK about delivering the text as speech:

  1. Humanize raw text into natural spoken form
     - Inject punctuation (... ! ?) where a human speaker would pause/emphasize
     - Break long clauses into conversational breath groups
     - Add optional fillers (you know, actually, look) for casual delivery
     - Preserve Hinglish / code-mixed language and Indian conversational rhythm

  2. Produce a per-segment prosody arc:
     - Each segment gets a small prosody DELTA (not absolute) on top of the base
     - Delta range: rate ±10%, pitch ±5Hz, volume ±2dB, pause 0–700ms
     - This creates natural micro-variation instead of robotic flat delivery

  3. Identify emphasis words for <emphasis> injection in SSML

  All output is validated and clamped; if any Groq call fails or times out,
  the module falls back to the existing rule-based pipeline.

Architecture position:
  Input → [THIS MODULE] → Text Humanization → Prosody Engine → SSML → TTS
"""
import json
import re
import logging
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import httpx

from app.core.config import settings


logger = logging.getLogger("empathy_engine")

GROQ_API_URL   = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL     = "llama-3.1-8b-instant"   # fast, low-latency for real-time TTS
GROQ_TIMEOUT   = 4.0                       # seconds — short so we always have a fallback

# ── Data Classes ───────────────────────────────────────────────────────────────

@dataclass
class SegmentAnalysis:
    """Prosody guidance for one text segment from the LLM."""
    text:          str
    emotion:       str = "neutral"
    emphasis_words: List[str] = field(default_factory=list)
    # Deltas applied ON TOP of the shared base prosody:
    rate_delta_pct:   int   = 0      # -10 to +10
    pitch_delta_hz:   int   = 0      # -5  to +5
    volume_delta_db:  float = 0.0    # -2.0 to +2.0
    pause_before_ms:  int   = 0      # 0 to 700


@dataclass
class SpeechAnalysis:
    """Full LLM analysis output for one synthesis request."""
    humanized_text:  str
    delivery_style:  str                       # casual | empathetic | authoritative | excited | somber
    filler:          str = ""                  # "you know" | "actually" | "" etc.
    segments:        List[SegmentAnalysis] = field(default_factory=list)
    llm_used:        bool = False
    error:           Optional[str] = None


# ── System prompt (terse — cheaper tokens, faster responses) ───────────────────

_SYSTEM = (
    "You are a TTS preprocessing expert. "
    "Rewrite text as natural SPOKEN dialogue. "
    "Output ONLY valid JSON. No explanation. No markdown."
)

_USER_TMPL = """Analyze for speech delivery:
TEXT: {text}
EMOTION: {emotion}
INTENSITY: {intensity}

Return this exact JSON (no other text):
{{
  "humanized_text": "...",
  "delivery_style": "casual|empathetic|authoritative|excited|somber|conversational",
  "filler": "",
  "segments": [
    {{
      "text": "segment text",
      "emotion": "emotion label",
      "emphasis_words": [],
      "rate_delta_pct": 0,
      "pitch_delta_hz": 0,
      "volume_delta_db": 0.0,
      "pause_before_ms": 0
    }}
  ]
}}

RULES:
- humanized_text: how a HUMAN would say it (add ... ! ? naturally, preserve Hinglish)
- filler: one of "you know","actually","look","I mean","acha" OR "" if not appropriate
- segments: one per sentence/clause, max 5
- rate_delta_pct: -10 to +10 (relative shift, NOT absolute)
- pitch_delta_hz: -5 to +5
- volume_delta_db: -2.0 to +2.0
- pause_before_ms: 0 to 700
- Vary the deltas naturally — do NOT use 0 for everything"""


# ── Public API ─────────────────────────────────────────────────────────────────

async def analyze_speech(
    text: str,
    emotion: str = "neutral",
    intensity: float = 1.0,
) -> SpeechAnalysis:
    """
    Run Groq LLM analysis. Falls back gracefully if API is unavailable.

    Returns a SpeechAnalysis with:
    - humanized_text: conversational rewrite of the input
    - segments: list of SegmentAnalysis with per-segment prosody deltas
    - delivery_style, filler
    """
    if not settings.GROQ_API_KEY:
        return _fallback(text, emotion, "No GROQ_API_KEY in .env")

    prompt = _USER_TMPL.format(
        text=text[:800],   # cap to keep tokens low + fast
        emotion=emotion,
        intensity=round(intensity, 2),
    )

    try:
        async with httpx.AsyncClient(timeout=GROQ_TIMEOUT) as client:
            resp = await client.post(
                GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {settings.GROQ_API_KEY}",
                    "Content-Type":  "application/json",
                },
                json={
                    "model":       GROQ_MODEL,
                    "messages":    [
                        {"role": "system", "content": _SYSTEM},
                        {"role": "user",   "content": prompt},
                    ],
                    "temperature":      0.4,
                    "max_tokens":       600,
                    "response_format":  {"type": "json_object"},
                },
            )

        if resp.status_code != 200:
            return _fallback(text, emotion, f"Groq HTTP {resp.status_code}")

        raw_json = resp.json()["choices"][0]["message"]["content"]
        return _parse_response(raw_json, text, emotion)

    except (httpx.TimeoutException, httpx.NetworkError) as e:
        return _fallback(text, emotion, f"Groq timeout/network: {e}")
    except Exception as e:
        logger.warning(f"SpeechAnalyzer Groq error (non-fatal): {e}")
        return _fallback(text, emotion, str(e))


# ── Parsing & validation ───────────────────────────────────────────────────────

def _parse_response(raw: str, original_text: str, emotion: str) -> SpeechAnalysis:
    """Parse Groq JSON response into a SpeechAnalysis, clamping all values."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown fences
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(1))
            except Exception:
                return _fallback(original_text, emotion, "JSON parse error")
        else:
            return _fallback(original_text, emotion, "JSON parse error")

    humanized = str(data.get("humanized_text", original_text)).strip() or original_text
    delivery  = str(data.get("delivery_style", "conversational"))
    filler    = str(data.get("filler", "")).strip()

    segments: List[SegmentAnalysis] = []
    for seg in (data.get("segments") or []):
        try:
            segments.append(SegmentAnalysis(
                text           = str(seg.get("text", "")).strip(),
                emotion        = str(seg.get("emotion", emotion)),
                emphasis_words = [str(w) for w in (seg.get("emphasis_words") or [])],
                rate_delta_pct = max(-10, min(10, int(seg.get("rate_delta_pct", 0)))),
                pitch_delta_hz = max(-5,  min(5,  int(seg.get("pitch_delta_hz", 0)))),
                volume_delta_db= max(-2.0,min(2.0, float(seg.get("volume_delta_db", 0)))),
                pause_before_ms= max(0,   min(700, int(seg.get("pause_before_ms", 0)))),
            ))
        except Exception:
            continue

    logger.info(
        f"SpeechAnalyzer [Groq]: style={delivery}, "
        f"filler='{filler}', segments={len(segments)}, "
        f"humanized_len={len(humanized)}"
    )

    return SpeechAnalysis(
        humanized_text = humanized,
        delivery_style = delivery,
        filler         = filler,
        segments       = segments,
        llm_used       = True,
    )


# ── Rule-based fallback ────────────────────────────────────────────────────────

def _fallback(text: str, emotion: str, reason: str) -> SpeechAnalysis:
    """
    Pure rule-based fallback — no LLM required.
    Produces a minimal but valid SpeechAnalysis for the existing pipeline.
    """
    logger.debug(f"SpeechAnalyzer fallback ({reason})")

    # Rule-based: split into sentences, assign slight prosody deltas based on position
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    segments  = []
    n         = len(sentences)

    for i, sent in enumerate(sentences[:5]):
        sent = sent.strip()
        if not sent:
            continue

        # Simple position-based arc: opening → closing
        pos_frac = i / max(1, n - 1)   # 0.0 = first, 1.0 = last
        # Natural arc: slight slowdown at end, slight pitch drop
        rate_d  = round(-2 * pos_frac)           # 0 → -2%
        pitch_d = round(-1 * pos_frac)            # 0 → -1Hz
        pause_ms = 200 if sent.endswith("...") else (150 if i > 0 else 0)

        segments.append(SegmentAnalysis(
            text            = sent,
            emotion         = emotion,
            emphasis_words  = [],
            rate_delta_pct  = rate_d,
            pitch_delta_hz  = pitch_d,
            volume_delta_db = 0.0,
            pause_before_ms = pause_ms,
        ))

    return SpeechAnalysis(
        humanized_text = text,
        delivery_style = "conversational",
        filler         = "",
        segments       = segments,
        llm_used       = False,
        error          = reason,
    )
