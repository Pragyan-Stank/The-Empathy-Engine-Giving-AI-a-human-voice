"""
Groq-powered Speech Analyzer — the LLM brain of the TTS pipeline.

This module prepares a speech performance:
  1. Detects the overall emotion (or honors an override) and confidence.
  2. Identifies a delivery style, tone arc, and intent.
  3. Breaks text into segments (breath groups) with specific rate, pitch, volume, and pause deltas.
  4. Identifies emphasis words.

If Groq fails, the module falls back to a rule-based pipeline powered by VADER sentiment.
"""
import json
import re
import logging
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import httpx

from app.core.config import settings
from app.services.text.language_detector import detect_language
from app.services.emotion.sentiment_fallback import VaderSentimentFallback

logger = logging.getLogger("empathy_engine")

GROQ_API_URL   = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL     = "llama-3.1-8b-instant"   # fast, low-latency for real-time TTS
GROQ_TIMEOUT   = 6.0                       # seconds

_vader = VaderSentimentFallback()

# ── Data Classes ───────────────────────────────────────────────────────────────

@dataclass
class SegmentAnalysis:
    """Prosody guidance for one text segment from the LLM."""
    text:          str
    emotion:       str = "neutral"
    emphasis_words: List[str] = field(default_factory=list)
    rate_delta_pct:   int   = 0      # -20 to +20
    pitch_delta_hz:   int   = 0      # -10 to +10
    volume_delta_db:  float = 0.0    # -4.0 to +4.0
    pause_before_ms:  int   = 0      # 0 to 900
    arc_position:     str   = "middle"  # opening | middle | climax | closing


@dataclass
class SpeechAnalysis:
    """Full LLM analysis output for one synthesis request."""
    detected_emotion: str
    confidence:      float
    delivery_style:  str                       # casual | empathetic | authoritative | excited | somber
    segments:        List[SegmentAnalysis] = field(default_factory=list)
    llm_used:        bool = False
    error:           Optional[str] = None
    tone_arc:        str = "steady"            # e.g. "slow_build", "peak_then_fade", "emotional_wave"
    intent:          str = "informational"     # informational | empathetic | persuasive | urgent | reflective


# ── System Prompts ─────────────────────────────────────────────────────────────

_SYSTEM_BASE = (
    "You are a TTS speech performance director. "
    "Your job is to analyze text and design a HUMAN SPEECH PERFORMANCE. "
    "Determine the overall emotion, confidence score, delivery style, intent, tone arc, and break "
    "the text into natural segments (breath groups) with specific rate, pitch, volume, and pause adjustments. "
    "Output ONLY valid JSON. No explanation. No markdown."
)

_SYSTEM_HINDI_CONSTRAINT = (
    "\n\n### ABSOLUTE LANGUAGE RULE ###\n"
    "The input text is in Hindi / Hinglish. You MUST follow these rules:\n"
    "1. NEVER translate ANY word to English. Not a single word.\n"
    "2. Output ALL segment texts in THE EXACT SAME LANGUAGE and script as the input.\n"
    "3. If input is in Devanagari (हिंदी), output MUST stay in Devanagari.\n"
    "4. If input is in Roman Hindi (Hinglish), output MUST stay in Roman Hindi.\n"
    "Violating language constraints is an absolute failure."
)

_SYSTEM_ENGLISH = (
    "\nCRITICAL: NEVER translate text to another language. "
    "If the input is in Hindi, Hinglish, or any Indian language, keep it in that EXACT language. "
    "Only improve spoken delivery rhythm, do NOT translate."
)


def _build_system_prompt(lang: str) -> str:
    if lang in ("hi", "hi-Latn"):
        return _SYSTEM_BASE + _SYSTEM_HINDI_CONSTRAINT
    return _SYSTEM_BASE + _SYSTEM_ENGLISH


_USER_TMPL = """Analyze for speech PERFORMANCE and EMOTION (not just reading):
TEXT: {text}
EMOTION_OVERRIDE: {emotion_override}
INTENSITY: {intensity}
DETECTED_LANGUAGE: {language}

Return this exact JSON format (no markdown blocks, no other text):
{{
  "detected_emotion": "overall dominant emotion (joy, sadness, anger, fear, surprise, neutral, excitement, contentment, grief, frustration, rage, anxiety, disgust) - MUST use EMOTION_OVERRIDE if it is not 'None'",
  "confidence": 0.95,
  "delivery_style": "casual|empathetic|authoritative|excited|somber|conversational|reflective",
  "intent": "informational|empathetic|persuasive|urgent|reflective",
  "tone_arc": "steady|slow_build|peak_then_fade|emotional_wave|building_urgency",
  "segments": [
    {{
      "text": "segment text IN THE EXACT SAME LANGUAGE as input (one phrase/clause/sentence)",
      "emotion": "emotion label for this segment",
      "emphasis_words": ["key", "words"],
      "arc_position": "opening|middle|climax|closing",
      "rate_delta_pct": 0,
      "pitch_delta_hz": 0,
      "volume_delta_db": 0.0,
      "pause_before_ms": 0
    }}
  ]
}}

MANDATORY RULES:
- *** LANGUAGE PRESERVATION IS THE #1 RULE ***
- All segments must match input language and script exactly.
- detected_emotion: Must match EMOTION_OVERRIDE if one is provided.
- segments: break text into natural clauses/sentences (maximum 8 segments).
- rate_delta_pct: VARY between -20 to +20 (NEVER all zeros — that's robotic).
- pitch_delta_hz: VARY between -10 to +10.
- volume_delta_db: VARY between -4.0 to +4.0.
- pause_before_ms: 0 to 900.
- emphasis_words: 1-3 key words per segment that need stress (in the input text language).
"""

# ── Emotion → Default Arc Profiles (Fallback) ─────────────────────────────────

_EMOTION_ARC = {
    "joy":         {"rate_range": (5, 15),  "pitch_range": (3, 8),   "vol_range": (1, 3),   "pause_range": (50, 200)},
    "excitement":  {"rate_range": (8, 20),  "pitch_range": (5, 10),  "vol_range": (2, 4),   "pause_range": (30, 150)},
    "contentment": {"rate_range": (-5, 5),  "pitch_range": (1, 4),   "vol_range": (0, 2),   "pause_range": (150, 400)},
    "sadness":     {"rate_range": (-15, -5),"pitch_range": (-8, -3), "vol_range": (-3, -1), "pause_range": (300, 700)},
    "grief":       {"rate_range": (-20, -8),"pitch_range": (-10, -5),"vol_range": (-4, -2), "pause_range": (400, 900)},
    "anger":       {"rate_range": (5, 18),  "pitch_range": (2, 7),   "vol_range": (2, 4),   "pause_range": (80, 250)},
    "frustration": {"rate_range": (3, 12),  "pitch_range": (1, 5),   "vol_range": (1, 3),   "pause_range": (100, 350)},
    "rage":        {"rate_range": (10, 20), "pitch_range": (3, 8),   "vol_range": (3, 4),   "pause_range": (50, 200)},
    "fear":        {"rate_range": (5, 15),  "pitch_range": (3, 8),   "vol_range": (-1, 2),  "pause_range": (150, 400)},
    "anxiety":     {"rate_range": (2, 10),  "pitch_range": (2, 6),   "vol_range": (-2, 1),  "pause_range": (200, 500)},
    "surprise":    {"rate_range": (5, 15),  "pitch_range": (5, 10),  "vol_range": (1, 3),   "pause_range": (100, 300)},
    "disgust":     {"rate_range": (-5, 5),  "pitch_range": (-3, 2),  "vol_range": (1, 3),   "pause_range": (150, 350)},
    "neutral":     {"rate_range": (-5, 5),  "pitch_range": (-3, 3),  "vol_range": (-1, 1),  "pause_range": (100, 300)},
}

# ── Public API ─────────────────────────────────────────────────────────────────

async def analyze_speech(
    text: str,
    emotion_override: Optional[str] = None,
    intensity: float = 1.0,
    detected_lang: Optional[str] = None,
) -> SpeechAnalysis:
    """
    Run Groq LLM analysis. Falls back gracefully to VADER if the API is unavailable.
    """
    if not settings.GROQ_API_KEY:
        return _fallback(text, emotion_override, "No GROQ_API_KEY in .env")

    if detected_lang is not None:
        lang = detected_lang
        lang_conf = 1.0
    else:
        lang, lang_conf = detect_language(text)

    lang_label = {
        "hi": "Hindi (Devanagari)",
        "hi-Latn": "Hinglish (Romanized Hindi)",
        "en": "English",
    }.get(lang, "English")

    logger.info(f"SpeechAnalyzer: detected language={lang} ({lang_label}), conf={lang_conf:.2f}")

    prompt = _USER_TMPL.format(
        text=text[:1200],
        emotion_override=emotion_override or "None",
        intensity=round(intensity, 2),
        language=lang_label,
    )

    system_prompt = _build_system_prompt(lang)

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
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": prompt},
                    ],
                    "temperature":      0.5,
                    "max_tokens":       800,
                    "response_format":  {"type": "json_object"},
                },
            )

        if resp.status_code != 200:
            return _fallback(text, emotion_override, f"Groq HTTP {resp.status_code}")

        raw_json = resp.json()["choices"][0]["message"]["content"]
        return _parse_response(raw_json, text, emotion_override)

    except (httpx.TimeoutException, httpx.NetworkError) as e:
        return _fallback(text, emotion_override, f"Groq timeout/network: {e}")
    except Exception as e:
        logger.warning(f"SpeechAnalyzer Groq error (non-fatal): {e}")
        return _fallback(text, emotion_override, str(e))


# ── Parsing & validation ───────────────────────────────────────────────────────

def _parse_response(raw: str, original_text: str, emotion_override: Optional[str]) -> SpeechAnalysis:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(1))
            except Exception:
                return _fallback(original_text, emotion_override, "JSON parse error")
        else:
            return _fallback(original_text, emotion_override, "JSON parse error")

    detected_emo = str(data.get("detected_emotion", "neutral")).strip().lower()
    confidence   = float(data.get("confidence", 0.8))
    delivery     = str(data.get("delivery_style", "conversational"))
    tone_arc     = str(data.get("tone_arc", "steady"))
    intent       = str(data.get("intent", "informational"))

    if emotion_override:
        detected_emo = emotion_override.lower()
        confidence   = 1.0

    segments: List[SegmentAnalysis] = []
    for seg in (data.get("segments") or []):
        try:
            segments.append(SegmentAnalysis(
                text           = str(seg.get("text", "")).strip(),
                emotion        = str(seg.get("emotion", detected_emo)),
                emphasis_words = [str(w) for w in (seg.get("emphasis_words") or [])],
                rate_delta_pct = max(-20, min(20, int(seg.get("rate_delta_pct", 0)))),
                pitch_delta_hz = max(-10, min(10, int(seg.get("pitch_delta_hz", 0)))),
                volume_delta_db= max(-4.0,min(4.0, float(seg.get("volume_delta_db", 0)))),
                pause_before_ms= max(0,   min(900, int(seg.get("pause_before_ms", 0)))),
                arc_position   = str(seg.get("arc_position", "middle")),
            ))
        except Exception:
            continue

    if len(segments) > 1:
        rates = [s.rate_delta_pct for s in segments]
        if len(set(rates)) == 1:
            segments = _inject_variation(segments, detected_emo)

    logger.info(
        f"SpeechAnalyzer [Groq]: emotion={detected_emo} ({confidence:.2f}), style={delivery}, arc={tone_arc}, segments={len(segments)}"
    )

    return SpeechAnalysis(
        detected_emotion = detected_emo,
        confidence       = confidence,
        delivery_style   = delivery,
        segments         = segments,
        llm_used         = True,
        tone_arc         = tone_arc,
        intent           = intent,
    )


def _inject_variation(segments: List[SegmentAnalysis], emotion: str) -> List[SegmentAnalysis]:
    arc_profile = _EMOTION_ARC.get(emotion, _EMOTION_ARC["neutral"])
    n = len(segments)

    for i, seg in enumerate(segments):
        t = i / max(1, n - 1)

        if t < 0.3:
            arc_factor = t / 0.3
            seg.arc_position = "opening"
        elif t < 0.7:
            arc_factor = 1.0
            seg.arc_position = "climax"
        else:
            arc_factor = 1.0 - (t - 0.7) / 0.3
            seg.arc_position = "closing"

        r_lo, r_hi = arc_profile["rate_range"]
        p_lo, p_hi = arc_profile["pitch_range"]
        v_lo, v_hi = arc_profile["vol_range"]
        pa_lo, pa_hi = arc_profile["pause_range"]

        seg.rate_delta_pct  = round(r_lo + (r_hi - r_lo) * arc_factor)
        seg.pitch_delta_hz  = round(p_lo + (p_hi - p_lo) * arc_factor)
        seg.volume_delta_db = round(v_lo + (v_hi - v_lo) * arc_factor, 1)
        if seg.arc_position == "climax":
            seg.pause_before_ms = pa_lo
        else:
            seg.pause_before_ms = round(pa_lo + (pa_hi - pa_lo) * (1 - arc_factor))

    return segments


# ── Rule-based fallback ────────────────────────────────────────────────────────

def _fallback(text: str, emotion_override: Optional[str], reason: str) -> SpeechAnalysis:
    logger.debug(f"SpeechAnalyzer fallback ({reason})")

    if emotion_override:
        detected_emo = emotion_override.lower()
        confidence   = 1.0
    else:
        detected_emo, confidence = _vader.analyze(text)

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    segments  = []
    n         = len(sentences)

    arc_profile = _EMOTION_ARC.get(detected_emo, _EMOTION_ARC["neutral"])

    for i, sent in enumerate(sentences[:8]):
        sent = sent.strip()
        if not sent:
            continue

        t = i / max(1, n - 1)

        if t < 0.25:
            arc_factor = t / 0.25
            arc_pos = "opening"
        elif t < 0.65:
            arc_factor = 1.0
            arc_pos = "climax"
        else:
            arc_factor = 1.0 - (t - 0.65) / 0.35
            arc_pos = "closing"

        r_lo, r_hi = arc_profile["rate_range"]
        p_lo, p_hi = arc_profile["pitch_range"]
        v_lo, v_hi = arc_profile["vol_range"]
        pa_lo, pa_hi = arc_profile["pause_range"]

        rate_d  = round(r_lo + (r_hi - r_lo) * arc_factor)
        pitch_d = round(p_lo + (p_hi - p_lo) * arc_factor)
        vol_d   = round(v_lo + (v_hi - v_lo) * arc_factor, 1)

        if arc_pos == "climax":
            pause_ms = pa_lo
        else:
            pause_ms = round(pa_lo + (pa_hi - pa_lo) * (1 - arc_factor))

        emphasis = []
        words = sent.split()
        for w in words:
            clean = re.sub(r"[^\w]", "", w).lower()
            if len(clean) > 4 and clean.isupper():
                emphasis.append(w)
            elif clean in {"never", "always", "amazing", "terrible", "love", "hate",
                          "please", "help", "sorry", "important", "really"}:
                emphasis.append(w)

        if sent.endswith("?"):
            pitch_d = max(pitch_d, 3)
        elif sent.endswith("!"):
            vol_d = min(vol_d + 1.5, 4.0)
            rate_d = min(rate_d + 3, 20)
        elif sent.endswith("..."):
            pause_ms = max(pause_ms, 500)
            rate_d = max(rate_d - 5, -20)

        segments.append(SegmentAnalysis(
            text            = sent,
            emotion         = detected_emo,
            emphasis_words  = emphasis[:3],
            rate_delta_pct  = rate_d,
            pitch_delta_hz  = pitch_d,
            volume_delta_db = vol_d,
            pause_before_ms = pause_ms if i > 0 else 0,
            arc_position    = arc_pos,
        ))

    tone_arcs = {
        "joy": "slow_build", "excitement": "building_urgency",
        "sadness": "emotional_wave", "grief": "peak_then_fade",
        "anger": "building_urgency", "neutral": "steady",
    }

    return SpeechAnalysis(
        detected_emotion = detected_emo,
        confidence       = confidence,
        delivery_style   = "conversational",
        segments         = segments,
        llm_used         = False,
        error            = reason,
        tone_arc         = tone_arcs.get(detected_emo, "steady"),
        intent           = "informational",
    )
