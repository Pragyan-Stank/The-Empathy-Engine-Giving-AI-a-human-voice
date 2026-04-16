"""
Groq-powered Speech Analyzer — the LLM brain of the TTS pipeline.

Architecture:  Input → [THIS MODULE] → Text Humanization → Prosody Engine → SSML → TTS

This module simulates how a skilled voice actor PREPARES to deliver text:

  1. Humanize raw text into natural spoken form
     - Inject punctuation (... ! ?) where a human speaker would pause/emphasize
     - Break long clauses into conversational breath groups
     - Add optional fillers (you know, actually, look, acha) for casual delivery
     - Preserve Hinglish / code-mixed language and Indian conversational rhythm
     - Rewrite formal text into spoken-word phrasing

  2. Model a DELIVERY ARC (opening → climax → resolution)
     - Opening: moderate pace, building attention
     - Middle: peak energy, fastest, most expressive
     - Closing: slowing, trailing off, reflective

  3. Produce per-segment prosody deltas (NOT flat — variation is mandatory)
     - rate:   ±20%  (dynamic range for real expressiveness)
     - pitch:  ±10Hz (noticeable intonation shifts)
     - volume: ±4dB  (energy variation)
     - pause:  0–900ms (breathing, dramatic pauses, hesitation)

  4. Identify emphasis words for <emphasis> injection in SSML

  5. Detect tone transitions within a single response
     - e.g. "start slow emotional → transition to fast expressive"

  All output is validated and clamped; if Groq fails, the module falls back
  to a sophisticated rule-based pipeline that still produces natural variation.
"""
import json
import re
import logging
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import httpx

from app.core.config import settings
from app.services.text.language_detector import detect_language, is_hindi_or_hinglish


logger = logging.getLogger("empathy_engine")

GROQ_API_URL   = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL     = "llama-3.1-8b-instant"   # fast, low-latency for real-time TTS
GROQ_TIMEOUT   = 6.0                       # seconds — slightly more room for quality

# ── Data Classes ───────────────────────────────────────────────────────────────

@dataclass
class SegmentAnalysis:
    """Prosody guidance for one text segment from the LLM."""
    text:          str
    emotion:       str = "neutral"
    emphasis_words: List[str] = field(default_factory=list)
    # Deltas applied ON TOP of the shared base prosody:
    rate_delta_pct:   int   = 0      # -20 to +20
    pitch_delta_hz:   int   = 0      # -10 to +10
    volume_delta_db:  float = 0.0    # -4.0 to +4.0
    pause_before_ms:  int   = 0      # 0 to 900
    # Delivery arc position:
    arc_position:     str   = "middle"  # opening | middle | climax | closing


@dataclass
class SpeechAnalysis:
    """Full LLM analysis output for one synthesis request."""
    humanized_text:  str
    delivery_style:  str                       # casual | empathetic | authoritative | excited | somber
    filler:          str = ""                  # "you know" | "actually" | "" etc.
    segments:        List[SegmentAnalysis] = field(default_factory=list)
    llm_used:        bool = False
    error:           Optional[str] = None
    # Tone transition descriptor:
    tone_arc:        str = "steady"            # e.g. "slow_build", "peak_then_fade", "emotional_wave"
    intent:          str = "informational"     # informational | empathetic | persuasive | urgent | reflective


# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM_BASE = (
    "You are a TTS speech performance director. "
    "Your job is to transform written text into a HUMAN SPEECH PERFORMANCE. "
    "Think like a voice actor preparing delivery — model pauses, "
    "emotional rises, breath groups, emphasis, and natural rhythm. "
    "Output ONLY valid JSON. No explanation. No markdown."
)

# Extra-strict constraint injected when Hindi/Hinglish is detected
_SYSTEM_HINDI_CONSTRAINT = (
    "\n\n### ABSOLUTE LANGUAGE RULE ###\n"
    "The input text is in Hindi / Hinglish. You MUST follow these rules:\n"
    "1. NEVER translate ANY word to English. Not a single word.\n"
    "2. Output humanized_text and ALL segment texts in THE EXACT SAME LANGUAGE as the input.\n"
    "3. If input is in Devanagari (हिंदी), output MUST stay in Devanagari.\n"
    "4. If input is in Roman Hindi (Hinglish), output MUST stay in Roman Hindi.\n"
    "5. You may ONLY change punctuation, add pauses, restructure phrasing — "
    "but EVERY word must remain in the input language.\n"
    "6. Fillers should be Hindi: 'acha', 'toh', 'dekho', 'matlab', 'suno' — NOT English fillers.\n"
    "Violating these rules is an ABSOLUTE FAILURE."
)

_SYSTEM_ENGLISH = (
    "\nCRITICAL: NEVER translate text to another language. "
    "If the input is in Hindi, Hinglish, or any Indian language, keep it in that EXACT language. "
    "Only improve spoken delivery, do NOT change the language."
)


def _build_system_prompt(lang: str) -> str:
    """Build the system prompt with language-specific constraints."""
    if lang in ("hi", "hi-Latn"):
        return _SYSTEM_BASE + _SYSTEM_HINDI_CONSTRAINT
    return _SYSTEM_BASE + _SYSTEM_ENGLISH

_USER_TMPL = """Analyze for speech PERFORMANCE (not just reading):
TEXT: {text}
EMOTION: {emotion}
INTENSITY: {intensity}
DETECTED_LANGUAGE: {language}

Return this exact JSON (no other text):
{{
  "humanized_text": "rewrite as natural spoken words — how a HUMAN would ACTUALLY say this IN THE SAME LANGUAGE",
  "delivery_style": "casual|empathetic|authoritative|excited|somber|conversational|reflective",
  "intent": "informational|empathetic|persuasive|urgent|reflective",
  "tone_arc": "steady|slow_build|peak_then_fade|emotional_wave|building_urgency",
  "filler": "",
  "segments": [
    {{
      "text": "segment text IN THE SAME LANGUAGE as input (one phrase/clause)",
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
- humanized_text: how a HUMAN would SAY it (add ... ! ? naturally)
- *** LANGUAGE PRESERVATION IS THE #1 RULE ***
- If DETECTED_LANGUAGE is "hi" or "hi-Latn": output MUST be in Hindi/Hinglish. ZERO English translation.
- If input is Devanagari Hindi, output MUST be Devanagari Hindi.
- If input is Romanized Hindi (Hinglish), output MUST be Romanized Hindi.
- Only improve spoken delivery style (pauses, emphasis, phrasing) — do NOT convert to English
- Rewrite formal text into conversational phrasing IN THE SAME LANGUAGE as input
- filler: if Hindi/Hinglish → use "acha","toh","dekho","matlab","suno","haan" ONLY
         if English → use "you know","actually","look","I mean" OR ""
- segments: one per breath group / clause, max 8
- DELIVERY ARC: opening segments slower, climax fastest, closing slowing down
- rate_delta_pct: VARY between -20 to +20 (NEVER all zeros — that's robotic)
- pitch_delta_hz: VARY between -10 to +10 (rising on questions, falling on statements)
- volume_delta_db: VARY between -4.0 to +4.0 (louder at emphasis, softer at reflection)
- pause_before_ms: 0 to 900 (longer before emotional moments, shorter in fast sections)
- emphasis_words: 1-3 KEY words per segment that need stress (in the INPUT LANGUAGE)
- Every segment MUST have DIFFERENT prosody values — NO flat delivery allowed
- If text has questions: raise pitch. If exclamations: louder + faster.
- Preserve Indian English / Hinglish / Hindi EXACTLY as-is — never translate"""


# ── Emotion → default arc profiles ─────────────────────────────────────────────
# Used by the fallback engine when LLM is unavailable

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
    emotion: str = "neutral",
    intensity: float = 1.0,
) -> SpeechAnalysis:
    """
    Run Groq LLM analysis. Falls back gracefully if API is unavailable.

    Returns a SpeechAnalysis with:
    - humanized_text: conversational rewrite of the input
    - segments: list of SegmentAnalysis with per-segment prosody deltas
    - delivery_style, filler, tone_arc, intent
    """
    if not settings.GROQ_API_KEY:
        return _fallback(text, emotion, "No GROQ_API_KEY in .env")

    # Detect input language to drive prompt behaviour
    lang, lang_conf = detect_language(text)
    lang_label = {
        "hi": "Hindi (Devanagari)",
        "hi-Latn": "Hinglish (Romanized Hindi)",
        "en": "English",
    }.get(lang, "English")

    logger.info(f"SpeechAnalyzer: detected language={lang} ({lang_label}), conf={lang_conf:.2f}")

    prompt = _USER_TMPL.format(
        text=text[:1200],   # more room for context
        emotion=emotion,
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
                    "temperature":      0.5,   # slightly more creative for variation
                    "max_tokens":       800,   # more room for richer segments
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
    tone_arc  = str(data.get("tone_arc", "steady"))
    intent    = str(data.get("intent", "informational"))

    segments: List[SegmentAnalysis] = []
    for seg in (data.get("segments") or []):
        try:
            segments.append(SegmentAnalysis(
                text           = str(seg.get("text", "")).strip(),
                emotion        = str(seg.get("emotion", emotion)),
                emphasis_words = [str(w) for w in (seg.get("emphasis_words") or [])],
                rate_delta_pct = max(-20, min(20, int(seg.get("rate_delta_pct", 0)))),
                pitch_delta_hz = max(-10, min(10, int(seg.get("pitch_delta_hz", 0)))),
                volume_delta_db= max(-4.0,min(4.0, float(seg.get("volume_delta_db", 0)))),
                pause_before_ms= max(0,   min(900, int(seg.get("pause_before_ms", 0)))),
                arc_position   = str(seg.get("arc_position", "middle")),
            ))
        except Exception:
            continue

    # ── Validate: enforce variation (reject if all deltas are identical) ───────
    if len(segments) > 1:
        rates = [s.rate_delta_pct for s in segments]
        if len(set(rates)) == 1:
            # LLM gave flat prosody — inject synthetic variation
            segments = _inject_variation(segments, emotion)

    logger.info(
        f"SpeechAnalyzer [Groq]: style={delivery}, arc={tone_arc}, intent={intent}, "
        f"filler='{filler}', segments={len(segments)}, "
        f"humanized_len={len(humanized)}"
    )

    return SpeechAnalysis(
        humanized_text = humanized,
        delivery_style = delivery,
        filler         = filler,
        segments       = segments,
        llm_used       = True,
        tone_arc       = tone_arc,
        intent         = intent,
    )


def _inject_variation(segments: List[SegmentAnalysis], emotion: str) -> List[SegmentAnalysis]:
    """
    Post-process to inject natural variation when LLM returned flat deltas.
    Uses a delivery arc: opening → build → climax → resolution.
    """
    arc_profile = _EMOTION_ARC.get(emotion, _EMOTION_ARC["neutral"])
    n = len(segments)

    for i, seg in enumerate(segments):
        # Position in arc: 0.0 = opening, 1.0 = end
        t = i / max(1, n - 1)

        # Delivery arc shape: slow start → peak at 60% → gentle fade
        if t < 0.3:
            arc_factor = t / 0.3           # 0 → 1 (building)
            seg.arc_position = "opening"
        elif t < 0.7:
            arc_factor = 1.0               # peak
            seg.arc_position = "climax"
        else:
            arc_factor = 1.0 - (t - 0.7) / 0.3  # 1 → 0 (fading)
            seg.arc_position = "closing"

        r_lo, r_hi = arc_profile["rate_range"]
        p_lo, p_hi = arc_profile["pitch_range"]
        v_lo, v_hi = arc_profile["vol_range"]
        pa_lo, pa_hi = arc_profile["pause_range"]

        seg.rate_delta_pct  = round(r_lo + (r_hi - r_lo) * arc_factor)
        seg.pitch_delta_hz  = round(p_lo + (p_hi - p_lo) * arc_factor)
        seg.volume_delta_db = round(v_lo + (v_hi - v_lo) * arc_factor, 1)
        # Pauses: longer at opening and closing, shorter at climax
        if seg.arc_position == "climax":
            seg.pause_before_ms = pa_lo
        else:
            seg.pause_before_ms = round(pa_lo + (pa_hi - pa_lo) * (1 - arc_factor))

    return segments


# ── Rule-based fallback ────────────────────────────────────────────────────────

def _fallback(text: str, emotion: str, reason: str) -> SpeechAnalysis:
    """
    Sophisticated rule-based fallback — no LLM required.
    Produces a valid SpeechAnalysis with natural prosody variation
    using the delivery arc model.
    """
    logger.debug(f"SpeechAnalyzer fallback ({reason})")

    # Rule-based: split into sentences, assign prosody via delivery arc
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    segments  = []
    n         = len(sentences)

    arc_profile = _EMOTION_ARC.get(emotion, _EMOTION_ARC["neutral"])

    for i, sent in enumerate(sentences[:8]):
        sent = sent.strip()
        if not sent:
            continue

        # Delivery arc position
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

        # Auto-detect emphasis words (simple heuristic)
        emphasis = []
        words = sent.split()
        for w in words:
            clean = re.sub(r"[^\w]", "", w).lower()
            if len(clean) > 4 and clean.isupper():
                emphasis.append(w)
            elif clean in {"never", "always", "amazing", "terrible", "love", "hate",
                          "please", "help", "sorry", "important", "really"}:
                emphasis.append(w)

        # Adjust for punctuation
        if sent.endswith("?"):
            pitch_d = max(pitch_d, 3)  # rising intonation for questions
        elif sent.endswith("!"):
            vol_d = min(vol_d + 1.5, 4.0)  # louder for exclamations
            rate_d = min(rate_d + 3, 20)
        elif sent.endswith("..."):
            pause_ms = max(pause_ms, 500)
            rate_d = max(rate_d - 5, -20)

        segments.append(SegmentAnalysis(
            text            = sent,
            emotion         = emotion,
            emphasis_words  = emphasis[:3],
            rate_delta_pct  = rate_d,
            pitch_delta_hz  = pitch_d,
            volume_delta_db = vol_d,
            pause_before_ms = pause_ms if i > 0 else 0,
            arc_position    = arc_pos,
        ))

    # Determine tone arc from emotion
    tone_arcs = {
        "joy": "slow_build", "excitement": "building_urgency",
        "sadness": "emotional_wave", "grief": "peak_then_fade",
        "anger": "building_urgency", "neutral": "steady",
    }

    return SpeechAnalysis(
        humanized_text = text,
        delivery_style = "conversational",
        filler         = "",
        segments       = segments,
        llm_used       = False,
        error          = reason,
        tone_arc       = tone_arcs.get(emotion, "steady"),
        intent         = "informational",
    )
