"""
Advanced SSMLBuilder — produces rich, emotionally expressive SSML.

Features:
  1. Punctuation-aware <break> timing:
       "..."  → 600ms (hesitation)
       "."    → 300ms (sentence end)
       "!"    → 200ms (excitement beat)
       "?"    → 250ms (curiosity)
       ","    → 120ms (breath)

  2. Per-segment <prosody> wrapping when emotion varies within text.
     The ||Xms|| markers injected by TextEnhancer become <break time="Xms"/>.

  3. Selective <emphasis> on emotionally charged words:
       level="strong"   → anger, urgency, critical words
       level="moderate" → happy, important, positive words
       level="reduced"  → calm, whispered states

  4. Emotion-aware SSML shaping:
       - Sad: no emphasis, gentle breaks
       - Angry: strong emphasis on key words, shorter breaks
       - Joy: moderate emphasis, bright cadence
       - Anxiety: whispering emphasis, longer hesitation breaks

  5. COMPOSITE SSML for multi-segment synthesis:
     build_composite_ssml() — produces a single <speak> with per-segment
     <prosody> wrappers and inter-segment <break> tags, enabling
     tone transitions within a single response.

  6. build_ssml_engine(text, prosody, emotion) — used by TTS engines
     build_ssml_display(text, prosody, emotion) — for UI preview
     build_segment_ssml(segment_text, prosody, emotion) — single segment
"""
import re
from typing import Dict, List, Optional, Tuple


# ── Pause map for punctuation ──────────────────────────────────────────────────
_PUNCT_BREAK: Dict[str, str] = {
    "...": "600ms",
    "…":   "600ms",
    "!":   "200ms",
    "?":   "250ms",
    ".":   "300ms",
    ",":   "120ms",
    ";":   "180ms",
    "—":   "350ms",
    "-":   "150ms",
}

# ── Emphasis word sets ─────────────────────────────────────────────────────────
_STRONG_WORDS = {
    # Anger / urgency
    "never", "always", "stop", "hate", "furious", "outraged", "worst",
    "terrible", "horrible", "disgusting", "unacceptable", "immediately",
    "urgent", "critical", "emergency", "now", "must", "demand",
    # Deep sadness
    "lost", "gone", "miss", "goodbye", "forever", "alone", "broken",
    "shattered", "destroyed", "hopeless",
    # Excitement
    "amazing", "incredible", "fantastic", "brilliant", "outstanding",
    "perfect", "wonderful", "spectacular",
}

_MODERATE_WORDS = {
    "important", "please", "sorry", "love", "great", "good", "help",
    "need", "want", "hope", "believe", "trust", "care", "feel",
    "really", "very", "truly", "definitely", "absolutely",
}

_REDUCED_WORDS = {
    "calm", "quiet", "soft", "gentle", "slowly", "carefully", "peacefully",
}

# Emotions where emphasis should be suppressed (sounds unnatural)
_NO_EMPHASIS_EMOTIONS = {"grief", "sadness", "anxiety", "contentment", "neutral"}
_STRONG_EMPHASIS_EMOTIONS = {"anger", "rage", "excitement", "frustration", "fear"}

# ── Prosody profiles per emotion ───────────────────────────────────────────────
# These are the "display SSML" prosody values injected as literal SSML attributes
_EMOTION_PROSODY: Dict[str, Dict[str, str]] = {
    "joy":         {"rate": "medium",  "pitch": "+3st"},
    "excitement":  {"rate": "fast",    "pitch": "+5st",  "volume": "+3dB"},
    "contentment": {"rate": "slow",    "pitch": "+1st"},
    "sadness":     {"rate": "slow",    "pitch": "-2st",  "volume": "-1dB"},
    "grief":       {"rate": "x-slow",  "pitch": "-3.5st","volume": "-2dB"},
    "anger":       {"rate": "fast",    "pitch": "+1st",  "volume": "+4dB"},
    "frustration": {"rate": "medium",  "pitch": "+2st",  "volume": "+2dB"},
    "rage":        {"rate": "x-fast",  "pitch": "+2st",  "volume": "+5dB"},
    "fear":        {"rate": "fast",    "pitch": "+3st"},
    "anxiety":     {"rate": "medium",  "pitch": "+3st"},
    "surprise":    {"rate": "fast",    "pitch": "+4st"},
    "disgust":     {"rate": "medium",  "pitch": "-1st",  "volume": "+1dB"},
    "neutral":     {"rate": "medium",  "pitch": "0st"},
}


class SSMLBuilder:

    # ── Public API ─────────────────────────────────────────────────────────────

    def build_ssml_display(
        self,
        text: str,
        prosody: Dict[str, str],
        emotion: str = "neutral",
        extra_emphasis: list = None,
    ) -> str:
        """Full SSML with <prosody> for UI preview panel."""
        inner = self._build_inner(text, emotion, extra_emphasis=extra_emphasis)
        wrapper = self._prosody_attrs_from_dict(prosody)
        if wrapper:
            inner = f'<prosody {wrapper}>{inner}</prosody>'
        return f"<speak>{inner}</speak>"

    def build_ssml_engine(
        self,
        text: str,
        prosody: Optional[Dict[str, str]] = None,
        emotion: str = "neutral",
        extra_emphasis: list = None,
    ) -> str:
        """
        SSML for the TTS engine.
        Uses prosody dict if provided; otherwise uses the emotion profile.
        Includes emphasis + pauses but NOT a double-wrapped prosody on top
        (Google uses audioConfig; Edge uses Communicate() params).
        """
        inner = self._build_inner(text, emotion, extra_emphasis=extra_emphasis)
        return f"<speak>{inner}</speak>"

    def build_segment_ssml(
        self,
        segment_text: str,
        prosody: Dict[str, str],
        emotion: str = "neutral",
        extra_emphasis: list = None,
    ) -> str:
        """
        Build SSML for a single segment (used in multi-sentence synthesis).
        Wraps with <prosody> so each segment has its own voice shaping.
        """
        inner = self._build_inner(segment_text, emotion, extra_emphasis=extra_emphasis)
        wrapper = self._prosody_attrs_from_dict(prosody)
        if wrapper:
            inner = f'<prosody {wrapper}>{inner}</prosody>'
        return f"<speak>{inner}</speak>"

    def build_composite_ssml(
        self,
        segments: List[Dict],
        emotion: str = "neutral",
    ) -> str:
        """
        Build a SINGLE composite <speak> with per-segment <prosody> wrappers.

        This is the key to achieving tone transitions within a single TTS call.
        Google Cloud TTS processes this as one synthesis request but applies
        different prosody to each segment.

        Args:
            segments: list of dicts with keys:
                - text: segment text
                - rate: speaking rate (e.g. "1.1" for Google, "+10%" for display)
                - pitch: pitch in semitones (e.g. "+2.0st")
                - volume: volume gain (e.g. "+3.0dB")
                - pause_before_ms: pause before this segment in ms
                - emotion: segment-level emotion
                - emphasis_words: list of words to emphasize
            emotion: overall emotion for emphasis decisions

        Returns:
            Complete <speak> SSML string with per-segment prosody.
        """
        parts = []

        for i, seg in enumerate(segments):
            seg_text = seg.get("text", "")
            if not seg_text.strip():
                continue

            seg_emotion = seg.get("emotion", emotion)
            emphasis_words = seg.get("emphasis_words", [])

            # Build inner content with emphasis and punctuation breaks
            inner = self._build_inner(seg_text, seg_emotion, extra_emphasis=emphasis_words)

            # Insert inter-segment pause
            pause_ms = seg.get("pause_before_ms", 0)
            if pause_ms > 0 and i > 0:
                parts.append(f'<break time="{pause_ms}ms"/>')

            # Wrap with per-segment prosody
            rate = seg.get("rate", "")
            pitch = seg.get("pitch", "")
            volume = seg.get("volume", "")

            prosody_attrs = self._build_prosody_attrs(rate, pitch, volume)
            if prosody_attrs:
                parts.append(f'<prosody {prosody_attrs}>{inner}</prosody>')
            else:
                parts.append(inner)

        return f"<speak>{' '.join(parts)}</speak>"

    def build_google_composite_ssml(
        self,
        segments: List[Dict],
        emotion: str = "neutral",
    ) -> str:
        """
        Build composite SSML specifically optimized for Google Cloud TTS.

        Uses Google's SSML-compatible prosody attributes:
          - rate: "x-slow" / "slow" / "medium" / "fast" / "x-fast" / percentage
          - pitch: "+Xst" / "-Xst" / "x-low" / "low" / "medium" / "high" / "x-high"
          - volume: "+XdB" / "-XdB" / "silent" / "x-soft" / "soft" / "medium" / "loud"

        Google TTS processes this as a single synthesis call, applying different
        prosody per segment for natural tone transitions.
        """
        parts = []

        for i, seg in enumerate(segments):
            seg_text = seg.get("text", "")
            if not seg_text.strip():
                continue

            seg_emotion = seg.get("emotion", emotion)
            emphasis_words = seg.get("emphasis_words", [])

            inner = self._build_inner(seg_text, seg_emotion, extra_emphasis=emphasis_words)

            # Inter-segment pause
            pause_ms = seg.get("pause_before_ms", 0)
            if pause_ms > 0 and i > 0:
                parts.append(f'<break time="{pause_ms}ms"/>')

            # Google-format prosody
            rate_mult = seg.get("rate_mult", 1.0)
            pitch_st = seg.get("pitch_st", 0.0)
            vol_db = seg.get("vol_db", 0.0)

            attrs = []
            if rate_mult != 1.0:
                # Google accepts percentage: "+20%" means 20% faster
                rate_pct = round((rate_mult - 1.0) * 100)
                rate_str = f"+{rate_pct}%" if rate_pct >= 0 else f"{rate_pct}%"
                attrs.append(f'rate="{rate_str}"')
            if abs(pitch_st) > 0.1:
                pitch_str = f"+{pitch_st}st" if pitch_st >= 0 else f"{pitch_st}st"
                attrs.append(f'pitch="{pitch_str}"')
            if abs(vol_db) > 0.1:
                vol_str = f"+{vol_db}dB" if vol_db >= 0 else f"{vol_db}dB"
                attrs.append(f'volume="{vol_str}"')

            if attrs:
                parts.append(f'<prosody {" ".join(attrs)}>{inner}</prosody>')
            else:
                parts.append(inner)

        return f"<speak>{' '.join(parts)}</speak>"

    # Backward-compatible alias
    def build_ssml(self, text: str, prosody: Dict[str, str]) -> str:
        return self.build_ssml_display(text, prosody)

    # ── Private Helpers ────────────────────────────────────────────────────────

    def _build_inner(
        self, text: str, emotion: str = "neutral", extra_emphasis: list = None
    ) -> str:
        """Core pipeline: escape → pause markers → emphasis → punctuation breaks."""
        escaped = self._escape_xml(text)
        # Replace ||Xms|| pause markers injected by TextEnhancer
        processed = self._resolve_pause_markers(escaped)
        # Apply word-level emphasis (skipped for calm/sad emotions)
        processed = self._apply_emphasis(processed, emotion, extra_emphasis=extra_emphasis)
        # Convert punctuation to <break> tags
        processed = self._apply_punctuation_breaks(processed)
        return processed

    def _escape_xml(self, text: str) -> str:
        text = text.replace("&", "&amp;")
        text = text.replace("<", "&lt;")
        text = text.replace(">", "&gt;")
        text = text.replace('"', "&quot;")
        text = text.replace("'", "&apos;")
        return text

    def _resolve_pause_markers(self, text: str) -> str:
        """Convert ||300ms|| → <break time="300ms"/>"""
        return re.sub(
            r"\|\|(\d+ms)\|\|",
            r'<break time="\1"/>',
            text,
        )

    def _apply_emphasis(
        self, text: str, emotion: str, extra_emphasis: list = None
    ) -> str:
        """
        Apply <emphasis> tags to emotionally charged words.
        - Skipped entirely for calm/sad emotions (sounds robotic)
        - level="strong" for anger/excitement on critical words
        - level="moderate" for positive/important on key words
        - extra_emphasis: additional words from LLM context analysis
        """
        if emotion in _NO_EMPHASIS_EMOTIONS:
            return text

        # Merge static word set with LLM-identified extra words
        extra_set = {w.lower().strip() for w in (extra_emphasis or [])}

        level = "strong" if emotion in _STRONG_EMPHASIS_EMOTIONS else "moderate"
        words = text.split()
        out = []
        for w in words:
            # Don't apply emphasis inside existing SSML tags
            if w.startswith("<"):
                out.append(w)
                continue
            clean = re.sub(r"[^\w]", "", w.lower())
            if clean in _STRONG_WORDS or clean in extra_set:
                out.append(f'<emphasis level="{level}">{w}</emphasis>')
            elif clean in _MODERATE_WORDS and level == "moderate":
                out.append(f'<emphasis level="moderate">{w}</emphasis>')
            elif clean in _REDUCED_WORDS:
                out.append(f'<emphasis level="reduced">{w}</emphasis>')
            else:
                out.append(w)
        return " ".join(out)

    def _apply_punctuation_breaks(self, text: str) -> str:
        """
        Replace punctuation with <break> tags for realistic speech rhythm.

        Handles:
          "..."  → break 600ms
          "."    → break 300ms
          "!"    → break 200ms
          "?"    → break 250ms
          ","    → break 120ms
          ";"    → break 180ms
        """
        # Ellipsis first (must come before single-dot rule)
        text = re.sub(r"\.{3}", '<break time="600ms"/>', text)
        text = text.replace("…", '<break time="600ms"/>')

        # Em-dash
        text = text.replace(" — ", '<break time="350ms"/>')
        text = text.replace("—", '<break time="350ms"/>')

        # Terminal punctuation — insert break AFTER (keep punctuation for display)
        def _replace_terminal(m: re.Match) -> str:
            char = m.group(1)
            breaks = {"!": "200ms", "?": "250ms", ".": "300ms"}
            t = breaks.get(char, "300ms")
            return f'{char}<break time="{t}"/>'

        # Only after punctuation not already inside a tag
        text = re.sub(r'([!?.])(?!\s*<|[!?.])', _replace_terminal, text)

        # Comma / semicolon (inline breath)
        text = re.sub(r',(?!\s*<)', ',<break time="120ms"/>', text)
        text = re.sub(r';(?!\s*<)', ';<break time="180ms"/>', text)

        return text

    def _prosody_attrs_from_dict(self, prosody: Dict[str, str]) -> str:
        """Build e.g. 'rate=\"fast\" pitch=\"+3st\" volume=\"+4dB\"' from dict."""
        attrs = []
        rate   = prosody.get("rate", "default")
        pitch  = prosody.get("pitch", "default")
        volume = prosody.get("volume", "default")

        if rate not in ("default", "", None):
            attrs.append(f'rate="{rate}"')
        if pitch not in ("default", "0st", "", None):
            attrs.append(f'pitch="{pitch}"')
        if volume not in ("default", "0dB", "", None):
            attrs.append(f'volume="{volume}"')
        return " ".join(attrs)

    def _build_prosody_attrs(self, rate: str, pitch: str, volume: str) -> str:
        """Build prosody attribute string from individual values."""
        attrs = []
        if rate and rate not in ("default", "0", "0%", "+0%"):
            attrs.append(f'rate="{rate}"')
        if pitch and pitch not in ("default", "0", "0st", "+0st", "0Hz", "+0Hz"):
            attrs.append(f'pitch="{pitch}"')
        if volume and volume not in ("default", "0", "0dB", "+0dB", "0%", "+0%"):
            attrs.append(f'volume="{volume}"')
        return " ".join(attrs)
