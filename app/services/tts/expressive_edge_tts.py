"""
Expressive Edge TTS — Emotion-styled speech with per-sentence emotion detection.

Key design decisions:
  1. Monkey-patches edge_tts.communicate.mkssml to inject mstts:express-as without
     styledegree (which the free Edge TTS endpoint rejects).
  2. en-US-AriaNeural is used for ALL emotion styles — she supports the broadest
     style palette: angry | cheerful | sad | terrified | unfriendly | excited |
     empathetic | friendly | whispering | hopeful | lyrical | shouting | chat
  3. Multi-sentence texts are split at sentence boundaries. Each sentence gets its
     own per-sentence VADER emotion detection → separate audio → concatenated.
     e.g. "I lost my dog today. LOL, it was a soft toy!"
          → Sentence 1: sad voice   (slow, heavy)
          → Sentence 2: cheerful    (bright, quick)
"""
import os
import re
import math
import contextvars
from typing import Dict, List

import edge_tts
import edge_tts.communicate as _edge_comm

from app.services.tts.base import TTSEngine
from app.core.exceptions import TTSGenerationError
from app.core.logging_config import logger
from app.services.emotion.sentence_analysis import (
    analyze_text, detect_sentence_emotion, split_sentences
)

_PAUSE_MARKER_RE = re.compile(r"\|\|\d+ms\|\|")

# ── Emotion → (voice, azure_style) ────────────────────────────────────────────
# AriaNeural confirmed to support all listed styles on the Edge TTS free endpoint.
# Note: styledegree is intentionally OMITTED — it breaks the free endpoint silently.
_ARIA = "en-US-AriaNeural"
_GUY  = "en-US-GuyNeural"   # limited styles: angry | friendly | sad | newscast

EMOTION_STYLE_MAP: Dict[str, tuple] = {
    "joy":         (_ARIA, "cheerful"),
    "excitement":  (_ARIA, "excited"),
    "contentment": (_ARIA, "friendly"),
    "sadness":     (_ARIA, "sad"),
    "grief":       (_ARIA, "sad"),
    "anger":       (_ARIA, "angry"),
    "frustration": (_ARIA, "unfriendly"),
    "rage":        (_ARIA, "shouting"),
    "disgust":     (_ARIA, "unfriendly"),
    "fear":        (_ARIA, "terrified"),
    "anxiety":     (_ARIA, "whispering"),
    "surprise":    (_ARIA, "excited"),
    "neutral":     (_ARIA, "chat"),
}
_DEFAULT_STYLE = EMOTION_STYLE_MAP["neutral"]

# ── Context variables (safe for concurrent async tasks) ───────────────────────
_ctx_style = contextvars.ContextVar("edge_style", default=None)

# ── Monkey-patch mkssml ───────────────────────────────────────────────────────
_original_mkssml = _edge_comm.mkssml


def _styled_mkssml(tc, escaped_text):
    """
    Replacement for edge_tts.communicate.mkssml.
    Injects mstts:express-as style tag when _ctx_style is set in the current
    async task context.  Falls through to the original function otherwise.
    """
    style = _ctx_style.get()

    if isinstance(escaped_text, bytes):
        escaped_text = escaped_text.decode("utf-8")

    if style:
        return (
            "<speak version='1.0' "
            "xmlns='http://www.w3.org/2001/10/synthesis' "
            "xmlns:mstts='https://www.w3.org/2001/mstts' "
            "xml:lang='en-US'>"
            f"<voice name='{tc.voice}'>"
            f"<mstts:express-as style='{style}'>"
            f"<prosody pitch='{tc.pitch}' rate='{tc.rate}' volume='{tc.volume}'>"
            f"{escaped_text}"
            "</prosody>"
            "</mstts:express-as>"
            "</voice>"
            "</speak>"
        )

    return _original_mkssml(tc, escaped_text)


# Apply patch once at module import time
_edge_comm.mkssml = _styled_mkssml


# ── Prosody converters ────────────────────────────────────────────────────────

def _to_rate(r: str) -> str:
    if r in ("default", None, ""):
        return "+0%"
    try:
        pct = int(float(r.replace("%", "").replace("+", "")))
        # Edge TTS safe range: -25% to +75%.
        # More conservative clamp to avoid 'No audio received' reliably.
        pct = max(-25, min(75, pct))
        return f"+{pct}%" if pct >= 0 else f"{pct}%"
    except Exception:
        return "+0%"


def _to_pitch(p: str) -> str:
    if p in ("default", None, ""):
        return "+0Hz"
    try:
        st = float(p.replace("st", "").replace("+", ""))
        hz = round(st * 8.5)
        # Edge TTS safe range: -20Hz to +20Hz (conservative to avoid failures)
        hz = max(-20, min(20, hz))
        return f"+{hz}Hz" if hz >= 0 else f"{hz}Hz"
    except Exception:
        return "+0Hz"


def _to_volume(v: str) -> str:
    if v in ("default", None, ""):
        return "+0%"
    try:
        db = float(v.replace("dB", "").replace("+", ""))
        pct = round((math.pow(10, db / 20) - 1) * 100)
        pct = max(-90, min(90, pct))
        return f"+{pct}%" if pct >= 0 else f"{pct}%"
    except Exception:
        return "+0%"


def _concat_mp3s(paths: List[str], output: str) -> str:
    """
    Concatenate MP3 files by appending raw bytes.
    MP3 frames are self-contained, so sequential concatenation is playable
    in all standard players even without an ID3 header.
    """
    with open(output, "wb") as out:
        for path in paths:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    out.write(f.read())
    return output




# ── Engine ────────────────────────────────────────────────────────────────────

class ExpressiveEdgeTTS(TTSEngine):
    """
    Emotion-styled Edge TTS with per-sentence emotion synthesis.
    - Single sentence → one synthesis call with mstts:express-as style
    - Multiple sentences → each sentence synthesized with its own detected emotion,
      then all segments concatenated into a single MP3
    """

    available = True

    def __init__(self):
        logger.info(
            "ExpressiveEdgeTTS ready — "
            "AriaNeural styles + per-sentence emotion splitting"
        )

    async def _synth_sentence(
        self,
        text: str,
        filepath: str,
        emotion: str,
        rate: str,
        pitch: str,
        volume: str,
    ) -> str:
        """
        Synthesize ONE sentence with pre-computed (clamped) prosody values.

        Prosody is passed in already converted by the caller so that ALL
        segments across a multi-sentence synthesis share IDENTICAL rate/pitch/volume.
        Retry once on transient 'No audio received' with the SAME values
        (NOT neutral fallback) to maintain consistency.
        """
        voice, style = EMOTION_STYLE_MAP.get(emotion.lower(), _DEFAULT_STYLE)

        for attempt in range(2):
            tok = _ctx_style.set(style)
            try:
                communicate = edge_tts.Communicate(
                    text, voice=voice, rate=rate, pitch=pitch, volume=volume
                )
                await communicate.save(filepath)
                return filepath
            except Exception as exc:
                _ctx_style.reset(tok)
                if "No audio was received" in str(exc) and attempt == 0:
                    logger.warning(
                        f"ExpressiveEdgeTTS: 'No audio received' for [{emotion}] "
                        f"(rate={rate} pitch={pitch}) — retrying once..."
                    )
                    continue  # retry with SAME prosody (transient error)
                raise
            finally:
                try:
                    _ctx_style.reset(tok)
                except Exception:
                    pass

        raise RuntimeError("ExpressiveEdgeTTS: synthesis failed after retry")

    async def synthesize(
        self,
        text: str,
        ssml: str,
        filepath: str,
        prosody: Dict[str, str],
        emotion: str = "neutral",
    ) -> str:
        # Safety: strip any ||Xms|| pause markers that weren't converted upstream
        text = _PAUSE_MARKER_RE.sub(", ", text).strip()

        if not filepath.endswith(".mp3"):
            filepath = os.path.splitext(filepath)[0] + ".mp3"

        # ── Pre-compute shared prosody values ONCE ──────────────────────────
        # Converting here guarantees every segment gets IDENTICAL rate/pitch/volume.
        # The emotional STYLE (cheerful/sad/angry) changes per sentence but the
        # acoustic properties are locked to the request-level prosody.
        shared_rate   = _to_rate(prosody.get("rate", "default"))
        shared_pitch  = _to_pitch(prosody.get("pitch", "default"))
        shared_volume = _to_volume(prosody.get("volume", "default"))

        sentences = split_sentences(text)

        # ── Single sentence ────────────────────────────────────────────────
        if len(sentences) <= 1:
            _, style = EMOTION_STYLE_MAP.get(emotion.lower(), _DEFAULT_STYLE)
            logger.info(
                f"ExpressiveEdgeTTS: [{emotion}→{style}] "
                f"rate={shared_rate} pitch={shared_pitch} vol={shared_volume}"
            )
            try:
                await self._synth_sentence(
                    text, filepath, emotion,
                    shared_rate, shared_pitch, shared_volume,
                )
                logger.info(f"Audio saved: {filepath}")
                return filepath
            except Exception as e:
                raise TTSGenerationError(f"ExpressiveEdgeTTS single-sentence error: {e}")

        # ── Multi-sentence: per-sentence emotion style, SHARED prosody ─────
        # Style (cheerful / sad / angry…) changes per sentence.
        # rate / pitch / volume are pre-computed above and identical for ALL segments.
        logger.info(
            f"Multi-sentence mode: {len(sentences)} sentences — "
            f"shared prosody: rate={shared_rate} pitch={shared_pitch} vol={shared_volume}"
        )

        temp_paths = []
        for i, sentence in enumerate(sentences):
            s_emotion, _ = detect_sentence_emotion(sentence)
            _, s_style   = EMOTION_STYLE_MAP.get(s_emotion.lower(), _DEFAULT_STYLE)
            temp = filepath.replace(".mp3", f"__seg{i}.mp3")
            temp_paths.append(temp)

            logger.info(
                f"  Seg {i+1}/{len(sentences)}: "
                f"'{sentence[:60]}{'...' if len(sentence)>60 else ''}' "
                f"→ style={s_style} | "
                f"rate={shared_rate} pitch={shared_pitch} vol={shared_volume}"
            )

            try:
                await self._synth_sentence(
                    sentence, temp, s_emotion,
                    shared_rate, shared_pitch, shared_volume,
                )
            except Exception as e:
                logger.error(f"  Seg {i+1} synthesis failed: {e}. Skipping.")

        valid = [p for p in temp_paths if os.path.exists(p) and os.path.getsize(p) > 0]
        if not valid:
            raise TTSGenerationError("All sentence segments failed — no audio generated.")

        _concat_mp3s(valid, filepath)

        # Cleanup segments
        for p in temp_paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

        logger.info(f"Multi-segment audio concatenated → {filepath}")
        return filepath
