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
    return "+0%" if r in ("default", None, "") else r


def _to_pitch(p: str) -> str:
    if p in ("default", None, ""):
        return "+0Hz"
    try:
        st = float(p.replace("st", "").replace("+", ""))
        hz = round(st * 8.5)
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
        prosody: Dict[str, str],
    ) -> str:
        """Synthesize ONE sentence with the given emotion and prosody."""
        voice, style = EMOTION_STYLE_MAP.get(emotion.lower(), _DEFAULT_STYLE)
        rate   = _to_rate(prosody.get("rate", "default"))
        pitch  = _to_pitch(prosody.get("pitch", "default"))
        volume = _to_volume(prosody.get("volume", "default"))

        tok = _ctx_style.set(style)
        try:
            communicate = edge_tts.Communicate(
                text, voice=voice, rate=rate, pitch=pitch, volume=volume
            )
            await communicate.save(filepath)
            return filepath
        finally:
            _ctx_style.reset(tok)

    async def synthesize(
        self,
        text: str,
        ssml: str,
        filepath: str,
        prosody: Dict[str, str],
        emotion: str = "neutral",
    ) -> str:
        if not filepath.endswith(".mp3"):
            filepath = os.path.splitext(filepath)[0] + ".mp3"

        sentences = split_sentences(text)

        # ── Single sentence ────────────────────────────────────────────────
        if len(sentences) <= 1:
            voice, style = EMOTION_STYLE_MAP.get(emotion.lower(), _DEFAULT_STYLE)
            logger.info(
                f"ExpressiveEdgeTTS: [{emotion}→{style}] "
                f"rate={_to_rate(prosody.get('rate','default'))} "
                f"pitch={_to_pitch(prosody.get('pitch','default'))}"
            )
            try:
                await self._synth_sentence(text, filepath, emotion, prosody)
                logger.info(f"Audio saved: {filepath}")
                return filepath
            except Exception as e:
                raise TTSGenerationError(f"ExpressiveEdgeTTS single-sentence error: {e}")

        # ── Multi-sentence: per-sentence emotion, shared prosody ──────────
        # Style (cheerful / sad / angry…) changes per sentence so the voice
        # feels emotionally responsive.  Rate / pitch / volume are kept
        # consistent across all segments so the overall speech tempo and
        # loudness don't jump between sentences.
        logger.info(
            f"Multi-sentence mode: {len(sentences)} sentences — "
            "per-sentence emotion detection active (shared rate/pitch/volume)"
        )

        temp_paths = []
        for i, sentence in enumerate(sentences):
            s_emotion, _ = detect_sentence_emotion(sentence)
            temp = filepath.replace(".mp3", f"__seg{i}.mp3")
            temp_paths.append(temp)

            logger.info(
                f"  Seg {i+1}/{len(sentences)}: "
                f"'{sentence[:60]}{'...' if len(sentence)>60 else ''}' "
                f"→ emotion={s_emotion} | "
                f"rate={prosody.get('rate','default')} "
                f"pitch={prosody.get('pitch','default')} "
                f"vol={prosody.get('volume','default')}"
            )

            try:
                # Pass the top-level prosody so rate/pitch/volume stay uniform
                await self._synth_sentence(sentence, temp, s_emotion, prosody)
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
