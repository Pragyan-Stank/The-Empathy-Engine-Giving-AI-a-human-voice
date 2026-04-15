"""
Expressive Edge TTS — Emotion-styled speech via edge-tts monkey-patch.

Strategy:
  The standard `edge_tts.Communicate` uses `mkssml()` to build the SSML
  payload sent over its validated WebSocket. We monkey-patch that function
  to inject `mstts:express-as style="{style}"` when an emotion style is
  active, then restore the original after synthesis.

  This uses edge-tts's own, working WebSocket transport so we never touch
  the raw protocol — preventing the 403 errors that hit our custom aiohttp
  implementation.

Emotion → Voice × Style mapping (en-US-JennyNeural has the richest styles):
  angry | assistant | chat | cheerful | customerservice | empathetic |
  excited | friendly | general | hopeful | newscast | sad | shouting |
  terrified | unfriendly | whispering
"""
import os
import asyncio
import math
import contextvars
from typing import Dict
from xml.sax.saxutils import escape

import edge_tts
import edge_tts.communicate as _edge_comm

from app.services.tts.base import TTSEngine
from app.core.exceptions import TTSGenerationError
from app.core.logging_config import logger

# ── Emotion → (voice, azure_style, styledegree) ───────────────────────────────
_JENNY = "en-US-JennyNeural"
_GUY   = "en-US-GuyNeural"
_ARIA  = "en-US-AriaNeural"

EMOTION_STYLE_MAP: Dict[str, tuple] = {
    "joy":         (_JENNY, "cheerful",      "2"),
    "excitement":  (_JENNY, "excited",       "2"),
    "contentment": (_ARIA,  "empathetic",    "1.5"),
    "sadness":     (_JENNY, "sad",           "2"),
    "grief":       (_JENNY, "sad",           "2"),
    "anger":       (_GUY,   "angry",         "2"),
    "frustration": (_JENNY, "unfriendly",    "2"),
    "rage":        (_GUY,   "angry",         "2"),
    "disgust":     (_JENNY, "unfriendly",    "2"),
    "fear":        (_JENNY, "terrified",     "2"),
    "anxiety":     (_JENNY, "whispering",    "1.5"),
    "surprise":    (_JENNY, "excited",       "2"),
    "neutral":     (_ARIA,  "chat",          "1"),
}
_DEFAULT_STYLE = EMOTION_STYLE_MAP["neutral"]

# ── Context variable (safe for concurrent async tasks) ────────────────────────
_ctx_style  = contextvars.ContextVar("edge_style",  default=None)
_ctx_degree = contextvars.ContextVar("edge_degree", default="2")

# ── Monkey-patch mkssml ───────────────────────────────────────────────────────
_original_mkssml = _edge_comm.mkssml


def _styled_mkssml(tc, escaped_text):
    """
    Drop-in replacement for edge_tts.communicate.mkssml.
    When _ctx_style holds a style name, wraps the output in
    <mstts:express-as style="..."> for genuine emotional delivery.
    Otherwise delegates to the original function.
    """
    style  = _ctx_style.get()
    degree = _ctx_degree.get()

    if isinstance(escaped_text, bytes):
        escaped_text = escaped_text.decode("utf-8")

    if style:
        return (
            "<speak version='1.0' "
            "xmlns='http://www.w3.org/2001/10/synthesis' "
            "xmlns:mstts='https://www.w3.org/2001/mstts' "
            "xml:lang='en-US'>"
            f"<voice name='{tc.voice}'>"
            f"<mstts:express-as style='{style}' styledegree='{degree}'>"
            f"<prosody pitch='{tc.pitch}' rate='{tc.rate}' volume='{tc.volume}'>"
            f"{escaped_text}"
            "</prosody>"
            "</mstts:express-as>"
            "</voice>"
            "</speak>"
        )

    return _original_mkssml(tc, escaped_text)


# Apply patch once at import time
_edge_comm.mkssml = _styled_mkssml


# ── Prosody converters ────────────────────────────────────────────────────────

def _to_edge_rate(r: str) -> str:
    return "+0%" if r in ("default", None, "") else r


def _to_edge_pitch(p: str) -> str:
    if p in ("default", None, ""):
        return "+0Hz"
    try:
        st = float(p.replace("st", "").replace("+", ""))
        hz = round(st * 8.5)
        return f"+{hz}Hz" if hz >= 0 else f"{hz}Hz"
    except Exception:
        return "+0Hz"


def _to_edge_volume(v: str) -> str:
    if v in ("default", None, ""):
        return "+0%"
    try:
        db = float(v.replace("dB", "").replace("+", ""))
        pct = round((math.pow(10, db / 20) - 1) * 100)
        pct = max(-90, min(90, pct))
        return f"+{pct}%" if pct >= 0 else f"{pct}%"
    except Exception:
        return "+0%"


# ── Engine ────────────────────────────────────────────────────────────────────

class ExpressiveEdgeTTS(TTSEngine):
    """
    Edge TTS with genuine emotional voice styles injected via monkey-patched
    mkssml.  Uses the same validated WebSocket as the standard edge-tts library.
    Each emotion uses a different voice character AND style:
      joy → Jenny cheerful, anger → Guy angry, fear → Jenny terrified, etc.
    """

    available = True   # always available if edge-tts is installed

    def __init__(self):
        logger.info(
            "ExpressiveEdgeTTS ready (mkssml patched — "
            "emotion styles active via mstts:express-as)"
        )

    async def synthesize(
        self,
        text: str,
        ssml: str,
        filepath: str,
        prosody: Dict[str, str],
        emotion: str = "neutral",
    ) -> str:
        voice, style, degree = EMOTION_STYLE_MAP.get(
            emotion.lower(), _DEFAULT_STYLE
        )
        rate   = _to_edge_rate(prosody.get("rate", "default"))
        pitch  = _to_edge_pitch(prosody.get("pitch", "default"))
        volume = _to_edge_volume(prosody.get("volume", "default"))

        logger.info(
            f"ExpressiveEdgeTTS: emotion={emotion}, voice={voice}, "
            f"style={style}(×{degree}), rate={rate}, pitch={pitch}, volume={volume}"
        )

        # Ensure .mp3 extension
        if not filepath.endswith(".mp3"):
            filepath = os.path.splitext(filepath)[0] + ".mp3"

        # Set context vars for this async task (safe for concurrent requests)
        tok_style  = _ctx_style.set(style)
        tok_degree = _ctx_degree.set(degree)
        try:
            communicate = edge_tts.Communicate(
                text,
                voice=voice,
                rate=rate,
                pitch=pitch,
                volume=volume,
            )
            await communicate.save(filepath)
            logger.info(f"ExpressiveEdgeTTS audio saved: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"ExpressiveEdgeTTS error: {e}", exc_info=True)
            raise TTSGenerationError(f"ExpressiveEdgeTTS failed: {e}")
        finally:
            # Always restore context
            _ctx_style.reset(tok_style)
            _ctx_degree.reset(tok_degree)
