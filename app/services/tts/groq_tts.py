"""
Groq Orpheus TTS Engine — playai-tts (English) via Groq's audio/speech endpoint.

Emotion → voice mapping uses Orpheus English voices from Groq's PlayAI collection.
Rate prosody is converted to Groq's speed parameter (0.25–4.0).
Runs the synchronous Groq SDK call in a thread executor to avoid blocking FastAPI.

API key: read from settings.GROQ_API_KEY (same key used for LLM calls).
"""
import os
import asyncio
from typing import Dict, Optional

from app.services.tts.base import TTSEngine
from app.core.exceptions import TTSGenerationError
from app.core.logging_config import logger
from app.core.config import settings

try:
    from groq import Groq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False

# ── Emotion → Orpheus voice ────────────────────────────────────────────────────
# Voices chosen to match the emotional character of each state.
_EMOTION_VOICE_MAP: Dict[str, str] = {
    "joy":         "Arista-PlayAI",    # bright, upbeat female
    "excitement":  "Chip-PlayAI",      # energetic male
    "contentment": "Celeste-PlayAI",   # warm, calm female
    "sadness":     "Gail-PlayAI",      # soft, gentle female
    "grief":       "Gail-PlayAI",
    "anger":       "Thunder-PlayAI",   # powerful, assertive male
    "rage":        "Thunder-PlayAI",
    "frustration": "Briggs-PlayAI",    # firm male
    "disgust":     "Briggs-PlayAI",
    "fear":        "Quinn-PlayAI",     # measured, tense
    "anxiety":     "Quinn-PlayAI",
    "surprise":    "Arista-PlayAI",
    "neutral":     "Atlas-PlayAI",     # clear, neutral male
}
_DEFAULT_VOICE = "Atlas-PlayAI"
_MODEL_ENGLISH = "playai-tts"


def _rate_to_speed(rate: str) -> float:
    """Convert prosody rate string (e.g. '+20%', '-25%') to Groq speed (0.25–4.0)."""
    if not rate or rate == "default":
        return 1.0
    try:
        pct = float(rate.replace("%", "").replace("+", ""))
        speed = 1.0 + pct / 100.0
        return round(max(0.5, min(2.0, speed)), 2)  # conservative clamp
    except Exception:
        return 1.0


class GroqOrpheusTTS(TTSEngine):
    """
    Orpheus English TTS via Groq's playai-tts model.
    Emotion-mapped voices + prosody rate → speed conversion.
    Auto-disables on 401/403 auth errors.
    """

    def __init__(self):
        if not _GROQ_AVAILABLE:
            self.available = False
            logger.warning("Groq SDK not installed — run: pip install groq")
            return

        api_key = getattr(settings, "GROQ_API_KEY", None)
        if not api_key:
            self.available = False
            logger.warning("GroqOrpheusTTS: GROQ_API_KEY not set — Orpheus TTS disabled.")
            return

        try:
            self.client = Groq(api_key=api_key)
            self.available = True
            logger.info(f"Groq Orpheus TTS ready — model={_MODEL_ENGLISH}")
        except Exception as e:
            self.available = False
            logger.error(f"Groq Orpheus TTS init error: {e}")

    async def synthesize(
        self,
        text: str,
        ssml: str,
        filepath: str,
        prosody: Dict[str, str],
        emotion: str = "neutral",
        segment_deltas: list = None,
        detected_lang: Optional[str] = None,
    ) -> str:
        if not self.available:
            raise TTSGenerationError("Groq Orpheus TTS not available.")

        if not filepath.endswith(".mp3"):
            filepath = os.path.splitext(filepath)[0] + ".mp3"

        voice = _EMOTION_VOICE_MAP.get(emotion.lower(), _DEFAULT_VOICE)
        speed = _rate_to_speed(prosody.get("rate", "default"))

        logger.info(
            f"Groq Orpheus TTS: emotion={emotion}, voice={voice}, speed={speed}x"
        )

        def _synth_sync() -> str:
            response = self.client.audio.speech.create(
                model=_MODEL_ENGLISH,
                voice=voice,
                input=text,
                response_format="mp3",
                speed=speed,
            )
            with open(filepath, "wb") as f:
                f.write(response.content)
            return filepath

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, _synth_sync)
            logger.info(f"Groq Orpheus TTS audio saved: {result}")
            return result

        except Exception as e:
            err = str(e)
            if any(x in err.lower() for x in ("401", "403", "unauthorized", "invalid_api_key")):
                self.available = False
                logger.warning(f"Groq Orpheus TTS disabled for session: {err[:400]}")
            raise TTSGenerationError(f"Groq Orpheus TTS error: {err[:500]}")
