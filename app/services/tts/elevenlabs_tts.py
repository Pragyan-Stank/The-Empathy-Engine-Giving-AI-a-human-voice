"""
ElevenLabs TTS Engine — official elevenlabs Python SDK (v2.x).

Uses:
  - elevenlabs.client.ElevenLabs for authentication
  - client.text_to_speech.convert() → Iterator[bytes]
  - model_id="eleven_v3" (most expressive, supports style/stability/similarity)
  - Emotion → VoiceSettings mapping for per-emotion expressiveness tuning
  - asyncio executor to run the synchronous SDK call without blocking the event loop

API key: read from settings.ELEVEN_LABS (env var: ELEVEN_LABS)
"""
import os
import asyncio
from typing import Dict

try:
    from elevenlabs.client import ElevenLabs
    from elevenlabs import VoiceSettings
    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False

from app.services.tts.base import TTSEngine
from app.core.exceptions import TTSGenerationError
from app.core.logging_config import logger
from app.core.config import settings

# ── Default voice (Rachel — neutral female, great expressiveness range) ────────
_DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"   # Rachel

# ── Emotion → VoiceSettings ────────────────────────────────────────────────────
# stability:        0 = variable/expressive,  1 = stable/monotone
# similarity_boost: how closely to stay to the original voice character
# style:            0 = restrained,  1 = maximum stylistic expression
_EMOTION_SETTINGS: Dict[str, dict] = {
    "joy":         {"stability": 0.35, "similarity_boost": 0.75, "style": 0.65, "use_speaker_boost": True},
    "excitement":  {"stability": 0.25, "similarity_boost": 0.70, "style": 0.85, "use_speaker_boost": True},
    "contentment": {"stability": 0.55, "similarity_boost": 0.80, "style": 0.40, "use_speaker_boost": True},
    "sadness":     {"stability": 0.60, "similarity_boost": 0.85, "style": 0.50, "use_speaker_boost": False},
    "grief":       {"stability": 0.70, "similarity_boost": 0.90, "style": 0.55, "use_speaker_boost": False},
    "anger":       {"stability": 0.30, "similarity_boost": 0.70, "style": 0.80, "use_speaker_boost": True},
    "frustration": {"stability": 0.35, "similarity_boost": 0.75, "style": 0.70, "use_speaker_boost": True},
    "rage":        {"stability": 0.20, "similarity_boost": 0.65, "style": 0.90, "use_speaker_boost": True},
    "disgust":     {"stability": 0.40, "similarity_boost": 0.75, "style": 0.70, "use_speaker_boost": True},
    "fear":        {"stability": 0.45, "similarity_boost": 0.80, "style": 0.70, "use_speaker_boost": True},
    "anxiety":     {"stability": 0.50, "similarity_boost": 0.80, "style": 0.60, "use_speaker_boost": True},
    "surprise":    {"stability": 0.30, "similarity_boost": 0.70, "style": 0.75, "use_speaker_boost": True},
    "neutral":     {"stability": 0.65, "similarity_boost": 0.85, "style": 0.25, "use_speaker_boost": False},
}
_DEFAULT_SETTINGS = _EMOTION_SETTINGS["neutral"]


class ElevenLabsTTS(TTSEngine):
    """
    ElevenLabs TTS using the official Python SDK.
    Runs the synchronous SDK call in a thread executor so FastAPI's async
    event loop is not blocked.  Auto-disables on 401/403 auth errors.
    """

    def __init__(self):
        if not _SDK_AVAILABLE:
            self.available = False
            logger.warning("ElevenLabs SDK not installed — run: pip install elevenlabs")
            return

        api_key = getattr(settings, "ELEVEN_LABS", None)
        if not api_key:
            self.available = False
            logger.warning("ElevenLabs: ELEVEN_LABS env var not set.")
            return

        try:
            self.client   = ElevenLabs(api_key=api_key)
            self.voice_id = _DEFAULT_VOICE_ID
            self.available = True
            logger.info(
                f"ElevenLabs TTS ready — SDK v2, model=eleven_v3, voice={self.voice_id}"
            )
        except Exception as e:
            self.available = False
            logger.error(f"ElevenLabs init error: {e}")

    async def synthesize(
        self,
        text: str,
        ssml: str,
        filepath: str,
        prosody: Dict[str, str],
        emotion: str = "neutral",
        segment_deltas: list = None,
    ) -> str:
        if not self.available:
            raise TTSGenerationError("ElevenLabs not available.")

        cfg = _EMOTION_SETTINGS.get(emotion.lower(), _DEFAULT_SETTINGS)
        voice_settings = VoiceSettings(
            stability=cfg["stability"],
            similarity_boost=cfg["similarity_boost"],
            style=cfg["style"],
            use_speaker_boost=cfg["use_speaker_boost"],
        )

        if not filepath.endswith(".mp3"):
            filepath = os.path.splitext(filepath)[0] + ".mp3"

        logger.info(
            f"ElevenLabs: emotion={emotion}, "
            f"stability={cfg['stability']}, style={cfg['style']}"
        )

        def _synthesize_sync() -> str:
            """Runs in a thread executor — SDK is synchronous."""
            audio_iter = self.client.text_to_speech.convert(
                voice_id=self.voice_id,
                text=text,
                model_id="eleven_v3",
                output_format="mp3_44100_128",
                voice_settings=voice_settings,
            )
            with open(filepath, "wb") as f:
                for chunk in audio_iter:
                    if chunk:
                        f.write(chunk)
            return filepath

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, _synthesize_sync)
            logger.info(f"ElevenLabs audio saved: {result}")
            return result

        except Exception as e:
            err = str(e)
            # Permanent auth / abuse errors — disable for the session
            if any(x in err.lower() for x in ("401", "403", "unusual", "disabled", "unauthorized")):
                self.available = False
                logger.warning(
                    f"ElevenLabs permanently disabled for this session: {err[:250]}"
                )
            raise TTSGenerationError(f"ElevenLabs SDK error: {err[:300]}")
