"""
ElevenLabs TTS — Primary expressive synthesis engine.

Unlike Edge TTS (which modulates rate/pitch/volume numerically), ElevenLabs
maps emotion directly to voice character via:
  - stability       (0=variable/expressive  ↔  1=stable/monotone)
  - style           (0=neutral              ↔  1=highly styled)
  - similarity_boost(0=allow deviation      ↔  1=strict to voice character)

This produces genuinely emotion-aware speech without needing SSML prosody tags.
API docs: https://api.elevenlabs.io/docs
"""
import httpx
from typing import Dict

from app.services.tts.base import TTSEngine
from app.core.exceptions import TTSGenerationError
from app.core.logging_config import logger
from app.core.config import settings

ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/text-to-speech"
MODEL_ID = "eleven_multilingual_v2"

# Charlie: highly expressive, natural male American voice
# Free tier alternatives: Adam (pNInz6obpgDQGcFmaJgB), Bella (EXAVITQu4vr4xnSDxMaL)
DEFAULT_VOICE_ID = "IKne3meq5aSn9XLyUdCD"  # Charlie

# ── Emotion → Voice Settings Map ─────────────────────────────────────────────
# Each entry defines the character of the voice for a given emotion.
# style and stability are the most impactful levers.
VOICE_SETTINGS_MAP: Dict[str, Dict] = {
    # ── Positive ──────────────────────────────────────────────────────────
    "joy": {
        "stability": 0.35, "similarity_boost": 0.80,
        "style": 0.65,     "use_speaker_boost": True,
    },
    "excitement": {
        "stability": 0.15, "similarity_boost": 0.85,
        "style": 0.95,     "use_speaker_boost": True,
    },
    "contentment": {
        "stability": 0.70, "similarity_boost": 0.75,
        "style": 0.25,     "use_speaker_boost": False,
    },
    # ── Negative ──────────────────────────────────────────────────────────
    "sadness": {
        "stability": 0.75, "similarity_boost": 0.60,
        "style": 0.15,     "use_speaker_boost": False,
    },
    "grief": {
        "stability": 0.85, "similarity_boost": 0.55,
        "style": 0.05,     "use_speaker_boost": False,
    },
    "anger": {
        "stability": 0.20, "similarity_boost": 0.90,
        "style": 0.80,     "use_speaker_boost": True,
    },
    "rage": {
        "stability": 0.05, "similarity_boost": 0.95,
        "style": 1.00,     "use_speaker_boost": True,
    },
    "frustration": {
        "stability": 0.35, "similarity_boost": 0.80,
        "style": 0.60,     "use_speaker_boost": True,
    },
    "disgust": {
        "stability": 0.40, "similarity_boost": 0.70,
        "style": 0.55,     "use_speaker_boost": True,
    },
    # ── Fearful ────────────────────────────────────────────────────────────
    "fear": {
        "stability": 0.25, "similarity_boost": 0.75,
        "style": 0.55,     "use_speaker_boost": True,
    },
    "anxiety": {
        "stability": 0.20, "similarity_boost": 0.75,
        "style": 0.65,     "use_speaker_boost": True,
    },
    # ── Other ──────────────────────────────────────────────────────────────
    "surprise": {
        "stability": 0.10, "similarity_boost": 0.85,
        "style": 0.90,     "use_speaker_boost": True,
    },
    "neutral": {
        "stability": 0.75, "similarity_boost": 0.75,
        "style": 0.00,     "use_speaker_boost": False,
    },
}

_DEFAULT_SETTINGS = VOICE_SETTINGS_MAP["neutral"]


class ElevenLabsTTS(TTSEngine):
    """ElevenLabs TTS with emotion-mapped voice settings."""

    def __init__(self):
        self.api_key = settings.ELEVEN_LABS
        self.available = bool(self.api_key)
        if not self.available:
            logger.warning("ELEVEN_LABS key not set. ElevenLabs TTS is disabled.")
        else:
            logger.info(f"ElevenLabs TTS ready — voice: {DEFAULT_VOICE_ID}, model: {MODEL_ID}")

    async def synthesize(
        self,
        text: str,
        ssml: str,
        filepath: str,
        prosody: Dict[str, str],
        emotion: str = "neutral",
    ) -> str:
        if not self.available:
            raise TTSGenerationError("ElevenLabs API key not configured.")

        voice_settings = VOICE_SETTINGS_MAP.get(emotion.lower(), _DEFAULT_SETTINGS).copy()

        logger.info(
            f"ElevenLabs: emotion={emotion}, "
            f"stability={voice_settings['stability']}, "
            f"style={voice_settings['style']}"
        )

        payload = {
            "text": text,
            "model_id": MODEL_ID,
            "voice_settings": voice_settings,
        }

        try:
            async with httpx.AsyncClient(timeout=25.0) as client:
                response = await client.post(
                    f"{ELEVENLABS_API_URL}/{DEFAULT_VOICE_ID}",
                    headers={
                        "xi-api-key": self.api_key,
                        "Content-Type": "application/json",
                        "Accept": "audio/mpeg",
                    },
                    json=payload,
                )

                if response.status_code != 200:
                    detail = response.text[:300]
                    logger.error(f"ElevenLabs API error {response.status_code}: {detail}")
                    # Permanent failures — disable provider for this session
                    if response.status_code in (401, 403):
                        self.available = False
                        logger.warning(
                            "ElevenLabs permanently disabled for this session "
                            f"(HTTP {response.status_code}). Will use next provider."
                        )
                    raise TTSGenerationError(
                        f"ElevenLabs returned {response.status_code}: {detail}"
                    )

                with open(filepath, "wb") as f:
                    f.write(response.content)

                logger.info(
                    f"ElevenLabs audio saved: {filepath} ({len(response.content):,} bytes)"
                )
                return filepath

        except TTSGenerationError:
            raise
        except Exception as e:
            logger.error(f"ElevenLabs request failed: {e}", exc_info=True)
            raise TTSGenerationError(f"ElevenLabs error: {e}")
