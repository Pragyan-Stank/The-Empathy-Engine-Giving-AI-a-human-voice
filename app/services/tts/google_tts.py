"""
Google Cloud Text-to-Speech via REST API using GOOGLE_API_KEY.
Uses audioConfig for prosody (rate/pitch/volume) so SSML is only
used for emphasis and pauses — avoids double-application of prosody.
"""
import httpx
import base64
import asyncio
from typing import Dict

from app.services.tts.base import TTSEngine
from app.core.exceptions import TTSGenerationError
from app.core.logging_config import logger
from app.core.config import settings

TTS_REST_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"


def _parse_rate(rate_str: str) -> float:
    """'+20%' → 1.2 | '-20%' → 0.8 | 'default' → 1.0"""
    if rate_str in ("default", None, ""):
        return 1.0
    try:
        pct = float(rate_str.replace("%", "").replace("+", ""))
        return round(max(0.25, min(4.0, 1.0 + pct / 100.0)), 2)
    except Exception:
        return 1.0


def _parse_pitch(pitch_str: str) -> float:
    """'+2.0st' → 2.0 | '-2.0st' → -2.0 | 'default' → 0.0"""
    if pitch_str in ("default", None, ""):
        return 0.0
    try:
        return round(float(pitch_str.replace("st", "").replace("+", "")), 1)
    except Exception:
        return 0.0


def _parse_volume(vol_str: str) -> float:
    """+2.0dB' → 2.0 | '-2.0dB' → -2.0 | 'default' → 0.0"""
    if vol_str in ("default", None, ""):
        return 0.0
    try:
        return round(float(vol_str.replace("dB", "").replace("+", "")), 1)
    except Exception:
        return 0.0


class GoogleCloudTTS(TTSEngine):
    def __init__(self):
        self.api_key = settings.GOOGLE_API_KEY
        self.available = bool(self.api_key)
        if not self.available:
            logger.warning("GOOGLE_API_KEY not set. Google TTS will not be used.")
        else:
            logger.info("Google Cloud TTS (REST) initialized with API key.")

    async def synthesize(
        self,
        text: str,
        ssml: str,
        filepath: str,
        prosody: Dict[str, str],
        emotion: str = "neutral",
    ) -> str:
        if not self.available:
            raise TTSGenerationError("Google TTS API key not configured.")

        speaking_rate = _parse_rate(prosody.get("rate", "default"))
        pitch = _parse_pitch(prosody.get("pitch", "default"))
        volume_gain_db = _parse_volume(prosody.get("volume", "default"))

        logger.info(
            f"Google TTS: rate={speaking_rate}, pitch={pitch}st, volume={volume_gain_db}dB"
        )

        payload = {
            "input": {"ssml": ssml},
            "voice": {
                "languageCode": "en-US",
                "name": "en-US-Neural2-D",
                "ssmlGender": "MALE",
            },
            "audioConfig": {
                "audioEncoding": "MP3",
                "speakingRate": speaking_rate,
                "pitch": pitch,
                "volumeGainDb": volume_gain_db,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    TTS_REST_URL,
                    params={"key": self.api_key},
                    json=payload,
                )
                if response.status_code != 200:
                    detail = response.text
                    logger.error(f"Google TTS API error {response.status_code}: {detail}")
                    # 403 = API not enabled or billing disabled — mark permanently unavailable
                    if response.status_code in (401, 403):
                        self.available = False
                        logger.warning(
                            "Google TTS permanently disabled for this session "
                            f"(HTTP {response.status_code}). Will use next provider."
                        )
                    raise TTSGenerationError(f"Google TTS returned {response.status_code}: {detail}")

                data = response.json()
                audio_bytes = base64.b64decode(data["audioContent"])

                with open(filepath, "wb") as f:
                    f.write(audio_bytes)

                logger.info(f"Google TTS audio saved to {filepath} ({len(audio_bytes)} bytes)")
                return filepath

        except TTSGenerationError:
            raise
        except Exception as e:
            logger.error(f"Google TTS REST call failed: {e}", exc_info=True)
            raise TTSGenerationError(f"Google TTS request failed: {e}")
