"""
Fallback TTS using pyttsx3 (SAPI5 on Windows).
Applies rate and volume from the prosody dict.
Note: SAPI5 does not support pitch — it is silently ignored here.
Output format: WAV (pyttsx3 save_to_file on Windows writes WAV).
"""
import asyncio
import re
import xml.etree.ElementTree as ET
from typing import Dict

import pyttsx3

from app.services.tts.base import TTSEngine
from app.core.logging_config import logger

# pyttsx3 default word-per-minute rate on SAPI5
_BASE_WPM = 180


def _parse_rate_to_wpm(rate_str: str) -> int:
    """Convert '+20%' → 216 WPM | '-20%' → 144 WPM | 'default' → 180 WPM"""
    if rate_str in ("default", None, ""):
        return _BASE_WPM
    try:
        pct = float(rate_str.replace("%", "").replace("+", ""))
        wpm = int(_BASE_WPM * (1.0 + pct / 100.0))
        return max(80, min(400, wpm))
    except Exception:
        return _BASE_WPM


def _parse_volume_to_float(vol_str: str) -> float:
    """Convert '+4dB' → ~1.0 (clamped) | '-4dB' → ~0.6 | 'default' → 1.0"""
    if vol_str in ("default", None, ""):
        return 1.0
    try:
        db = float(vol_str.replace("dB", "").replace("+", ""))
        # Simple linear approximation; clamp to [0.1, 1.0]
        vol = 1.0 + db * 0.05
        return round(max(0.1, min(1.0, vol)), 2)
    except Exception:
        return 1.0


def _strip_tags(xml_text: str) -> str:
    """Extract plain text from SSML / XML."""
    try:
        root = ET.fromstring(xml_text)
        return " ".join(root.itertext()).strip()
    except Exception:
        return re.sub(r"<[^>]+>", "", xml_text).strip()


class FallbackTTS(TTSEngine):
    async def synthesize(
        self,
        text: str,
        ssml: str,
        filepath: str,
        prosody: Dict[str, str],
        emotion: str = "neutral",
    ) -> str:
        wpm = _parse_rate_to_wpm(prosody.get("rate", "default"))
        vol = _parse_volume_to_float(prosody.get("volume", "default"))
        plain_text = _strip_tags(ssml)

        logger.info(
            f"pyttsx3 synthesizing: rate={wpm} WPM, volume={vol}, "
            f"pitch=N/A (unsupported on SAPI5)"
        )

        def _run():
            engine = pyttsx3.init()
            engine.setProperty("rate", wpm)
            engine.setProperty("volume", vol)
            engine.save_to_file(plain_text, filepath)
            engine.runAndWait()
            engine.stop()
            return filepath

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _run)
