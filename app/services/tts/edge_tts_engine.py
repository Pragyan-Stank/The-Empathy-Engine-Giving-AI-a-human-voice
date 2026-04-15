"""
Edge TTS engine — Microsoft Edge neural TTS (free, no API key required).
Supports rate, pitch, and volume modulation with high-quality voices.
Pitch is given in Hz; rate and volume as percentages.

Unit conversions from our internal prosody format:
  rate:   "+20%"   → "+20%"    (pass-through)
  pitch:  "+2.0st" → "+17Hz"  (1 semitone ≈ 8.5 Hz at ~150 Hz fundamental)
  volume: "+2.0dB" → "+26%"   (amplitude ratio: 10^(dB/20) - 1)
"""
import math
import os
from typing import Dict

from app.services.tts.base import TTSEngine
from app.core.exceptions import TTSGenerationError
from app.core.logging_config import logger

try:
    import edge_tts
    _EDGE_AVAILABLE = True
except ImportError:
    _EDGE_AVAILABLE = False

# Best expressive voice for en-US — female (Aria) or male (Davis)
_VOICE = "en-US-AriaNeural"


def _to_edge_rate(rate_str: str) -> str:
    """'+20%' → '+20%'  |  'default' → '+0%'"""
    if rate_str in ("default", None, ""):
        return "+0%"
    return rate_str  # already in correct format


def _to_edge_pitch(pitch_str: str) -> str:
    """'+2.0st' → '+17Hz'  |  'default' → '+0Hz'"""
    if pitch_str in ("default", None, ""):
        return "+0Hz"
    try:
        st = float(pitch_str.replace("st", "").replace("+", ""))
        hz = round(st * 8.5)
        return f"+{hz}Hz" if hz >= 0 else f"{hz}Hz"
    except Exception:
        return "+0Hz"


def _to_edge_volume(vol_str: str) -> str:
    """'+2.0dB' → '+26%'  |  'default' → '+0%'"""
    if vol_str in ("default", None, ""):
        return "+0%"
    try:
        db = float(vol_str.replace("dB", "").replace("+", ""))
        pct = round((math.pow(10, db / 20) - 1) * 100)
        pct = max(-90, min(90, pct))
        return f"+{pct}%" if pct >= 0 else f"{pct}%"
    except Exception:
        return "+0%"


class EdgeTTSEngine(TTSEngine):
    """High-quality neural TTS using Microsoft Edge's online service. No auth required."""

    def __init__(self):
        self.available = _EDGE_AVAILABLE
        if not _EDGE_AVAILABLE:
            logger.warning("edge-tts not installed. Run: pip install edge-tts")
        else:
            logger.info(f"Edge TTS initialized — voice: {_VOICE}")

    async def synthesize(
        self,
        text: str,
        ssml: str,
        filepath: str,
        prosody: Dict[str, str],
    ) -> str:
        if not self.available:
            raise TTSGenerationError("edge-tts library not installed.")

        rate   = _to_edge_rate(prosody.get("rate", "default"))
        pitch  = _to_edge_pitch(prosody.get("pitch", "default"))
        volume = _to_edge_volume(prosody.get("volume", "default"))

        logger.info(
            f"Edge TTS: voice={_VOICE}, rate={rate}, pitch={pitch}, volume={volume}"
        )

        # Ensure output path ends with .mp3
        if not filepath.endswith(".mp3"):
            filepath = os.path.splitext(filepath)[0] + ".mp3"

        try:
            communicate = edge_tts.Communicate(
                text=text,
                voice=_VOICE,
                rate=rate,
                pitch=pitch,
                volume=volume,
            )
            await communicate.save(filepath)
            logger.info(f"Edge TTS audio saved: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Edge TTS synthesis failed: {e}", exc_info=True)
            raise TTSGenerationError(f"Edge TTS error: {e}")
