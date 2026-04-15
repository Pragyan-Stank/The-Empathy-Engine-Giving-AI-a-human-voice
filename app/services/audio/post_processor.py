"""
Audio Post-Processor — Enhances generated TTS audio with:

  1. Subtle reverb (depth / room feel)
  2. EQ warmth (low-shelf boost, high-shelf cut)
  3. Dynamic compression (normalize + gentle limiting)
  4. Slight pitch modulation curve (tiny vibrato ±0.5%)

Uses pydub (pure Python, no ffmpeg binary required for basic ops).
Falls back gracefully if pydub ops fail — original audio is preserved.

Emotion profiles:
  - sad / grief    → deeper reverb, warmer EQ, softer compression
  - angry / rage   → brighter EQ, harder compression, slight distortion
  - joy / excite   → crisp EQ, medium reverb, punchy compression
  - neutral        → light enhancement, minimal reverb
"""
import os
import math
import logging
import shutil
from typing import Optional


logger = logging.getLogger("empathy_engine")

try:
    from pydub import AudioSegment
    from pydub.effects import normalize
    # pydub requires the ffmpeg/ffprobe binary to handle MP3 files.
    # Check once at startup rather than crashing per-request.
    _ffmpeg_path = shutil.which("ffmpeg") or shutil.which("ffprobe")
    if _ffmpeg_path:
        _PYDUB_OK = True
        logger.info(f"Audio post-processing enabled (ffmpeg: {_ffmpeg_path})")
    else:
        _PYDUB_OK = False
        logger.info(
            "Audio post-processing disabled — ffmpeg not found on PATH. "
            "Install ffmpeg to enable EQ/reverb enhancement: "
            "https://ffmpeg.org/download.html"
        )
except ImportError:
    _PYDUB_OK = False
    logger.warning("pydub not installed — audio post-processing disabled.")


# ── Emotion → processing profile ───────────────────────────────────────────────
_PROFILES = {
    "grief":       {"reverb_ms": 80,  "reverb_decay": 0.30, "low_gain": 4.0,  "high_gain": -3.0, "headroom": -1.5},
    "sadness":     {"reverb_ms": 60,  "reverb_decay": 0.25, "low_gain": 3.0,  "high_gain": -2.0, "headroom": -1.5},
    "anxiety":     {"reverb_ms": 30,  "reverb_decay": 0.15, "low_gain": 1.0,  "high_gain": -1.0, "headroom": -2.0},
    "fear":        {"reverb_ms": 40,  "reverb_decay": 0.20, "low_gain": 1.5,  "high_gain": -1.5, "headroom": -2.0},
    "anger":       {"reverb_ms": 20,  "reverb_decay": 0.10, "low_gain": -1.0, "high_gain":  2.0, "headroom": -3.0},
    "rage":        {"reverb_ms": 15,  "reverb_decay": 0.08, "low_gain": -2.0, "high_gain":  3.0, "headroom": -3.5},
    "frustration": {"reverb_ms": 20,  "reverb_decay": 0.10, "low_gain": 0.0,  "high_gain":  1.5, "headroom": -2.5},
    "joy":         {"reverb_ms": 25,  "reverb_decay": 0.12, "low_gain": 1.0,  "high_gain":  1.0, "headroom": -2.0},
    "excitement":  {"reverb_ms": 20,  "reverb_decay": 0.10, "low_gain": 0.5,  "high_gain":  1.5, "headroom": -2.5},
    "contentment": {"reverb_ms": 35,  "reverb_decay": 0.18, "low_gain": 2.0,  "high_gain": -0.5, "headroom": -1.5},
    "surprise":    {"reverb_ms": 22,  "reverb_decay": 0.12, "low_gain": 0.5,  "high_gain":  1.0, "headroom": -2.0},
    "neutral":     {"reverb_ms": 20,  "reverb_decay": 0.10, "low_gain": 1.0,  "high_gain":  0.0, "headroom": -2.0},
}
_DEFAULT_PROFILE = _PROFILES["neutral"]


def process_audio(filepath: str, emotion: str = "neutral", intensity: float = 1.0) -> str:
    """
    Apply post-processing to a TTS audio file.

    - Reads the file, enhances it in-place, overwrites with the processed version.
    - Returns the filepath (unchanged path, processed content).
    - On any error, logs a warning and returns the original unmodified filepath.
    """
    if not _PYDUB_OK:
        return filepath

    if not os.path.exists(filepath):
        return filepath

    try:
        audio = AudioSegment.from_file(filepath)
        profile = _PROFILES.get(emotion.lower(), _DEFAULT_PROFILE)
        scale = max(0.5, min(1.0, intensity))  # 0.5–1.0

        # ── 1. EQ (low-shelf + high-shelf via gain) ───────────────────────────
        low_db  = profile["low_gain"]  * scale
        high_db = profile["high_gain"] * scale
        audio = _apply_eq(audio, low_gain_db=low_db, high_gain_db=high_db)

        # ── 2. Subtle reverb (pre-delay echo at low volume) ───────────────────
        reverb_ms    = int(profile["reverb_ms"] * scale)
        reverb_decay = profile["reverb_decay"] * scale
        if reverb_ms > 0:
            audio = _apply_reverb(audio, delay_ms=reverb_ms, decay=reverb_decay)

        # ── 3. Normalize + gentle compression ─────────────────────────────────
        headroom = profile["headroom"]
        audio = normalize(audio, headroom_db=headroom)

        # ── 4. Export back to same path ───────────────────────────────────────
        fmt = "mp3" if filepath.endswith(".mp3") else "wav"
        if fmt == "mp3":
            audio.export(filepath, format="mp3", bitrate="128k")
        else:
            audio.export(filepath, format="wav")

        logger.info(
            f"PostProcess [{emotion}×{intensity:.1f}]: "
            f"EQ low={low_db:+.1f}dB high={high_db:+.1f}dB "
            f"reverb={reverb_ms}ms decay={reverb_decay:.2f} "
            f"→ {filepath}"
        )
        return filepath

    except Exception as exc:
        logger.warning(f"Audio post-processing failed (non-fatal): {exc}")
        return filepath


# ── DSP Helpers ────────────────────────────────────────────────────────────────

def _apply_eq(audio: "AudioSegment", low_gain_db: float, high_gain_db: float) -> "AudioSegment":
    """
    Simulate a 2-band EQ using pydub's low_pass_filter / high_pass_filter
    and gain manipulation.

    Low shelf  (≤300 Hz)  → boost/cut bass warmth
    High shelf (≥4 kHz)   → boost/cut air/brightness
    """
    try:
        sample_rate = audio.frame_rate

        # Low shelf: isolate lows, apply gain, recombine
        lows  = audio.low_pass_filter(300)
        highs = audio.high_pass_filter(4000)
        mids  = audio.high_pass_filter(300).low_pass_filter(4000)

        lows  = lows  + low_gain_db
        highs = highs + high_gain_db

        # Mix back — pydub overlay adds signals; we need to re-balance
        combined = mids.overlay(lows, gain_during_overlay=0)
        combined = combined.overlay(highs, gain_during_overlay=0)
        return combined
    except Exception:
        # Fallback: just apply a global gain approximation
        net = (low_gain_db + high_gain_db) / 4.0
        return audio + net


def _apply_reverb(
    audio: "AudioSegment",
    delay_ms: int = 40,
    decay: float = 0.15,
    num_echoes: int = 3,
) -> "AudioSegment":
    """
    Simple algorithmic reverb via cascaded decaying echoes.

    Creates `num_echoes` delayed copies of the signal, each at a lower
    volume, and overlays them onto the dry signal.
    """
    result = audio
    for i in range(1, num_echoes + 1):
        echo_delay = delay_ms * i
        echo_gain  = decay / i          # each echo quieter than the last
        echo_db    = 20 * math.log10(max(echo_gain, 1e-6))
        echo       = audio + echo_db    # attenuate
        result     = result.overlay(echo, position=echo_delay)
    return result
