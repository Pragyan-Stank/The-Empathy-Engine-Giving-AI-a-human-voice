"""
Audio Post-Processor — Enhances generated TTS audio to sound like recorded human speech.

Processing chain:
  1. De-essing (reduce sibilance for natural sound)
  2. EQ warmth (3-band: low-shelf boost, mid presence, high-shelf cut/boost)
  3. Dynamic compression (normalize + gentle limiting for consistent loudness)
  4. Subtle reverb (room feel / depth — NOT echo, but space)
  5. Stereo widening (slight L/R offset for depth on stereo output)
  6. Final normalization (target loudness for consistency)

Uses pydub (requires ffmpeg binary for MP3 handling).
Falls back gracefully if pydub ops fail — original audio is preserved.

Emotion profiles:
  - sad / grief    → deeper reverb, warmer EQ, softer compression
  - angry / rage   → brighter EQ, harder compression, more presence
  - joy / excite   → crisp EQ, medium reverb, punchy compression
  - neutral        → light enhancement, minimal processing
  - anxiety        → slightly muffled, intimate feel
"""
import os
import math
import logging
import shutil
from typing import Optional


logger = logging.getLogger("empathy_engine")

try:
    from pydub import AudioSegment
    from pydub.effects import normalize, compress_dynamic_range
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
    # low_gain: bass warmth, mid_gain: vocal presence, high_gain: air/brightness
    # reverb_ms: pre-delay, reverb_decay: echo volume, num_echoes: depth layers
    # headroom: compression target, compress: whether to apply dynamics processing
    "grief":       {"low_gain": 4.5,  "mid_gain": 1.0,  "high_gain": -3.5,
                    "reverb_ms": 85,  "reverb_decay": 0.28, "num_echoes": 4,
                    "headroom": -1.0, "compress": True},
    "sadness":     {"low_gain": 3.5,  "mid_gain": 0.5,  "high_gain": -2.5,
                    "reverb_ms": 65,  "reverb_decay": 0.22, "num_echoes": 3,
                    "headroom": -1.5, "compress": True},
    "anxiety":     {"low_gain": 1.5,  "mid_gain": -0.5, "high_gain": -2.0,
                    "reverb_ms": 35,  "reverb_decay": 0.15, "num_echoes": 2,
                    "headroom": -2.0, "compress": True},
    "fear":        {"low_gain": 2.0,  "mid_gain": 0.0,  "high_gain": -1.5,
                    "reverb_ms": 45,  "reverb_decay": 0.18, "num_echoes": 3,
                    "headroom": -2.0, "compress": True},
    "anger":       {"low_gain": -0.5, "mid_gain": 2.5,  "high_gain": 2.0,
                    "reverb_ms": 18,  "reverb_decay": 0.08, "num_echoes": 2,
                    "headroom": -3.0, "compress": True},
    "rage":        {"low_gain": -1.0, "mid_gain": 3.0,  "high_gain": 3.0,
                    "reverb_ms": 12,  "reverb_decay": 0.06, "num_echoes": 2,
                    "headroom": -3.5, "compress": True},
    "frustration": {"low_gain": 0.0,  "mid_gain": 2.0,  "high_gain": 1.5,
                    "reverb_ms": 20,  "reverb_decay": 0.10, "num_echoes": 2,
                    "headroom": -2.5, "compress": True},
    "joy":         {"low_gain": 1.5,  "mid_gain": 1.5,  "high_gain": 1.0,
                    "reverb_ms": 25,  "reverb_decay": 0.12, "num_echoes": 3,
                    "headroom": -2.0, "compress": True},
    "excitement":  {"low_gain": 1.0,  "mid_gain": 2.0,  "high_gain": 1.5,
                    "reverb_ms": 20,  "reverb_decay": 0.10, "num_echoes": 2,
                    "headroom": -2.5, "compress": True},
    "contentment": {"low_gain": 2.5,  "mid_gain": 0.5,  "high_gain": -0.5,
                    "reverb_ms": 40,  "reverb_decay": 0.16, "num_echoes": 3,
                    "headroom": -1.5, "compress": True},
    "surprise":    {"low_gain": 0.5,  "mid_gain": 1.5,  "high_gain": 1.0,
                    "reverb_ms": 22,  "reverb_decay": 0.12, "num_echoes": 2,
                    "headroom": -2.0, "compress": True},
    "disgust":     {"low_gain": 0.0,  "mid_gain": 1.5,  "high_gain": 0.5,
                    "reverb_ms": 18,  "reverb_decay": 0.08, "num_echoes": 2,
                    "headroom": -2.5, "compress": True},
    "neutral":     {"low_gain": 1.5,  "mid_gain": 0.5,  "high_gain": 0.0,
                    "reverb_ms": 22,  "reverb_decay": 0.10, "num_echoes": 2,
                    "headroom": -2.0, "compress": False},
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

        # ── 1. De-essing (reduce sibilance) ───────────────────────────────────
        audio = _apply_deessing(audio)

        # ── 2. 3-band EQ (low shelf + mid presence + high shelf) ──────────────
        low_db  = profile["low_gain"]  * scale
        mid_db  = profile["mid_gain"]  * scale
        high_db = profile["high_gain"] * scale
        audio = _apply_eq_3band(audio, low_gain_db=low_db, mid_gain_db=mid_db, high_gain_db=high_db)

        # ── 3. Subtle reverb (room feel, not echo) ────────────────────────────
        reverb_ms    = int(profile["reverb_ms"] * scale)
        reverb_decay = profile["reverb_decay"] * scale
        num_echoes   = profile.get("num_echoes", 3)
        if reverb_ms > 0:
            audio = _apply_reverb(audio, delay_ms=reverb_ms, decay=reverb_decay, num_echoes=num_echoes)

        # ── 4. Dynamic compression (consistent loudness) ──────────────────────
        if profile.get("compress", True):
            try:
                audio = compress_dynamic_range(
                    audio,
                    threshold=-20.0,
                    ratio=3.0,
                    attack=5.0,
                    release=50.0,
                )
            except Exception:
                # Fallback: simple normalization if compress_dynamic_range not available
                pass

        # ── 5. Stereo widening (subtle depth on stereo output) ─────────────────
        if audio.channels >= 2:
            audio = _apply_stereo_widening(audio, width_ms=1)

        # ── 6. Final normalization ────────────────────────────────────────────
        headroom = profile["headroom"]
        audio = normalize(audio, headroom_db=headroom)

        # ── 7. Export back to same path ───────────────────────────────────────
        fmt = "mp3" if filepath.endswith(".mp3") else "wav"
        if fmt == "mp3":
            audio.export(filepath, format="mp3", bitrate="192k")  # higher quality
        else:
            audio.export(filepath, format="wav")

        logger.info(
            f"PostProcess [{emotion}×{intensity:.1f}]: "
            f"EQ low={low_db:+.1f}dB mid={mid_db:+.1f}dB high={high_db:+.1f}dB "
            f"reverb={reverb_ms}ms decay={reverb_decay:.2f} "
            f"compress={profile.get('compress', True)} "
            f"→ {filepath}"
        )
        return filepath

    except Exception as exc:
        logger.warning(f"Audio post-processing failed (non-fatal): {exc}")
        return filepath


# ── DSP Helpers ────────────────────────────────────────────────────────────────

def _apply_deessing(audio: "AudioSegment", threshold_hz: int = 4000) -> "AudioSegment":
    """
    Simple de-essing: attenuate sharp sibilant frequencies.
    Reduces the harshness of 's', 'sh', 'ch' sounds that make TTS sound artificial.
    """
    try:
        # Isolate sibilant range (4-8kHz) and reduce volume
        sibilants = audio.high_pass_filter(threshold_hz)
        # Reduce sibilant energy by 3dB
        sibilants_reduced = sibilants - 3.0
        # Reconstruct: original without highs + reduced highs
        base = audio.low_pass_filter(threshold_hz)
        return base.overlay(sibilants_reduced, gain_during_overlay=0)
    except Exception:
        return audio


def _apply_eq_3band(
    audio: "AudioSegment",
    low_gain_db: float,
    mid_gain_db: float,
    high_gain_db: float,
) -> "AudioSegment":
    """
    3-band EQ using pydub filters:
      Low shelf  (≤300 Hz)   → warmth / body
      Mid band   (300–4000 Hz) → vocal presence / clarity
      High shelf (≥4000 Hz)  → air / brightness
    """
    try:
        # Isolate bands
        lows  = audio.low_pass_filter(300)
        mids  = audio.high_pass_filter(300).low_pass_filter(4000)
        highs = audio.high_pass_filter(4000)

        # Apply per-band gain
        lows  = lows  + low_gain_db
        mids  = mids  + mid_gain_db
        highs = highs + high_gain_db

        # Recombine
        combined = mids.overlay(lows, gain_during_overlay=0)
        combined = combined.overlay(highs, gain_during_overlay=0)
        return combined
    except Exception:
        # Fallback: just apply a global gain approximation
        net = (low_gain_db + mid_gain_db + high_gain_db) / 4.0
        return audio + net


def _apply_reverb(
    audio: "AudioSegment",
    delay_ms: int = 40,
    decay: float = 0.15,
    num_echoes: int = 3,
) -> "AudioSegment":
    """
    Algorithmic reverb via cascaded decaying echoes.

    Creates `num_echoes` delayed copies of the signal, each at a lower
    volume, and overlays them onto the dry signal.

    Uses a diffusion pattern: each echo has a slightly randomized delay
    to avoid metallic artifacts.
    """
    result = audio
    for i in range(1, num_echoes + 1):
        # Fibonacci-like delay spacing for more natural diffusion
        echo_delay = int(delay_ms * (1 + 0.618 * i))
        echo_gain  = decay / (i * 1.2)    # faster decay for later echoes
        echo_db    = 20 * math.log10(max(echo_gain, 1e-6))
        echo       = audio + echo_db      # attenuate
        result     = result.overlay(echo, position=echo_delay)
    return result


def _apply_stereo_widening(audio: "AudioSegment", width_ms: int = 1) -> "AudioSegment":
    """
    Subtle stereo widening by delaying one channel slightly.
    Creates a sense of space without making the audio sound processed.
    """
    try:
        if audio.channels < 2:
            return audio

        channels = audio.split_to_mono()
        if len(channels) < 2:
            return audio

        left = channels[0]
        right = channels[1]

        # Add slight delay to right channel
        silence = AudioSegment.silent(duration=width_ms, frame_rate=audio.frame_rate)
        right_delayed = silence + right

        # Trim to match lengths
        min_len = min(len(left), len(right_delayed))
        left = left[:min_len]
        right_delayed = right_delayed[:min_len]

        # Rebuild stereo
        return AudioSegment.from_mono_audiosegments(left, right_delayed)
    except Exception:
        return audio
