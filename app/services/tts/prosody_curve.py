"""
Prosody Curve Engine — models human speech dynamics across segments.

CORE CONCEPT:
  Real human speech has a CONSISTENT character (you always know it's the same
  person speaking) but INTENTIONAL VARIATION within each phrase/sentence.

  This module computes per-segment prosody by combining:
    Base prosody = emotional character of the whole speech (set at request level)
    Segment delta = intentional variation per phrase (from LLM analysis)
    Combined = base + delta, always clamped to safe ranges

  Supports BOTH Edge TTS (Hz/%) and Google Cloud TTS (rate multiplier/semitones/dB)
  output formats from the same internal representation.

DELIVERY ARC CONCEPT:
  Every speech has a natural shape:
    Opening:  moderate pace, building attention
    Climax:   peak energy, fastest, most expressive
    Closing:  slowing, trailing off, reflective

  The arc_position field on each segment influences how aggressively
  deltas are applied.

SAFE RANGES:
  Edge TTS:   rate -25% to +75%, pitch -20Hz to +20Hz, volume -90% to +90%
  Google TTS: rate 0.25x to 4.0x, pitch -20st to +20st, volume -12dB to +12dB
"""
import math
import re
from typing import Dict, List, Optional, Tuple


# ── Safe operating ranges ─────────────────────────────────────────────────────
# Edge TTS
_EDGE_RATE_MIN   = -25
_EDGE_RATE_MAX   = +75
_EDGE_PITCH_MIN  = -20
_EDGE_PITCH_MAX  = +20
_EDGE_VOL_MIN    = -50
_EDGE_VOL_MAX    = +50

# Google Cloud TTS
_GCLOUD_RATE_MIN   = 0.25
_GCLOUD_RATE_MAX   = 4.0
_GCLOUD_PITCH_MIN  = -20.0
_GCLOUD_PITCH_MAX  = +20.0
_GCLOUD_VOL_MIN    = -12.0
_GCLOUD_VOL_MAX    = +12.0


def apply_delta(
    base_prosody:   Dict[str, str],
    rate_delta_pct: int   = 0,
    pitch_delta_hz: int   = 0,
    volume_delta_db: float = 0.0,
) -> Dict[str, str]:
    """
    Apply a small delta to the shared base prosody and return a new prosody dict.

    Input:  base_prosody as {"rate": "-25%", "pitch": "-2.0st", "volume": "-3.0dB"}
            + deltas in percentage / Hz / dB
    Output: new prosody dict with the same format, values clamped to safe range.
    """
    base_rate_pct = _parse_rate_pct(base_prosody.get("rate", "+0%"))
    base_pitch_hz = _parse_pitch_hz(base_prosody.get("pitch", "+0Hz"))
    base_vol_pct  = _parse_vol_pct(base_prosody.get("volume", "+0%"))

    # Apply delta and clamp
    new_rate  = max(_EDGE_RATE_MIN,  min(_EDGE_RATE_MAX,  base_rate_pct  + rate_delta_pct))
    new_pitch = max(_EDGE_PITCH_MIN, min(_EDGE_PITCH_MAX, base_pitch_hz  + pitch_delta_hz))
    new_vol   = max(_EDGE_VOL_MIN,   min(_EDGE_VOL_MAX,   base_vol_pct   + _db_to_pct(volume_delta_db)))

    return {
        "rate":   f"+{new_rate}%"  if new_rate  >= 0 else f"{new_rate}%",
        "pitch":  f"+{new_pitch}Hz" if new_pitch >= 0 else f"{new_pitch}Hz",
        "volume": f"+{int(new_vol)}%" if new_vol >= 0 else f"{int(new_vol)}%",
    }


def build_segment_prosodies(
    base_prosody: Dict[str, str],
    segment_deltas: List[Dict],
) -> List[Dict[str, str]]:
    """
    Build a list of per-segment prosody dicts from base + deltas.

    segment_deltas: list of dicts with keys:
      rate_delta_pct, pitch_delta_hz, volume_delta_db
      (all optional, default 0)

    Returns one prosody dict per segment.
    """
    result = []
    for delta in segment_deltas:
        seg_prosody = apply_delta(
            base_prosody,
            rate_delta_pct  = int(delta.get("rate_delta_pct",  0)),
            pitch_delta_hz  = int(delta.get("pitch_delta_hz",  0)),
            volume_delta_db = float(delta.get("volume_delta_db", 0.0)),
        )
        result.append(seg_prosody)
    return result


def edge_tts_format(prosody: Dict[str, str]) -> Tuple[str, str, str]:
    """
    Convert a prosody dict to the three strings Edge TTS Communicate() expects:
    Returns (rate, pitch, volume) all as Edge TTS formatted strings.

    Note: base_prosody values may be in semitone/dB format (from calculate_prosody).
    This function normalises to Hz/% format if needed.
    """
    rate_str   = _normalise_rate(prosody.get("rate",   "+0%"))
    pitch_str  = _normalise_pitch(prosody.get("pitch", "+0Hz"))
    volume_str = _normalise_volume(prosody.get("volume", "+0%"))
    return rate_str, pitch_str, volume_str


def google_tts_format(prosody: Dict[str, str]) -> Tuple[float, float, float]:
    """
    Convert a prosody dict to Google Cloud TTS audioConfig values.

    Returns (speaking_rate, pitch_semitones, volume_gain_db):
      - speaking_rate: 0.25 to 4.0 (1.0 = normal)
      - pitch: -20.0 to +20.0 semitones
      - volume_gain_db: -12.0 to +12.0 dB
    """
    # Rate: percentage → multiplier
    rate_pct = _parse_rate_pct(prosody.get("rate", "+0%"))
    rate_mult = round(max(_GCLOUD_RATE_MIN, min(_GCLOUD_RATE_MAX, 1.0 + rate_pct / 100.0)), 2)

    # Pitch: Hz → semitones (approx: 1st ≈ 8.5Hz for speech)
    pitch_hz = _parse_pitch_hz(prosody.get("pitch", "+0Hz"))
    # If input is already in semitones, use directly
    pitch_raw = prosody.get("pitch", "+0Hz")
    if "st" in str(pitch_raw):
        try:
            pitch_st = float(str(pitch_raw).replace("st", "").replace("+", ""))
        except Exception:
            pitch_st = pitch_hz / 8.5
    else:
        pitch_st = pitch_hz / 8.5
    pitch_st = round(max(_GCLOUD_PITCH_MIN, min(_GCLOUD_PITCH_MAX, pitch_st)), 1)

    # Volume: percentage → dB
    vol_pct = _parse_vol_pct(prosody.get("volume", "+0%"))
    # If input is already in dB, use directly
    vol_raw = prosody.get("volume", "+0%")
    if "dB" in str(vol_raw):
        try:
            vol_db = float(str(vol_raw).replace("dB", "").replace("+", ""))
        except Exception:
            vol_db = _pct_to_db(vol_pct)
    else:
        vol_db = _pct_to_db(vol_pct)
    vol_db = round(max(_GCLOUD_VOL_MIN, min(_GCLOUD_VOL_MAX, vol_db)), 1)

    return rate_mult, pitch_st, vol_db


def google_tts_format_from_deltas(
    base_prosody: Dict[str, str],
    rate_delta_pct: int = 0,
    pitch_delta_hz: int = 0,
    volume_delta_db: float = 0.0,
) -> Tuple[float, float, float]:
    """
    Directly compute Google TTS values from base prosody + deltas.
    More accurate than going through the Edge TTS intermediate format.

    Returns (speaking_rate, pitch_semitones, volume_gain_db).
    """
    # Parse base from the prosody dict (handles both st/Hz and %/dB formats)
    base_rate = _parse_rate_pct(base_prosody.get("rate", "default"))
    base_pitch = _parse_pitch_st(base_prosody.get("pitch", "default"))
    base_vol = _parse_vol_db(base_prosody.get("volume", "default"))

    # Apply deltas
    new_rate_pct = base_rate + rate_delta_pct
    new_pitch_st = base_pitch + (pitch_delta_hz / 8.5)
    new_vol_db   = base_vol + volume_delta_db

    # Clamp to Google TTS ranges
    rate_mult = round(max(_GCLOUD_RATE_MIN, min(_GCLOUD_RATE_MAX, 1.0 + new_rate_pct / 100.0)), 2)
    pitch_st  = round(max(_GCLOUD_PITCH_MIN, min(_GCLOUD_PITCH_MAX, new_pitch_st)), 1)
    vol_db    = round(max(_GCLOUD_VOL_MIN, min(_GCLOUD_VOL_MAX, new_vol_db)), 1)

    return rate_mult, pitch_st, vol_db


# ── Parsers ────────────────────────────────────────────────────────────────────

def _parse_rate_pct(r: str) -> int:
    """Parse "+20%" or "-25%" or "default" → integer percentage."""
    if r in ("default", None, ""):
        return 0
    try:
        return int(float(r.replace("%", "").replace("+", "")))
    except Exception:
        return 0


def _parse_pitch_hz(p: str) -> int:
    """Parse "+0Hz"  or "-15Hz" or "-2.0st" → integer Hz."""
    if p in ("default", None, ""):
        return 0
    try:
        if "Hz" in p:
            return int(float(p.replace("Hz", "").replace("+", "")))
        elif "st" in p:
            # semitones → Hz (1st ≈ 8.5Hz for speech)
            st = float(p.replace("st", "").replace("+", ""))
            return max(_EDGE_PITCH_MIN, min(_EDGE_PITCH_MAX, round(st * 8.5)))
        return 0
    except Exception:
        return 0


def _parse_pitch_st(p: str) -> float:
    """Parse pitch value → semitones (float)."""
    if p in ("default", None, ""):
        return 0.0
    try:
        if "st" in p:
            return float(p.replace("st", "").replace("+", ""))
        elif "Hz" in p:
            hz = float(p.replace("Hz", "").replace("+", ""))
            return hz / 8.5
        return 0.0
    except Exception:
        return 0.0


def _parse_vol_db(v: str) -> float:
    """Parse volume value → dB (float)."""
    if v in ("default", None, ""):
        return 0.0
    try:
        if "dB" in v:
            return float(v.replace("dB", "").replace("+", ""))
        elif "%" in v:
            pct = float(v.replace("%", "").replace("+", ""))
            return _pct_to_db(int(pct))
        return 0.0
    except Exception:
        return 0.0


def _parse_vol_pct(v: str) -> int:
    """Parse "+0%" or "-29%" or "-3.0dB" → integer percentage."""
    if v in ("default", None, ""):
        return 0
    try:
        if "%" in v:
            return int(float(v.replace("%", "").replace("+", "")))
        elif "dB" in v:
            db = float(v.replace("dB", "").replace("+", ""))
            return _db_to_pct(db)
        return 0
    except Exception:
        return 0


def _db_to_pct(db: float) -> int:
    """Convert dB gain to approximate percentage shift."""
    try:
        pct = round((math.pow(10, db / 20) - 1) * 100)
        return max(-90, min(90, pct))
    except Exception:
        return 0


def _pct_to_db(pct: int) -> float:
    """Convert percentage gain to approximate dB."""
    try:
        if pct <= -100:
            return -96.0
        ratio = 1.0 + pct / 100.0
        if ratio <= 0:
            return -96.0
        return round(20 * math.log10(ratio), 1)
    except Exception:
        return 0.0


def _normalise_rate(r: str) -> str:
    """Ensure rate is in Edge TTS format (percentage string) within safe range."""
    pct = _parse_rate_pct(r)
    pct = max(_EDGE_RATE_MIN, min(_EDGE_RATE_MAX, pct))
    return f"+{pct}%" if pct >= 0 else f"{pct}%"


def _normalise_pitch(p: str) -> str:
    """Ensure pitch is in Edge TTS Hz format within safe range."""
    hz = _parse_pitch_hz(p)
    hz = max(_EDGE_PITCH_MIN, min(_EDGE_PITCH_MAX, hz))
    return f"+{hz}Hz" if hz >= 0 else f"{hz}Hz"


def _normalise_volume(v: str) -> str:
    """Ensure volume is in Edge TTS percentage format within safe range."""
    pct = _parse_vol_pct(v)
    pct = max(_EDGE_VOL_MIN, min(_EDGE_VOL_MAX, pct))
    return f"+{pct}%" if pct >= 0 else f"{pct}%"
