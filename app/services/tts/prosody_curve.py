"""
Prosody Curve Engine — models human speech dynamics across segments.

CORE CONCEPT:
  Real human speech has a CONSISTENT character (you always know it's the same
  person speaking) but SUBTLE VARIATION within each phrase/sentence.

  This is fundamentally different from either:
    A) Flat prosody: same rate/pitch for every segment (robotic)
    B) Per-sentence independent prosody: wild jumps (the bug we fixed)

  Instead:
    Base prosody = emotional character of the whole speech (set at request level)
    Segment delta = small INTENTIONAL variation per phrase (from LLM analysis)
    Combined = base + delta, always clamped to Edge TTS safe range

EXAMPLE (grief speech, 3 segments):
  Base: rate=-25%, pitch=-15Hz, vol=-10%

  Seg1 (most somber opening):
    Delta: rate=-3%, pitch=-3Hz, pause_before=0ms
    Result: rate=-28%, pitch=-18Hz   (deeper, slower)

  Seg2 (slight emotional lift):
    Delta: rate=+4%, pitch=+2Hz, pause_before=500ms
    Result: rate=-21%, pitch=-13Hz   (breath, then slight shift)

  Seg3 (trailing off):
    Delta: rate=-2%, pitch=-1Hz, pause_before=300ms
    Result: rate=-27%, pitch=-16Hz   (trails off at end)

SAFE RANGES (Edge TTS):
  rate:   -25% to +75%  (total, after applying delta)
  pitch:  -20Hz to +20Hz
  volume: -90% to +90%
"""
import math
import re
from typing import Dict, List, Optional, Tuple


# ── Safe operating ranges for Edge TTS ────────────────────────────────────────
_RATE_MIN   = -25
_RATE_MAX   = +75
_PITCH_MIN  = -20
_PITCH_MAX  = +20
_VOL_MIN    = -50
_VOL_MAX    = +50


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
    new_rate  = max(_RATE_MIN,  min(_RATE_MAX,  base_rate_pct  + rate_delta_pct))
    new_pitch = max(_PITCH_MIN, min(_PITCH_MAX, base_pitch_hz  + pitch_delta_hz))
    new_vol   = max(_VOL_MIN,   min(_VOL_MAX,   base_vol_pct   + _db_to_pct(volume_delta_db)))

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


# ── Parsers ────────────────────────────────────────────────────────────────────

def _parse_rate_pct(r: str) -> int:
    """Parse "+20%" or "-25%" → integer percentage."""
    try:
        return int(float(r.replace("%", "").replace("+", "")))
    except Exception:
        return 0


def _parse_pitch_hz(p: str) -> int:
    """Parse "+0Hz"  or "-15Hz" or "-2.0st" → integer Hz."""
    try:
        if "Hz" in p:
            return int(float(p.replace("Hz", "").replace("+", "")))
        elif "st" in p:
            # semitones → Hz (1st ≈ 8.5Hz for speech)
            st = float(p.replace("st", "").replace("+", ""))
            return max(_PITCH_MIN, min(_PITCH_MAX, round(st * 8.5)))
        return 0
    except Exception:
        return 0


def _parse_vol_pct(v: str) -> int:
    """Parse "+0%" or "-29%" or "-3.0dB" → integer percentage."""
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


def _normalise_rate(r: str) -> str:
    """Ensure rate is in Edge TTS format (percentage string) within safe range."""
    pct = _parse_rate_pct(r)
    pct = max(_RATE_MIN, min(_RATE_MAX, pct))
    # If input looks like it was already a percentage string, just reclamp
    # If it looked like a multiplier (e.g. "1.2"), convert
    return f"+{pct}%" if pct >= 0 else f"{pct}%"


def _normalise_pitch(p: str) -> str:
    """Ensure pitch is in Edge TTS Hz format within safe range."""
    hz = _parse_pitch_hz(p)
    hz = max(_PITCH_MIN, min(_PITCH_MAX, hz))
    return f"+{hz}Hz" if hz >= 0 else f"{hz}Hz"


def _normalise_volume(v: str) -> str:
    """Ensure volume is in Edge TTS percentage format within safe range."""
    pct = _parse_vol_pct(v)
    pct = max(_VOL_MIN, min(_VOL_MAX, pct))
    return f"+{pct}%" if pct >= 0 else f"{pct}%"
