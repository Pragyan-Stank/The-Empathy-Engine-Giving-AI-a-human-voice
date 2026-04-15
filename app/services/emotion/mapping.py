from typing import Dict, Any

# ── Emotion Aliases → canonical labels ────────────────────────────────────────
EMOTION_ALIASES = {
    # Base positives
    "happy":       "joy",
    "joy":         "joy",
    "positive":    "joy",
    # Granular positives
    "excitement":  "excitement",
    "excited":     "excitement",
    "contentment": "contentment",
    "content":     "contentment",
    "calm":        "contentment",
    # Sadness family
    "sad":         "sadness",
    "sadness":     "sadness",
    "grief":       "grief",
    "grieving":    "grief",
    # Anger family
    "anger":       "anger",
    "angry":       "anger",
    "frustration": "frustration",
    "frustrated":  "frustration",
    "rage":        "rage",
    "furious":     "rage",
    # Fear family
    "fear":        "fear",
    "anxiety":     "anxiety",
    "anxious":     "anxiety",
    "nervous":     "anxiety",
    # Others
    "surprise":    "surprise",
    "surprised":   "surprise",
    "disgust":     "disgust",
    "disgusted":   "disgust",
    "neutral":     "neutral",
}

# ── Prosody Map (rate/pitch/volume deltas per canonical emotion) ───────────────
# Values are base deltas scaled later by intensity (0.0–1.0).
# rate_delta:   fraction of base WPM (e.g. 0.2 = +20%)
# pitch_shift:  semitones relative to default (e.g. +2 st)
# volume_delta: dB relative to default          (e.g. +2 dB)
PROSODY_MAP = {
    # ── Positives ──────────────────────────────────────────────────────────
    # Happy/Joy  → rate: medium (+20%), pitch: +3st, volume: +2dB
    "joy": {
        "rate_delta": 0.20,   "pitch_shift": 3.0,  "volume_delta":  2.0,
    },
    # Excitement → rate: fast (+40%), pitch: +5st, volume: +4dB
    "excitement": {
        "rate_delta": 0.40,   "pitch_shift": 5.0,  "volume_delta":  4.0,
    },
    # Contentment → rate: slightly slow (-5%), pitch: +1st
    "contentment": {
        "rate_delta": -0.05,  "pitch_shift": 1.0,  "volume_delta": -1.0,
    },
    # ── Sadness family ─────────────────────────────────────────────────────
    # Sad  → rate: slow (-25%), pitch: -2st
    "sadness": {
        "rate_delta": -0.25,  "pitch_shift": -2.0, "volume_delta": -2.0,
    },
    # Grief → rate: very slow (-40%), pitch: -3.5st, volume: -3dB
    "grief": {
        "rate_delta": -0.40,  "pitch_shift": -3.5, "volume_delta": -3.0,
    },
    # ── Anger family ───────────────────────────────────────────────────────
    # Angry → rate: fast (+30%), pitch: +1st, volume: +4dB  (per spec)
    "anger": {
        "rate_delta": 0.30,   "pitch_shift": 1.0,  "volume_delta":  4.0,
    },
    # Frustration → rate: medium-fast (+15%), pitch: +1.5st
    "frustration": {
        "rate_delta": 0.15,   "pitch_shift": 1.5,  "volume_delta":  2.5,
    },
    # Rage → rate: very fast (+50%), pitch: +2st, volume: +5.5dB
    "rage": {
        "rate_delta": 0.50,   "pitch_shift": 2.0,  "volume_delta":  5.5,
    },
    # ── Fear family ────────────────────────────────────────────────────────
    "fear": {
        "rate_delta": 0.15,   "pitch_shift": 2.5,  "volume_delta":  1.0,
    },
    "anxiety": {
        "rate_delta": 0.20,   "pitch_shift": 3.0,  "volume_delta":  1.5,
    },
    # ── Other ──────────────────────────────────────────────────────────────
    "surprise": {
        "rate_delta": 0.35,   "pitch_shift": 4.0,  "volume_delta":  0.0,
    },
    "disgust": {
        "rate_delta": 0.05,   "pitch_shift": -1.0, "volume_delta":  1.0,
    },
    # Neutral → rate: medium (0%), pitch: 0st  (per spec)
    "neutral": {
        "rate_delta": 0.00,   "pitch_shift": 0.0,  "volume_delta":  0.0,
    },
}


def get_canonical_emotion(label: str) -> str:
    """Resolve any alias or variant to a canonical label."""
    return EMOTION_ALIASES.get(label.lower().strip(), "neutral")


def get_prosody_base(emotion: str) -> Dict[str, float]:
    """Return the base prosody deltas for an emotion (resolved through aliases)."""
    canonical = get_canonical_emotion(emotion)
    return PROSODY_MAP.get(canonical, PROSODY_MAP["neutral"])


# ── Legacy compatibility ───────────────────────────────────────────────────────
def get_base_emotion(label: str) -> str:
    return get_canonical_emotion(label)
