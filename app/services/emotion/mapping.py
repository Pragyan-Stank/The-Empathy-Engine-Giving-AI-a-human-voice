from typing import Dict, Any

# Map descriptive labels to a standardized base emotion
EMOTION_ALIASES = {
    "happy": "joy",
    "joy": "joy",
    "positive": "joy",
    
    "sad": "sadness",
    "sadness": "sadness",
    
    "anger": "anger",
    "angry": "anger",
    "frustrated": "anger",
    
    "fear": "fear",
    "anxiety": "fear",
    
    "surprise": "surprise",
    
    "neutral": "neutral",
    "calm": "neutral",
}

# The 'voice style' rules (Base Mapping)
PROSODY_MAP = {
    "joy": {
        "rate_delta": 0.2,       # +20%
        "pitch_shift": 2,        # +2 st
        "volume_delta": 2.0      # +2dB
    },
    "sadness": {
        "rate_delta": -0.2,      # -20%
        "pitch_shift": -2,       # -2 st
        "volume_delta": -2.0     # -2dB
    },
    "anger": {
        "rate_delta": 0.3,       # +30%
        "pitch_shift": 4,        # +4 st
        "volume_delta": 4.0      # +4dB
    },
    "fear": {
        "rate_delta": 0.1,       # +10%
        "pitch_shift": 3,        # +3 st
        "volume_delta": 1.0      # +1dB
    },
    "surprise": {
        "rate_delta": 0.4,       # +40%
        "pitch_shift": 5,        # +5 st
        "volume_delta": 0.0      # 0dB
    },
    "neutral": {
        "rate_delta": 0.0,       
        "pitch_shift": 0,        
        "volume_delta": 0.0      
    }
}

def get_base_emotion(label: str) -> str:
    lbl = label.lower().strip()
    return EMOTION_ALIASES.get(lbl, "neutral")

def get_prosody_base(emotion: str) -> Dict[str, float]:
    base = get_base_emotion(emotion)
    return PROSODY_MAP.get(base, PROSODY_MAP["neutral"])
