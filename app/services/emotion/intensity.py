from app.services.emotion.mapping import get_prosody_base
from typing import Dict, Any

def calculate_prosody(emotion: str, intensity: float) -> Dict[str, str]:
    """
    Given an emotion and an intensity factor [0.0 - 1.0], compute the actual SSML / prosody attributes.
    Returns something like {"rate": "+20%", "pitch": "+2st", "volume": "+2dB"}
    """
    intensity = max(0.0, min(1.0, intensity))
    base = get_prosody_base(emotion)
    
    scaled_rate = base["rate_delta"] * intensity
    scaled_pitch = base["pitch_shift"] * intensity
    scaled_vol = base["volume_delta"] * intensity
    
    # Format for SSML
    # Rate: +X% or -X%
    rate_percent = int(scaled_rate * 100)
    rate_str = f"+{rate_percent}%" if rate_percent >= 0 else f"{rate_percent}%"
    if rate_percent == 0:
        rate_str = "default"
    
    # Pitch: +Xst or -Xst
    pitch_st = round(scaled_pitch, 1)
    pitch_str = f"+{pitch_st}st" if pitch_st > 0 else f"{pitch_st}st"
    if pitch_st == 0:
        pitch_str = "default"
        
    # Volume: +XdB or -XdB
    vol_db = round(scaled_vol, 1)
    vol_str = f"+{vol_db}dB" if vol_db > 0 else f"{vol_db}dB"
    if vol_db == 0:
        vol_str = "default"
        
    return {
        "rate": rate_str,
        "pitch": pitch_str,
        "volume": vol_str
    }
