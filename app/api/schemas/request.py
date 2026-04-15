from pydantic import BaseModel, Field
from typing import Optional


class SynthesizeRequest(BaseModel):
    text: str = Field(
        ..., min_length=1, max_length=2000,
        description="Input text (max 2000 characters)"
    )
    emotion_override: Optional[str] = Field(
        None, description="Force a specific emotion (joy, sadness, anger, fear, surprise, neutral)"
    )
    intensity: float = Field(
        1.0, ge=0.0, le=1.0,
        description="Emotion intensity scale (0.0 – 1.0)"
    )
    # Manual prosody overrides from UI sliders
    rate_override: Optional[str] = Field(
        None, description="Rate override, e.g. '+30%' or '-20%'"
    )
    pitch_override: Optional[str] = Field(
        None, description="Pitch override in semitones, e.g. '+3.0st' or '-2.0st'"
    )
    volume_override: Optional[str] = Field(
        None, description="Volume override in dB, e.g. '+4.0dB' or '-3.0dB'"
    )
