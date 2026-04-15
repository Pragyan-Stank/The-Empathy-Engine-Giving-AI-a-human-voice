from pydantic import BaseModel, constr, Field
from typing import Optional

class SynthesizeRequest(BaseModel):
    text: constr(min_length=1, max_length=500) = Field(
        ..., description="The input text to analyze and synthesize. Max 500 characters."
    )
    emotion_override: Optional[str] = Field(
        None, description="Manually override the detected emotion (e.g., 'joy', 'anger')"
    )
    intensity: float = Field(
        1.0, ge=0.0, le=1.0, description="Intensity of the emotion scaling (0.0 to 1.0)"
    )
