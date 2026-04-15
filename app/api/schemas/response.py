from pydantic import BaseModel
from typing import Optional

class ProsodyResponse(BaseModel):
    rate: str
    pitch: str
    volume: str

class SynthesizeResponse(BaseModel):
    success: bool
    detected_emotion: str
    sentiment: str
    confidence: float
    intensity: float
    prosody: ProsodyResponse
    audio_url: str
    ssml_preview: Optional[str] = None
    error: Optional[str] = None
