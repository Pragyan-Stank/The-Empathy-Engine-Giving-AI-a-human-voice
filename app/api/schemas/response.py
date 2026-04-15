from pydantic import BaseModel
from typing import Optional, List, Dict


class ProsodyResponse(BaseModel):
    rate: str
    pitch: str
    volume: str


class SentenceEmotionItem(BaseModel):
    text: str
    emotion: str
    style: str     # Azure voice style that was applied


class SynthesizeResponse(BaseModel):
    success: bool
    detected_emotion: str              # dominant / overall emotion
    sentiment: str                     # positive / negative / neutral
    confidence: float
    intensity: float
    prosody: ProsodyResponse
    audio_url: str
    ssml_preview: Optional[str] = None
    error: Optional[str] = None

    # Per-sentence breakdown (multi-emotion support)
    sentence_analysis: List[SentenceEmotionItem] = []
    emotion_breakdown: Dict[str, float] = {}   # {"grief": 50.0, "joy": 50.0}
    is_multi_emotion: bool = False
