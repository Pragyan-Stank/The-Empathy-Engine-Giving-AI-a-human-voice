"""
Sentence-level emotion analysis — shared utility.

Used by:
  - ExpressiveEdgeTTS  (per-sentence prosody in synthesis)
  - synthesize route   (emotion breakdown in API response)
"""
import re
from dataclasses import dataclass, field
from typing import List, Dict

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_vader = SentimentIntensityAnalyzer()

# Patterns for refinement
_EXCITEMENT_PAT = re.compile(r"[!]{2,}|[A-Z]{4,}|\bOMG\b|\bWOW\b|\bAMAZING\b|\bINCREDIBLE\b")
_GRIEF_PAT      = re.compile(r"\b(lost|miss|gone|never|goodbye|mourn|no more)\b", re.I)
_ANXIETY_PAT    = re.compile(r"\b(worried|anxious|nervous|stressed|panic|overwhelm)\b", re.I)
_RAGE_PAT       = re.compile(r"[A-Z]{5,}|\bFURIOUS\b|\bHATE\b|\bWORST\b")
_FRUSTRATION    = re.compile(r"\b(frustrated|annoyed|ugh|seriously|anymore|can.t)\b", re.I)


@dataclass
class SentenceResult:
    text: str
    emotion: str
    compound: float         # VADER compound -1..+1
    style: str             # Azure voice style applied


# Edge TTS style per emotion (duplicated here to avoid import cycles)
_STYLE_MAP = {
    "joy": "cheerful", "excitement": "excited", "contentment": "friendly",
    "sadness": "sad", "grief": "sad", "anger": "angry",
    "frustration": "unfriendly", "rage": "shouting", "disgust": "unfriendly",
    "fear": "terrified", "anxiety": "whispering", "surprise": "excited",
    "neutral": "chat",
}


def detect_sentence_emotion(sentence: str) -> tuple:
    """
    Returns (emotion_label, compound_score) for a single sentence.
    Uses VADER + rule-based refinement.
    """
    scores   = _vader.polarity_scores(sentence)
    compound = scores["compound"]

    if compound >= 0.5:
        emotion = "excitement" if _EXCITEMENT_PAT.search(sentence) else "joy"
    elif compound >= 0.2:
        emotion = "contentment"
    elif compound >= -0.2:
        emotion = "neutral"
    elif compound >= -0.5:
        if _GRIEF_PAT.search(sentence):
            return "grief", compound
        if _FRUSTRATION.search(sentence):
            return "frustration", compound
        emotion = "sadness"
    else:
        if _RAGE_PAT.search(sentence):
            return "rage", compound
        if _GRIEF_PAT.search(sentence):
            return "grief", compound
        if _ANXIETY_PAT.search(sentence):
            return "anxiety", compound
        emotion = "sadness"

    return emotion, compound


def split_sentences(text: str) -> List[str]:
    """Split text at sentence boundaries (.!?) preserving punctuation."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def analyze_text(text: str) -> List[SentenceResult]:
    """
    Analyze all sentences in text.
    Returns a SentenceResult per sentence.
    """
    sentences = split_sentences(text)
    if not sentences:
        sentences = [text.strip()]

    results = []
    for s in sentences:
        emotion, compound = detect_sentence_emotion(s)
        results.append(SentenceResult(
            text=s,
            emotion=emotion,
            compound=compound,
            style=_STYLE_MAP.get(emotion, "chat"),
        ))
    return results


def build_emotion_breakdown(sentence_results: List[SentenceResult]) -> Dict[str, float]:
    """
    Compute per-emotion percentage across all sentences.
    Returns dict like {"grief": 50.0, "joy": 50.0}.
    Sorted by percentage descending.
    """
    if not sentence_results:
        return {}
    totals: Dict[str, int] = {}
    for r in sentence_results:
        totals[r.emotion] = totals.get(r.emotion, 0) + 1
    n = len(sentence_results)
    breakdown = {e: round(c / n * 100, 1) for e, c in totals.items()}
    return dict(sorted(breakdown.items(), key=lambda x: -x[1]))
