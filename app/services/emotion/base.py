from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any

class EmotionAnalyzer(ABC):
    @abstractmethod
    def analyze(self, text: str) -> Tuple[str, float]:
        """
        Analyze text and return (emotion_label, confidence_score)
        """
        pass

class SentimentAnalyzer(ABC):
    @abstractmethod
    def analyze_sentiment(self, text: str) -> Tuple[str, float]:
        """
        Analyze text and return (sentiment_label, polarity_score)
        """
        pass
