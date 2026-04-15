from app.services.emotion.base import SentimentAnalyzer, EmotionAnalyzer
from typing import Tuple

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    SentimentIntensityAnalyzer = None

class VaderSentimentFallback(SentimentAnalyzer, EmotionAnalyzer):
    def __init__(self):
        self.analyzer = None
        if SentimentIntensityAnalyzer:
            self.analyzer = SentimentIntensityAnalyzer()
            
    def analyze_sentiment(self, text: str) -> Tuple[str, float]:
        if not self.analyzer:
            return "neutral", 0.0
            
        scores = self.analyzer.polarity_scores(text)
        compound = scores['compound']
        
        if compound >= 0.5:
            sentiment = "positive"
        elif compound <= -0.5:
            sentiment = "negative"
        else:
            sentiment = "neutral"
            
        return sentiment, compound

    def analyze(self, text: str) -> Tuple[str, float]:
        """Provides a rough emotion mapping from sentiment for fallback."""
        sentiment, compound = self.analyze_sentiment(text)
        confidence = abs(compound) if abs(compound) > 0.3 else 0.5
        
        if sentiment == "positive":
            return "joy", confidence
        elif sentiment == "negative":
            return "sadness", confidence # or anger, but sadness is safer fallback
        else:
            return "neutral", confidence
