from app.services.emotion.base import EmotionAnalyzer
from app.core.logging_config import logger
from typing import Tuple

try:
    from transformers import pipeline
except ImportError:
    logger.warning("Transformers library not installed. Transformer emotion model won't work.")
    pipeline = None

class TransformerEmotionAnalyzer(EmotionAnalyzer):
    def __init__(self, model_name: str = "mrm8488/distilbert-base-uncased-emotion"):
        self.model_name = model_name
        self._classifier = None
        
    def _load_model(self):
        if pipeline is None:
            raise RuntimeError("Transformers library is missing.")
        if self._classifier is None:
            logger.info(f"Loading transformer model {self.model_name}")
            try:
                from app.core.config import settings
                token = settings.HF_TOKEN
                if token:
                    self._classifier = pipeline("text-classification", model=self.model_name, top_k=1, token=token)
                else:
                    self._classifier = pipeline("text-classification", model=self.model_name, top_k=1)
            except Exception as e:
                logger.error(f"Failed to load user-specified emotion model: {e}")
                
    def analyze(self, text: str) -> Tuple[str, float]:
        try:
            self._load_model()
            if self._classifier is None:
                return "neutral", 0.0
            
            result = self._classifier(text)
            # pipeline top_k=1 returns [[{'label': 'joy', 'score': 0.99}]]
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], list):
                    prediction = result[0][0]
                else:
                    prediction = result[0]
                label = prediction['label']
                score = prediction['score']
                return label.lower(), float(score)
        except Exception as e:
            logger.error(f"Transformer model error: {e}", exc_info=True)
            
        return "neutral", 0.0
