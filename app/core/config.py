from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    ENVIRONMENT: str = "development"
    APP_NAME: str = "Empathy Engine API"
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    
    # Emotion Model Config
    USE_TRANSFORMERS_MODEL: bool = True
    # Stable, publicly available 7-class emotion model (anger/disgust/fear/joy/neutral/sadness/surprise)
    HUGGINGFACE_EMOTION_MODEL: str = "j-hartmann/emotion-english-distilroberta-base"

    # TTS Provider (google | edge | pyttsx3)
    # edge = Microsoft Edge neural voices, free, no API key needed
    TTS_PROVIDER: str = "edge"
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    HF_TOKEN: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    
    # Output Directory
    OUTPUT_AUDIO_DIR: str = "output_audio"

    # Audio caching (Optional improvement)
    ENABLE_AUDIO_CACHE: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        extra = 'ignore'   # silently ignore any unknown keys in .env

settings = Settings()
