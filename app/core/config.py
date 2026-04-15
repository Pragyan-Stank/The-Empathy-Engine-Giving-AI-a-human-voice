from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    ENVIRONMENT: str = "development"
    APP_NAME: str = "Empathy Engine API"
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    
    # Emotion Model Config
    USE_TRANSFORMERS_MODEL: bool = True
    HUGGINGFACE_EMOTION_MODEL: str = "mrm8488/distilbert-base-uncased-emotion"
    
    # TTS Provider Defaults
    TTS_PROVIDER: str = "pyttsx3"  # options: google, pyttsx3, dummy
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

settings = Settings()
