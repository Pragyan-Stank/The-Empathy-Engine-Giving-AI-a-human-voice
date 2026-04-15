from fastapi import APIRouter, Depends, HTTPException
from typing import Optional

from app.api.schemas.request import SynthesizeRequest
from app.api.schemas.response import SynthesizeResponse, ProsodyResponse
from app.core.config import settings
from app.core.logging_config import logger

from app.services.emotion.transformer_model import TransformerEmotionAnalyzer
from app.services.emotion.sentiment_fallback import VaderSentimentFallback
from app.services.emotion.intensity import calculate_prosody
from app.services.tts.ssml_builder import SSMLBuilder
from app.services.tts.google_tts import GoogleCloudTTS
from app.services.tts.fallback_tts import FallbackTTS
from app.services.audio.storage import AudioStorageService
from app.core.exceptions import TTSGenerationError

router = APIRouter()

# Globals for service initialization to reuse memory
if settings.USE_TRANSFORMERS_MODEL:
    emotion_analyzer = TransformerEmotionAnalyzer(settings.HUGGINGFACE_EMOTION_MODEL)
else:
    emotion_analyzer = VaderSentimentFallback()

sentiment_analyzer = VaderSentimentFallback()

ssml_builder = SSMLBuilder()
audio_storage = AudioStorageService()

# Setup TTS provider chain
primary_tts = GoogleCloudTTS()
fallback_tts = FallbackTTS()

@router.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize_speech(request: SynthesizeRequest):
    logger.info(f"Processing synthesize request for text length {len(request.text)}")
    
    # Text sanitization
    clean_text = request.text.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
        
    try:
        # Step 1: Sentiment & Emotion analysis
        sentiment_label, sentiment_score = sentiment_analyzer.analyze_sentiment(clean_text)
        
        if request.emotion_override:
            emotion_label = request.emotion_override.lower()
            confidence = 1.0
            logger.info(f"Using emotion override: {emotion_label}")
        else:
            emotion_label, confidence = emotion_analyzer.analyze(clean_text)
            logger.info(f"Predicted emotion: {emotion_label} (score: {confidence})")
            
        # Step 2: Prosody calculation
        prosody = calculate_prosody(emotion_label, request.intensity)
        
        # Step 3: Build SSML
        ssml_content = ssml_builder.build_ssml(clean_text, prosody)
        logger.info(f"Generated SSML payload length: {len(ssml_content)}")
        
        # Step 4: Check Cache
        filename = audio_storage.generate_filename(clean_text, emotion_label, request.intensity)
        filepath = audio_storage.get_filepath(filename)
        
        if settings.ENABLE_AUDIO_CACHE and audio_storage.file_exists(filename):
            logger.info(f"Cache hit. Returning existing file: {filename}")
        else:
            # Step 5: Synthesize (Primary -> Fallback)
            try:
                if settings.TTS_PROVIDER == "google" and primary_tts.client:
                    await primary_tts.synthesize(clean_text, ssml_content, filepath)
                else:
                    await fallback_tts.synthesize(clean_text, ssml_content, filepath)
            except TTSGenerationError as e:
                logger.error(f"Primary TTS failed: {e}. Attempting fallback...")
                await fallback_tts.synthesize(clean_text, ssml_content, filepath)
                
        # Format Host URL
        audio_url = f"/api/v1/audio/{filename}"
        
        return SynthesizeResponse(
            success=True,
            detected_emotion=emotion_label,
            sentiment=sentiment_label,
            confidence=round(confidence, 2),
            intensity=request.intensity,
            prosody=ProsodyResponse(**prosody),
            audio_url=audio_url,
            ssml_preview=ssml_content
        )
        
    except Exception as e:
        logger.error(f"Error in synthesizing pipeline: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during synthesis.")
