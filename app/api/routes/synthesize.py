"""
POST /api/v1/synthesize — full emotion-aware synthesis pipeline.

TTS provider chain (in order):
  1. Google Cloud TTS (REST, API-key based) — if GOOGLE_API_KEY is set and API is enabled
  2. Edge TTS (Microsoft Edge neural voices, free, internet required)
  3. pyttsx3 (offline SAPI5 fallback — rate + volume only, no pitch)
"""
import os
from fastapi import APIRouter, HTTPException

from app.api.schemas.request import SynthesizeRequest
from app.api.schemas.response import SynthesizeResponse, ProsodyResponse
from app.core.config import settings
from app.core.logging_config import logger
from app.core.exceptions import TTSGenerationError

from app.services.emotion.transformer_model import TransformerEmotionAnalyzer
from app.services.emotion.sentiment_fallback import VaderSentimentFallback
from app.services.emotion.intensity import calculate_prosody
from app.services.tts.ssml_builder import SSMLBuilder
from app.services.tts.google_tts import GoogleCloudTTS
from app.services.tts.edge_tts_engine import EdgeTTSEngine
from app.services.tts.fallback_tts import FallbackTTS
from app.services.audio.storage import AudioStorageService

router = APIRouter()

# ── Service singletons ─────────────────────────────────────────────────────
_transformer  = TransformerEmotionAnalyzer(settings.HUGGINGFACE_EMOTION_MODEL)
_vader        = VaderSentimentFallback()
_ssml_builder = SSMLBuilder()
_storage      = AudioStorageService()
_google_tts   = GoogleCloudTTS()
_edge_tts     = EdgeTTSEngine()
_pyttsx3_tts  = FallbackTTS()


# ── Emotion detection with automatic fallback ──────────────────────────────
def _detect_emotion(text: str):
    """Transformer → VADER fallback chain."""
    if settings.USE_TRANSFORMERS_MODEL:
        try:
            label, confidence = _transformer.analyze(text)
            if confidence >= 0.15:
                logger.info(f"Transformer → emotion={label}, conf={confidence:.2f}")
                return label, confidence
            logger.warning(
                f"Transformer low confidence ({confidence:.2f}), falling back to VADER."
            )
        except Exception as e:
            logger.error(f"Transformer error: {e} — using VADER.")

    label, confidence = _vader.analyze(text)
    logger.info(f"VADER → emotion={label}, conf={confidence:.2f}")
    return label, confidence


# ── TTS provider chain ─────────────────────────────────────────────────────
async def _synthesize_audio(
    text: str, engine_ssml: str, filepath: str, prosody: dict
) -> str:
    """
    Try providers in order.
    Returns the actual saved filepath (extension may differ from input).
    """
    # 1. Google Cloud TTS
    if _google_tts.available:
        try:
            return await _google_tts.synthesize(text, engine_ssml, filepath, prosody)
        except TTSGenerationError as e:
            logger.error(f"Google TTS failed ({e}). Trying Edge TTS.")

    # 2. Edge TTS (free neural voices, supports rate/pitch/volume)
    if _edge_tts.available:
        edge_path = os.path.splitext(filepath)[0] + ".mp3"
        try:
            return await _edge_tts.synthesize(text, engine_ssml, edge_path, prosody)
        except TTSGenerationError as e:
            logger.error(f"Edge TTS failed ({e}). Falling back to pyttsx3.")

    # 3. pyttsx3 offline (rate + volume; no pitch)
    wav_path = os.path.splitext(filepath)[0] + ".wav"
    return await _pyttsx3_tts.synthesize(text, engine_ssml, wav_path, prosody)


# ── Route ──────────────────────────────────────────────────────────────────
@router.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize_speech(request: SynthesizeRequest):
    logger.info(f"Synthesize — text length: {len(request.text)}")

    clean_text = request.text.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        # Step 1 — Sentiment label (VADER, always)
        sentiment_label, _ = _vader.analyze_sentiment(clean_text)

        # Step 2 — Emotion detection
        if request.emotion_override:
            emotion_label = request.emotion_override.lower()
            confidence = 1.0
            logger.info(f"Emotion override: {emotion_label}")
        else:
            emotion_label, confidence = _detect_emotion(clean_text)

        # Step 3 — Prosody calculation from emotion × intensity
        prosody = calculate_prosody(emotion_label, request.intensity)
        logger.info(f"Auto prosody: {prosody}")

        # Step 4 — Apply manual slider overrides (only when explicitly sent)
        if request.rate_override:
            prosody["rate"] = request.rate_override
        if request.pitch_override:
            prosody["pitch"] = request.pitch_override
        if request.volume_override:
            prosody["volume"] = request.volume_override

        if any([request.rate_override, request.pitch_override, request.volume_override]):
            logger.info(f"After manual overrides: {prosody}")

        # Step 5 — Build SSML
        engine_ssml  = _ssml_builder.build_ssml_engine(clean_text)   # for TTS engine
        display_ssml = _ssml_builder.build_ssml_display(clean_text, prosody)  # for UI

        # Step 6 — Resolve cached filename (prosody included in key)
        filename = _storage.generate_filename(
            clean_text, emotion_label, request.intensity,
            prosody=prosody, extension="mp3"
        )
        base_filepath = _storage.get_filepath(filename)

        # Step 7 — Synthesize (or serve from cache)
        if settings.ENABLE_AUDIO_CACHE and _storage.file_exists(filename):
            final_path = base_filepath
            logger.info(f"Cache hit: {filename}")
        else:
            final_path = await _synthesize_audio(
                clean_text, engine_ssml, base_filepath, prosody
            )
            logger.info(f"Audio saved: {final_path}")

        final_filename = os.path.basename(final_path)
        audio_url = f"/api/v1/audio/{final_filename}"

        return SynthesizeResponse(
            success=True,
            detected_emotion=emotion_label,
            sentiment=sentiment_label,
            confidence=round(confidence, 2),
            intensity=request.intensity,
            prosody=ProsodyResponse(**prosody),
            audio_url=audio_url,
            ssml_preview=display_ssml,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {e}")
