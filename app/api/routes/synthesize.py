"""
POST /api/v1/synthesize — full emotion-aware synthesis pipeline.

TTS provider chain (in priority order):
  1. ElevenLabs    — expressive neural, emotion-mapped voice settings (PRIMARY)
  2. Google Cloud  — SSML+audioConfig prosody (if API enabled)
  3. Edge TTS      — free Microsoft neural voices, rate/pitch/volume
  4. pyttsx3       — offline SAPI5 fallback (rate + volume only)

Emotion detection chain:
  1. Transformer model (j-hartmann/emotion-english-distilroberta-base)
  2. VADER sentiment fallback (if transformer confidence < 0.15)

Emotion refinement:
  - Rule-based granular sub-emotion detection (excitement, contentment,
    grief, frustration, rage, anxiety) runs after classification.
"""
import os
import re
from fastapi import APIRouter, HTTPException

from app.api.schemas.request import SynthesizeRequest
from app.api.schemas.response import SynthesizeResponse, ProsodyResponse, SentenceEmotionItem
from app.core.config import settings
from app.core.logging_config import logger
from app.core.exceptions import TTSGenerationError

from app.services.emotion.transformer_model import TransformerEmotionAnalyzer
from app.services.emotion.sentiment_fallback import VaderSentimentFallback
from app.services.emotion.granular import refine_emotion
from app.services.emotion.intensity import calculate_prosody
from app.services.emotion.sentence_analysis import (
    analyze_text, build_emotion_breakdown, split_sentences
)
from app.services.text.text_enhancer import enhance_text
from app.services.tts.ssml_builder import SSMLBuilder
from app.services.tts.elevenlabs_tts import ElevenLabsTTS
from app.services.tts.google_tts import GoogleCloudTTS
from app.services.tts.expressive_edge_tts import ExpressiveEdgeTTS
from app.services.tts.edge_tts_engine import EdgeTTSEngine
from app.services.tts.fallback_tts import FallbackTTS
from app.services.audio.storage import AudioStorageService
from app.services.audio.post_processor import process_audio

router = APIRouter()

# ── Service singletons ─────────────────────────────────────────────────────────
_transformer   = TransformerEmotionAnalyzer(settings.HUGGINGFACE_EMOTION_MODEL)
_vader         = VaderSentimentFallback()
_ssml_builder  = SSMLBuilder()
_storage       = AudioStorageService()
_elevenlabs      = ElevenLabsTTS()
_google_tts      = GoogleCloudTTS()
_expressive_edge = ExpressiveEdgeTTS()   # Primary free: emotion voice styles
_edge_tts        = EdgeTTSEngine()       # Fallback free: rate/pitch/volume only
_pyttsx3_tts     = FallbackTTS()         # Offline: SAPI5 (rate + volume)


# ── Emotion detection with fallback chain ──────────────────────────────────────
def _detect_emotion(text: str):
    if settings.USE_TRANSFORMERS_MODEL:
        try:
            label, confidence = _transformer.analyze(text)
            if confidence >= 0.15:
                logger.info(f"Transformer → emotion={label}, conf={confidence:.2f}")
                return label, confidence
            logger.warning(f"Transformer low confidence ({confidence:.2f}). Using VADER.")
        except Exception as e:
            logger.error(f"Transformer error: {e}. Using VADER.")

    label, confidence = _vader.analyze(text)
    logger.info(f"VADER → emotion={label}, conf={confidence:.2f}")
    return label, confidence


# ── TTS provider chain ─────────────────────────────────────────────────────────
async def _synthesize_audio(
    text: str,
    engine_ssml: str,
    filepath: str,
    prosody: dict,
    emotion: str,
) -> str:
    """Try providers in priority order; return actual saved filepath."""

    # 1 — ElevenLabs (emotion-mapped voice character)
    if _elevenlabs.available:
        try:
            return await _elevenlabs.synthesize(
                text, engine_ssml, filepath, prosody, emotion=emotion
            )
        except TTSGenerationError as e:
            logger.error(f"ElevenLabs failed ({e}). Trying Google TTS.")

    # 2 — Google Cloud TTS (SSML + audioConfig prosody)
    if _google_tts.available:
        try:
            return await _google_tts.synthesize(
                text, engine_ssml, filepath, prosody, emotion=emotion
            )
        except TTSGenerationError as e:
            logger.error(f"Google TTS failed ({e}). Trying Expressive Edge TTS.")

    # 3 — Expressive Edge TTS (emotion voice styles + prosody via WebSocket SSML)
    if _expressive_edge.available:
        edge_path = os.path.splitext(filepath)[0] + ".mp3"
        try:
            return await _expressive_edge.synthesize(
                text, engine_ssml, edge_path, prosody, emotion=emotion
            )
        except TTSGenerationError as e:
            logger.error(f"Expressive Edge TTS failed ({e}). Trying basic Edge TTS.")

    # 4 — Basic Edge TTS (prosody only, no style)
    if _edge_tts.available:
        edge_path = os.path.splitext(filepath)[0] + ".mp3"
        try:
            return await _edge_tts.synthesize(
                text, engine_ssml, edge_path, prosody, emotion=emotion
            )
        except TTSGenerationError as e:
            logger.error(f"Basic Edge TTS failed ({e}). Falling back to pyttsx3.")

    # 5 — pyttsx3 offline (rate + volume only)
    wav_path = os.path.splitext(filepath)[0] + ".wav"
    return await _pyttsx3_tts.synthesize(
        text, engine_ssml, wav_path, prosody, emotion=emotion
    )


# ── Route ──────────────────────────────────────────────────────────────────────
@router.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize_speech(request: SynthesizeRequest):
    logger.info(f"Synthesize — text length: {len(request.text)}")

    clean_text = request.text.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        # Step 1 — Sentiment polarity (VADER, always)
        sentiment_label, _ = _vader.analyze_sentiment(clean_text)

        # Step 2 — Emotion detection
        if request.emotion_override:
            base_emotion = request.emotion_override.lower()
            confidence = 1.0
            logger.info(f"Emotion override: {base_emotion}")
        else:
            base_emotion, confidence = _detect_emotion(clean_text)

        # Step 3 — Granular refinement (rule-based sub-emotion)
        emotion_label = refine_emotion(clean_text, base_emotion)
        if emotion_label != base_emotion:
            logger.info(f"Refined: {base_emotion} \u2192 {emotion_label}")

        # Step 3b — Cross-validation: when transformer confidence is low,
        # compare its polarity against VADER. If they disagree, prefer VADER.
        # Prevents false positives like 'yes, why not!' \u2192 anger at 50%.
        if not request.emotion_override and confidence < 0.65:
            _NEG = {"anger","frustration","rage","disgust","sadness","grief","fear","anxiety"}
            _POS = {"joy","excitement","contentment","surprise"}
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as _VI
            _cmp = _VI().polarity_scores(clean_text)["compound"]
            transformer_pol = "neg" if emotion_label in _NEG else ("pos" if emotion_label in _POS else "neu")
            vader_pol       = "pos" if _cmp > 0.15 else ("neg" if _cmp < -0.15 else "neu")
            if transformer_pol != vader_pol and vader_pol != "neu":
                vader_emotion, vader_conf = _vader.analyze(clean_text)
                logger.info(
                    f"Cross-validation: transformer={emotion_label}({confidence:.0%}) conflicts "
                    f"with VADER polarity ({vader_pol}); overriding \u2192 {vader_emotion}"
                )
                emotion_label = vader_emotion
                confidence    = vader_conf

        # Step 4 — Prosody from emotion \u00d7 intensity
        prosody = calculate_prosody(emotion_label, request.intensity)
        logger.info(f"Auto prosody: {prosody}")

        # Step 5 — Manual slider overrides
        if request.rate_override:
            prosody["rate"] = request.rate_override
        if request.pitch_override:
            prosody["pitch"] = request.pitch_override
        if request.volume_override:
            prosody["volume"] = request.volume_override
        if any([request.rate_override, request.pitch_override, request.volume_override]):
            logger.info(f"After overrides: {prosody}")

        # Step 6 — Per-sentence analysis (for breakdown + multi-emotion TTS)
        sentence_results = analyze_text(clean_text)
        emotion_breakdown = build_emotion_breakdown(sentence_results)
        is_multi = len(sentence_results) > 1
        if is_multi:
            unique_emotions = list(emotion_breakdown.keys())
            logger.info(
                f"Multi-emotion text — {len(sentence_results)} sentences, "
                f"emotions: {unique_emotions}"
            )

        # Step 7 — Text Enhancement (punctuation injection + chunking)
        enhanced_text = enhance_text(clean_text, emotion_label, request.intensity)
        # TTS-safe text: convert ||Xms|| pause markers to a natural pause.
        # If a marker follows punctuation (.!?) → just a space (punctuation gives the pause).
        # If a marker follows a word → insert a comma (light breath pause).
        tts_text = re.sub(r"(?<=[.!?,])\s*\|\|\d+ms\|\|\s*", " ", enhanced_text)
        tts_text = re.sub(r"\|\|\d+ms\|\|", ", ", tts_text).strip()
        logger.info(
            f"Text enhanced: {len(clean_text)}→{len(tts_text)} chars"
        )

        # Step 8 — SSML generation (now emotion-aware)
        engine_ssml  = _ssml_builder.build_ssml_engine(enhanced_text, prosody, emotion_label)
        display_ssml = _ssml_builder.build_ssml_display(enhanced_text, prosody, emotion_label)

        # Step 9 — Cache key (prosody + emotion breakdown)
        filename = _storage.generate_filename(
            clean_text, emotion_label, request.intensity,
            prosody=prosody, extension="mp3",
        )
        base_filepath = _storage.get_filepath(filename)

        if settings.ENABLE_AUDIO_CACHE and _storage.file_exists(filename):
            final_path = base_filepath
            logger.info(f"Cache hit: {filename}")
        else:
            # Step 10 — Synthesize (use enhanced tts_text for expressiveness)
            final_path = await _synthesize_audio(
                tts_text, engine_ssml, base_filepath, prosody, emotion_label
            )
            logger.info(f"Audio saved: {final_path}")

            # Step 11 — Post-processing (EQ, reverb, compression)
            final_path = process_audio(final_path, emotion_label, request.intensity)
            logger.info(f"Post-processing complete: {final_path}")

        final_filename = os.path.basename(final_path)
        audio_url = f"/api/v1/audio/{final_filename}"

        sentence_items = [
            SentenceEmotionItem(text=r.text, emotion=r.emotion, style=r.style)
            for r in sentence_results
        ]

        return SynthesizeResponse(
            success=True,
            detected_emotion=emotion_label,
            sentiment=sentiment_label,
            confidence=round(confidence, 2),
            intensity=request.intensity,
            prosody=ProsodyResponse(**prosody),
            audio_url=audio_url,
            ssml_preview=display_ssml,
            sentence_analysis=sentence_items,
            emotion_breakdown=emotion_breakdown,
            is_multi_emotion=is_multi,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {e}")
