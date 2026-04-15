"""
POST /api/v1/synthesize — full emotion-aware speech performance pipeline.

Architecture:
  Input
  → LLM (emotion + intent detection + delivery arc)
  → Text Humanization (Indian rhythm, punctuation, chunking)
  → Prosody Engine (per-segment dynamic prosody)
  → SSML Generator (composite multi-segment with per-segment <prosody>)
  → Google TTS / Edge TTS / ElevenLabs (with segment deltas)
  → Audio Post-processing (de-essing, 3-band EQ, compression, reverb)
  → Output

TTS provider chain (in priority order):
  1. ElevenLabs    — expressive neural, emotion-mapped voice settings (PRIMARY)
  2. Google Cloud  — SSML+audioConfig prosody + composite SSML for tone transitions
  3. Edge TTS      — free Microsoft neural voices, rate/pitch/volume + emotion styles
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
from app.services.llm.speech_analyzer import analyze_speech, SpeechAnalysis
from app.services.tts.prosody_curve import build_segment_prosodies, edge_tts_format

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
    segment_deltas: list = None,
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

    # 2 — Google Cloud TTS (SSML + audioConfig prosody + composite SSML)
    if _google_tts.available:
        try:
            return await _google_tts.synthesize(
                text, engine_ssml, filepath, prosody, emotion=emotion,
                segment_deltas=segment_deltas,
            )
        except TTSGenerationError as e:
            logger.error(f"Google TTS failed ({e}). Trying Expressive Edge TTS.")

    # 3 — Expressive Edge TTS (emotion voice styles + prosody curves from LLM)
    if _expressive_edge.available:
        edge_path = os.path.splitext(filepath)[0] + ".mp3"
        try:
            return await _expressive_edge.synthesize(
                text, engine_ssml, edge_path, prosody, emotion=emotion,
                segment_deltas=segment_deltas or [],
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

        # Step 4 — Prosody from emotion × intensity
        prosody = calculate_prosody(emotion_label, request.intensity)
        logger.info(f"Auto prosody: {prosody}")

        # Step 4.5 — LLM Speech Analysis (Groq)
        # Rewrites text into natural spoken form and produces per-segment prosody deltas.
        # Also provides: delivery_style, tone_arc, intent for downstream processing.
        # Falls back gracefully (<6s timeout) so synthesis is never blocked.
        speech_analysis: SpeechAnalysis = await analyze_speech(
            clean_text, emotion_label, request.intensity
        )
        logger.info(
            f"Speech analysis: llm={speech_analysis.llm_used}, "
            f"style={speech_analysis.delivery_style}, "
            f"arc={speech_analysis.tone_arc}, "
            f"intent={speech_analysis.intent}, "
            f"segments={len(speech_analysis.segments)}"
        )

        # Step 5 — Manual slider overrides (applied after LLM so user overrides win)
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

        # Step 7 — Text Enhancement (punctuation injection + chunking + Indian rhythm)
        # CRITICAL: Always use FULL clean_text as the TTS source.
        # LLM humanized_text is capped at 1200 chars by the token budget —
        # using it for long texts silently drops everything after ~1200 chars.
        # LLM value for long texts = segment_deltas + emphasis_words, NOT text rewriting.
        SHORT_TEXT_LIMIT = 800  # increased from 600 — LLM now has more token budget
        if (
            speech_analysis.llm_used
            and speech_analysis.humanized_text
            and len(clean_text) <= SHORT_TEXT_LIMIT
        ):
            base_for_tts = speech_analysis.humanized_text
            logger.info(f"Using LLM-humanized text ({len(base_for_tts)} chars)")
        else:
            base_for_tts = clean_text
            if speech_analysis.llm_used and len(clean_text) > SHORT_TEXT_LIMIT:
                logger.info(
                    f"Long text ({len(clean_text)} chars) — using full clean_text; "
                    "LLM prosody deltas + emphasis still applied"
                )

        enhanced_text = enhance_text(
            base_for_tts, emotion_label, request.intensity,
            tone_arc=speech_analysis.tone_arc,
        )
        tts_text = re.sub(r"(?<=[.!?,])\s*\|\|\d+ms\|\|\s*", " ", enhanced_text)
        tts_text = re.sub(r"\|\|\d+ms\|\|", ", ", tts_text).strip()
        logger.info(f"Text enhanced: {len(clean_text)}→{len(tts_text)} chars")

        # Collect LLM emphasis words and segment deltas for downstream use
        segment_deltas = [
            {
                "rate_delta_pct":  s.rate_delta_pct,
                "pitch_delta_hz":  s.pitch_delta_hz,
                "volume_delta_db": s.volume_delta_db,
                "pause_before_ms": s.pause_before_ms,
                "emotion":         s.emotion,
                "emphasis_words":  s.emphasis_words,
                "arc_position":    s.arc_position,
            }
            for s in speech_analysis.segments
        ]
        llm_emphasis = [w for s in speech_analysis.segments for w in s.emphasis_words]

        # Step 8 — SSML generation (emotion-aware + LLM emphasis words)
        engine_ssml  = _ssml_builder.build_ssml_engine(
            enhanced_text, prosody, emotion_label, extra_emphasis=llm_emphasis
        )
        display_ssml = _ssml_builder.build_ssml_display(
            enhanced_text, prosody, emotion_label, extra_emphasis=llm_emphasis
        )

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
            # Step 10 — Synthesize with LLM-driven per-segment prosody curves
            final_path = await _synthesize_audio(
                tts_text, engine_ssml, base_filepath, prosody, emotion_label,
                segment_deltas=segment_deltas,
            )
            logger.info(f"Audio saved: {final_path}")

            # Step 11 — Post-processing (de-essing, 3-band EQ, compression, reverb)
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
            delivery_style=speech_analysis.delivery_style,
            tone_arc=speech_analysis.tone_arc,
            intent=speech_analysis.intent,
            llm_used=speech_analysis.llm_used,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {e}")
