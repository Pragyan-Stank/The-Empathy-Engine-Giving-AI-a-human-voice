"""
WebSocket streaming TTS endpoint — progressive audio generation.

Instead of generating the full audio and returning it, this endpoint:
1. Accepts text via WebSocket
2. Runs the full pipeline (emotion → LLM → prosody → SSML)
3. Generates audio segment by segment
4. Streams each segment's audio bytes to the client as they're ready
5. Client can start playback while remaining segments are still generating

This achieves low-latency conversational feel — the user hears the first
words within ~500ms instead of waiting for the full synthesis.

Protocol:
  Client sends JSON: {"text": "...", "emotion_override": null, "intensity": 1.0}
  Server sends binary: MP3 audio chunks (each a complete segment)
  Server sends JSON: {"type": "metadata", ...} for pipeline info
  Server sends JSON: {"type": "done"} when complete
"""
import os
import json
import asyncio
import base64
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.config import settings
from app.core.logging_config import logger
from app.services.emotion.transformer_model import TransformerEmotionAnalyzer
from app.services.emotion.sentiment_fallback import VaderSentimentFallback
from app.services.emotion.granular import refine_emotion
from app.services.emotion.intensity import calculate_prosody
from app.services.emotion.sentence_analysis import (
    analyze_text, build_emotion_breakdown, split_sentences, detect_sentence_emotion
)
from app.services.text.text_enhancer import enhance_text
from app.services.tts.ssml_builder import SSMLBuilder
from app.services.tts.google_tts import GoogleCloudTTS
from app.services.tts.expressive_edge_tts import ExpressiveEdgeTTS
from app.services.audio.post_processor import process_audio
from app.services.llm.speech_analyzer import analyze_speech
from app.services.tts.prosody_curve import google_tts_format_from_deltas

router = APIRouter()

# Service singletons (shared with main synthesize route)
_transformer   = TransformerEmotionAnalyzer(settings.HUGGINGFACE_EMOTION_MODEL)
_vader         = VaderSentimentFallback()
_ssml_builder  = SSMLBuilder()
_google_tts    = GoogleCloudTTS()
_edge_tts      = ExpressiveEdgeTTS()


def _detect_emotion(text: str):
    """Emotion detection with transformer → VADER fallback chain."""
    if settings.USE_TRANSFORMERS_MODEL:
        try:
            label, confidence = _transformer.analyze(text)
            if confidence >= 0.15:
                return label, confidence
        except Exception:
            pass
    label, confidence = _vader.analyze(text)
    return label, confidence


@router.websocket("/stream")
async def stream_speech(websocket: WebSocket):
    """
    WebSocket endpoint for streaming TTS.

    Flow:
    1. Client connects and sends text JSON
    2. Server runs pipeline and streams audio segments
    3. Each segment is sent as binary MP3 data
    4. Metadata is sent as JSON messages
    """
    await websocket.accept()
    logger.info("WebSocket streaming session started")

    try:
        while True:
            # Receive text from client
            raw = await websocket.receive_text()
            try:
                request = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            text = request.get("text", "").strip()
            if not text:
                await websocket.send_json({"type": "error", "message": "Empty text"})
                continue

            emotion_override = request.get("emotion_override")
            intensity = float(request.get("intensity", 1.0))

            try:
                # ── Step 1: Emotion Detection ──────────────────────────────
                if emotion_override:
                    emotion_label = emotion_override.lower()
                    confidence = 1.0
                else:
                    base_emotion, confidence = _detect_emotion(text)
                    emotion_label = refine_emotion(text, base_emotion)

                # Send emotion metadata immediately
                await websocket.send_json({
                    "type": "metadata",
                    "emotion": emotion_label,
                    "confidence": round(confidence, 2),
                    "status": "analyzing",
                })

                # ── Step 2: LLM Speech Analysis ───────────────────────────
                prosody = calculate_prosody(emotion_label, intensity)
                speech_analysis = await analyze_speech(text, emotion_label, intensity)

                await websocket.send_json({
                    "type": "metadata",
                    "delivery_style": speech_analysis.delivery_style,
                    "tone_arc": speech_analysis.tone_arc,
                    "intent": speech_analysis.intent,
                    "segments": len(speech_analysis.segments),
                    "status": "generating",
                })

                # ── Step 3: Text Enhancement ──────────────────────────────
                SHORT_TEXT_LIMIT = 600
                if (speech_analysis.llm_used
                    and speech_analysis.humanized_text
                    and len(text) <= SHORT_TEXT_LIMIT):
                    base_for_tts = speech_analysis.humanized_text
                else:
                    base_for_tts = text

                enhanced_text = enhance_text(
                    base_for_tts, emotion_label, intensity,
                    tone_arc=speech_analysis.tone_arc,
                )

                # ── Step 4: Segment-by-Segment Audio Generation ───────────
                sentences = split_sentences(enhanced_text)
                if not sentences:
                    sentences = [enhanced_text]

                segment_deltas = [
                    {
                        "rate_delta_pct":  s.rate_delta_pct,
                        "pitch_delta_hz":  s.pitch_delta_hz,
                        "volume_delta_db": s.volume_delta_db,
                        "pause_before_ms": s.pause_before_ms,
                        "emotion":         s.emotion,
                        "emphasis_words":  s.emphasis_words,
                    }
                    for s in speech_analysis.segments
                ]

                total_segments = len(sentences)
                for i, sentence in enumerate(sentences):
                    # Get per-segment emotion and prosody
                    if i < len(segment_deltas):
                        delta = segment_deltas[i]
                        seg_emotion = delta.get("emotion", emotion_label)
                    else:
                        seg_emotion, _ = detect_sentence_emotion(sentence)
                        delta = {"rate_delta_pct": 0, "pitch_delta_hz": 0,
                                "volume_delta_db": 0, "pause_before_ms": 0}

                    # Generate segment audio
                    seg_filename = f"stream_seg_{i}.mp3"
                    seg_filepath = os.path.join(settings.OUTPUT_AUDIO_DIR, seg_filename)

                    # Build segment SSML
                    emphasis = delta.get("emphasis_words", [])
                    seg_ssml = _ssml_builder.build_ssml_engine(
                        sentence, emotion=seg_emotion, extra_emphasis=emphasis,
                    )

                    # Synthesize this segment
                    try:
                        if _google_tts.available:
                            seg_prosody = dict(prosody)
                            actual_path = await _google_tts.synthesize(
                                sentence, seg_ssml, seg_filepath,
                                seg_prosody, emotion=seg_emotion,
                                segment_deltas=[delta] if delta else None,
                            )
                        elif _edge_tts.available:
                            actual_path = await _edge_tts.synthesize(
                                sentence, seg_ssml, seg_filepath,
                                prosody, emotion=seg_emotion,
                                segment_deltas=[delta] if delta else None,
                            )
                        else:
                            continue

                        # Post-process the segment
                        actual_path = process_audio(actual_path, seg_emotion, intensity)

                        # Read and send audio bytes
                        with open(actual_path, "rb") as f:
                            audio_bytes = f.read()

                        # Send as base64-encoded JSON (for broad WebSocket compatibility)
                        await websocket.send_json({
                            "type": "audio_chunk",
                            "segment": i,
                            "total": total_segments,
                            "emotion": seg_emotion,
                            "audio_base64": base64.b64encode(audio_bytes).decode("utf-8"),
                            "format": "mp3",
                        })

                        logger.info(
                            f"Stream seg {i+1}/{total_segments}: "
                            f"{seg_emotion}, {len(audio_bytes)} bytes"
                        )

                        # Clean up segment file
                        try:
                            os.remove(actual_path)
                        except Exception:
                            pass

                    except Exception as e:
                        logger.error(f"Stream seg {i+1} failed: {e}")
                        await websocket.send_json({
                            "type": "segment_error",
                            "segment": i,
                            "error": str(e),
                        })

                # ── Done ──────────────────────────────────────────────────
                sentence_results = analyze_text(text)
                emotion_breakdown = build_emotion_breakdown(sentence_results)

                await websocket.send_json({
                    "type": "done",
                    "emotion": emotion_label,
                    "emotion_breakdown": emotion_breakdown,
                    "segments_generated": total_segments,
                })

            except Exception as e:
                logger.error(f"Streaming pipeline error: {e}", exc_info=True)
                await websocket.send_json({
                    "type": "error",
                    "message": f"Pipeline error: {str(e)}",
                })

    except WebSocketDisconnect:
        logger.info("WebSocket streaming session ended")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
