"""
Google Cloud Text-to-Speech — upgraded for maximum expressiveness.

Architecture position:
  Input → LLM → Text Humanization → Prosody Engine → SSML → [THIS MODULE] → Audio

What's upgraded:
  1. Per-segment prosody via LLM segment deltas:
       - Each segment gets its OWN rate/pitch/volume from the delivery arc
       - No more flat prosody across the entire response
       - Uses google_tts_format_from_deltas() for native GCloud values

  2. Composite SSML synthesis mode:
       - For multi-segment text, builds ONE <speak> with per-segment <prosody>
       - Google TTS processes tone transitions within a single call
       - Falls back to sequential segment synthesis if composite fails

  3. Emotion-based voice selection (Neural2 / Studio):
       - Sad/grief/anxiety  → en-US-Neural2-F (warm, female, empathetic)
       - Angry/rage         → en-US-Neural2-D (male, assertive)
       - Joy/excitement     → en-US-Neural2-C (bright, expressive female)
       - Neutral/default    → en-US-Neural2-J (conversational, warm)
       INDIAN ENGLISH OPTIMIZATION:
       - en-IN-Neural2-A (female, Indian cadence)
       - en-IN-Neural2-B (male, Indian cadence)
       Selected based on input text Hinglish detection.

  4. Finer audioConfig tuning per emotion:
       - Effective rate:  0.25–4.0x
       - Pitch: -20.0 to +20.0 semitones
       - VolumeGainDb: -12.0 to +12.0

  5. Device profile selection for warmth:
       - "headphone-class-device" for intimate/empathetic
       - "small-bluetooth-speaker-class-device" for conversational
       - "handset-class-device" for phone-like warmth

  6. Multi-segment support with per-segment emotion + prosody:
       - Synthesizes each segment with its own emotion AND prosody profile
       - Concatenates raw MP3 bytes into a single file
"""
import re
import httpx
import base64
import asyncio
from typing import Dict, List, Optional, Tuple

from app.services.tts.base import TTSEngine
from app.core.exceptions import TTSGenerationError
from app.core.logging_config import logger
from app.core.config import settings
from app.services.emotion.sentence_analysis import split_sentences, detect_sentence_emotion
from app.services.tts.ssml_builder import SSMLBuilder
from app.services.tts.prosody_curve import google_tts_format_from_deltas

TTS_REST_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"

# ── SSML builder instance ─────────────────────────────────────────────────────
_ssml = SSMLBuilder()

# ── Emotion → (voice_name, gender) ────────────────────────────────────────────
# All Neural2 voices — highest quality available without Chirp
_VOICE_MAP: Dict[str, Tuple[str, str]] = {
    "joy":         ("en-US-Neural2-C", "FEMALE"),  # bright, expressive
    "excitement":  ("en-US-Neural2-C", "FEMALE"),  # bright, energetic
    "contentment": ("en-US-Neural2-F", "FEMALE"),  # warm, calm
    "sadness":     ("en-US-Neural2-F", "FEMALE"),  # empathetic, soft
    "grief":       ("en-US-Neural2-F", "FEMALE"),  # empathetic, deep
    "anger":       ("en-US-Neural2-D", "MALE"),    # assertive, male
    "frustration": ("en-US-Neural2-D", "MALE"),    # clipped, male
    "rage":        ("en-US-Neural2-D", "MALE"),    # forceful
    "fear":        ("en-US-Neural2-F", "FEMALE"),  # tense, quiet
    "anxiety":     ("en-US-Neural2-F", "FEMALE"),  # hushed, uncertain
    "surprise":    ("en-US-Neural2-C", "FEMALE"),  # bright, lifted
    "disgust":     ("en-US-Neural2-D", "MALE"),    # clipped
    "neutral":     ("en-US-Neural2-J", "MALE"),    # conversational, warm
}
_DEFAULT_VOICE = ("en-US-Neural2-J", "MALE")

# Indian English voices for Hinglish / Indian cadence detection
_INDIAN_VOICE_MAP: Dict[str, Tuple[str, str]] = {
    "joy":         ("en-IN-Neural2-A", "FEMALE"),
    "excitement":  ("en-IN-Neural2-A", "FEMALE"),
    "contentment": ("en-IN-Neural2-A", "FEMALE"),
    "sadness":     ("en-IN-Neural2-A", "FEMALE"),
    "grief":       ("en-IN-Neural2-A", "FEMALE"),
    "anger":       ("en-IN-Neural2-B", "MALE"),
    "frustration": ("en-IN-Neural2-B", "MALE"),
    "rage":        ("en-IN-Neural2-B", "MALE"),
    "fear":        ("en-IN-Neural2-A", "FEMALE"),
    "anxiety":     ("en-IN-Neural2-A", "FEMALE"),
    "surprise":    ("en-IN-Neural2-A", "FEMALE"),
    "disgust":     ("en-IN-Neural2-B", "MALE"),
    "neutral":     ("en-IN-Neural2-B", "MALE"),
}
_DEFAULT_INDIAN_VOICE = ("en-IN-Neural2-B", "MALE")

# Hinglish / Indian language markers for voice selection
_HINGLISH_MARKERS = re.compile(
    r"\b(yaar|bhai|acha|haan|nahi|matlab|toh|bas|chal|arre|oye|suno|dekho|"
    r"samjhe|theek|bilkul|ekdum|kya|hai|ho|gaya|karo|accha|ji|arrey|"
    r"chalo|abhi|bohot|bahut|lekin|kyunki|isliye|woh|yeh|mera|tera|"
    r"kuch|koi|kaisa|kaise|kitna|kidhar|idhar|udhar|wahan|yahan)\b",
    re.IGNORECASE,
)

# ── Emotion → device profile ──────────────────────────────────────────────────
_EMOTION_PROFILE: Dict[str, str] = {
    "grief":       "handset-class-device",           # intimate, warm
    "sadness":     "handset-class-device",
    "anxiety":     "handset-class-device",
    "fear":        "headphone-class-device",
    "anger":       "small-bluetooth-speaker-class-device",
    "rage":        "small-bluetooth-speaker-class-device",
    "frustration": "small-bluetooth-speaker-class-device",
    "joy":         "headphone-class-device",
    "excitement":  "small-bluetooth-speaker-class-device",
    "contentment": "headphone-class-device",
    "surprise":    "headphone-class-device",
    "disgust":     "small-bluetooth-speaker-class-device",
    "neutral":     "headphone-class-device",
}


# ── Prosody parsers ────────────────────────────────────────────────────────────

def _parse_rate(rate_str: str) -> float:
    """'+20%' → 1.2 | '-25%' → 0.75 | 'default' → 1.0"""
    if rate_str in ("default", None, ""):
        return 1.0
    try:
        pct = float(rate_str.replace("%", "").replace("+", ""))
        return round(max(0.25, min(4.0, 1.0 + pct / 100.0)), 2)
    except Exception:
        return 1.0


def _parse_pitch(pitch_str: str) -> float:
    """'+3.0st' → 3.0 | '-2.0st' → -2.0 | '+3Hz' → approx st | 'default' → 0.0"""
    if pitch_str in ("default", None, ""):
        return 0.0
    try:
        if "st" in pitch_str:
            return round(max(-20.0, min(20.0,
                float(pitch_str.replace("st", "").replace("+", ""))
            )), 1)
        elif "Hz" in pitch_str:
            # Convert Hz to semitones (approx: 1st ≈ 8.5Hz)
            hz = float(pitch_str.replace("Hz", "").replace("+", ""))
            return round(max(-20.0, min(20.0, hz / 8.5)), 1)
        return 0.0
    except Exception:
        return 0.0


def _parse_volume(vol_str: str) -> float:
    """+2.0dB' → 2.0 | '-2.0dB' → -2.0 | 'default' → 0.0"""
    if vol_str in ("default", None, ""):
        return 0.0
    try:
        if "dB" in vol_str:
            return round(max(-12.0, min(12.0,
                float(vol_str.replace("dB", "").replace("+", ""))
            )), 1)
        elif "%" in vol_str:
            # Convert rough percentage to dB equivalent
            pct = float(vol_str.replace("%", "").replace("+", ""))
            return round(max(-12.0, min(12.0, pct * 0.15)), 1)
        return 0.0
    except Exception:
        return 0.0


def _detect_hinglish(text: str) -> bool:
    """Return True if text contains Hinglish / Indian language markers."""
    matches = _HINGLISH_MARKERS.findall(text)
    # Need at least 2 markers to switch to Indian voice
    return len(matches) >= 2


def _strip_outer_prosody(ssml: str) -> str:
    """
    Remove the outer <speak>...</speak> and top-level <prosody> wrapper from SSML.
    Google TTS uses audioConfig for prosody — we keep <break> and <emphasis> only.
    Returns just the inner content, ready to be re-wrapped in <speak>.
    """
    # Remove <speak>...</speak>
    inner = re.sub(r"<speak>(.*)</speak>", r"\1", ssml, flags=re.DOTALL).strip()
    # Remove top-level <prosody ...>...</prosody> wrapper (keep inner content)
    inner = re.sub(r"<prosody[^>]*>(.*?)</prosody>", r"\1", inner, flags=re.DOTALL).strip()
    return inner


class GoogleCloudTTS(TTSEngine):

    def __init__(self):
        self.api_key = settings.GOOGLE_API_KEY
        self.available = bool(self.api_key)
        if not self.available:
            logger.warning("GOOGLE_API_KEY not set. Google TTS will not be used.")
        else:
            logger.info("Google Cloud TTS (REST) initialized with API key.")

    async def synthesize(
        self,
        text: str,
        ssml: str,
        filepath: str,
        prosody: Dict[str, str],
        emotion: str = "neutral",
        segment_deltas: list = None,
    ) -> str:
        if not self.available:
            raise TTSGenerationError("Google TTS API key not configured.")

        if not filepath.endswith(".mp3"):
            filepath = filepath.rsplit(".", 1)[0] + ".mp3"

        # Detect if we should use Indian English voices
        use_indian = _detect_hinglish(text)
        voice_map = _INDIAN_VOICE_MAP if use_indian else _VOICE_MAP
        default_voice = _DEFAULT_INDIAN_VOICE if use_indian else _DEFAULT_VOICE
        lang_code = "en-IN" if use_indian else "en-US"

        if use_indian:
            logger.info("Hinglish detected — using Indian English Neural2 voices")

        # Parse base prosody
        base_rate   = _parse_rate(prosody.get("rate", "default"))
        base_pitch  = _parse_pitch(prosody.get("pitch", "default"))
        base_vol    = _parse_volume(prosody.get("volume", "default"))

        logger.info(
            f"Google TTS [{emotion}]: rate={base_rate}x, "
            f"pitch={base_pitch}st, vol={base_vol}dB, lang={lang_code}"
        )

        # Get device profile for warmth
        device_profile = _EMOTION_PROFILE.get(emotion.lower(), "headphone-class-device")

        sentences = split_sentences(text)
        if not sentences:
            sentences = [text.strip()]

        # ── Strategy selection ──────────────────────────────────────────────
        has_deltas = bool(segment_deltas) and len(segment_deltas) > 0

        if len(sentences) <= 1 and not has_deltas:
            # Single sentence, no deltas: simple synthesis
            inner = _strip_outer_prosody(ssml)
            full_ssml = f"<speak>{inner}</speak>"
            voice_name, gender = voice_map.get(emotion.lower(), default_voice)
            audio = await self._call_api(
                full_ssml, voice_name, gender,
                base_rate, base_pitch, base_vol,
                lang_code, device_profile,
            )
            with open(filepath, "wb") as f:
                f.write(audio)
            logger.info(f"Google TTS audio saved: {filepath} ({len(audio)} bytes)")
            return filepath

        # ── Multi-segment: try composite SSML first, fallback to sequential ──

        # Build per-segment prosody values
        if has_deltas and len(segment_deltas) >= len(sentences):
            # LLM provided deltas — use them for per-segment prosody
            return await self._synthesize_with_deltas(
                sentences, segment_deltas, prosody, emotion,
                filepath, voice_map, default_voice, lang_code,
                device_profile,
            )
        else:
            # No LLM deltas — per-sentence emotion with shared base prosody
            return await self._synthesize_sequential(
                sentences, prosody, emotion, ssml,
                filepath, voice_map, default_voice, lang_code,
                device_profile, base_rate, base_pitch, base_vol,
            )

    async def _synthesize_with_deltas(
        self,
        sentences: List[str],
        segment_deltas: List[Dict],
        base_prosody: Dict[str, str],
        emotion: str,
        filepath: str,
        voice_map: Dict,
        default_voice: Tuple,
        lang_code: str,
        device_profile: str,
    ) -> str:
        """
        Synthesize using LLM-driven per-segment prosody deltas.

        STRATEGY 1: Composite SSML (all segments in one <speak>)
        - Enables smooth tone transitions
        - Uses a single voice for consistency
        - Per-segment <prosody> wrappers for variation

        Falls back to STRATEGY 2 (sequential) on failure.
        """
        # Build composite SSML segments
        ssml_segments = []
        for i, (sent, delta) in enumerate(zip(sentences, segment_deltas)):
            rate_m, pitch_st, vol_db = google_tts_format_from_deltas(
                base_prosody,
                rate_delta_pct  = int(delta.get("rate_delta_pct", 0)),
                pitch_delta_hz  = int(delta.get("pitch_delta_hz", 0)),
                volume_delta_db = float(delta.get("volume_delta_db", 0.0)),
            )
            seg_emotion = delta.get("emotion", emotion)
            ssml_segments.append({
                "text": sent,
                "rate_mult": rate_m,
                "pitch_st": pitch_st,
                "vol_db": vol_db,
                "pause_before_ms": int(delta.get("pause_before_ms", 0)),
                "emotion": seg_emotion,
                "emphasis_words": delta.get("emphasis_words", []),
            })

        # Try composite SSML first (all segments in one call)
        composite_ssml = _ssml.build_google_composite_ssml(ssml_segments, emotion)
        voice_name, gender = voice_map.get(emotion.lower(), default_voice)

        # Use the MIDPOINT of all segment rates for audioConfig base
        avg_rate = sum(s["rate_mult"] for s in ssml_segments) / len(ssml_segments)
        avg_pitch = sum(s["pitch_st"] for s in ssml_segments) / len(ssml_segments)
        avg_vol = sum(s["vol_db"] for s in ssml_segments) / len(ssml_segments)

        logger.info(
            f"Google TTS composite SSML: {len(sentences)} segments, "
            f"avg_rate={avg_rate:.2f}x, avg_pitch={avg_pitch:.1f}st"
        )

        try:
            audio = await self._call_api(
                composite_ssml, voice_name, gender,
                avg_rate, avg_pitch, avg_vol,
                lang_code, device_profile,
            )
            with open(filepath, "wb") as f:
                f.write(audio)
            logger.info(
                f"Google TTS composite audio saved: {filepath} ({len(audio)} bytes)"
            )
            return filepath

        except TTSGenerationError as e:
            logger.warning(
                f"Google TTS composite SSML failed ({e}). "
                f"Falling back to sequential segment synthesis."
            )

        # FALLBACK: Sequential per-segment synthesis
        all_audio = bytearray()
        for i, (sent, seg_info) in enumerate(zip(sentences, ssml_segments)):
            s_emotion = seg_info.get("emotion", emotion)
            voice_name, gender = voice_map.get(s_emotion.lower(), default_voice)

            # Build individual segment SSML
            emphasis = seg_info.get("emphasis_words", [])
            seg_ssml = _ssml.build_ssml_engine(sent, emotion=s_emotion, extra_emphasis=emphasis)

            rate_m = seg_info["rate_mult"]
            pitch_st = seg_info["pitch_st"]
            vol_db = seg_info["vol_db"]

            logger.info(
                f"  Seg {i+1}/{len(sentences)}: emotion={s_emotion} "
                f"voice={voice_name} rate={rate_m}x pitch={pitch_st}st vol={vol_db}dB"
            )
            try:
                chunk = await self._call_api(
                    seg_ssml, voice_name, gender,
                    rate_m, pitch_st, vol_db,
                    lang_code, device_profile,
                )
                all_audio.extend(chunk)
            except Exception as e:
                logger.error(f"  Google TTS seg {i+1} failed: {e}. Skipping.")

        if not all_audio:
            raise TTSGenerationError("Google TTS: all segments failed.")

        with open(filepath, "wb") as f:
            f.write(bytes(all_audio))
        logger.info(f"Google TTS multi-segment audio saved: {filepath} ({len(all_audio)} bytes)")
        return filepath

    async def _synthesize_sequential(
        self,
        sentences: List[str],
        prosody: Dict[str, str],
        emotion: str,
        ssml: str,
        filepath: str,
        voice_map: Dict,
        default_voice: Tuple,
        lang_code: str,
        device_profile: str,
        base_rate: float,
        base_pitch: float,
        base_vol: float,
    ) -> str:
        """
        Sequential per-sentence synthesis with emotion-aware voice selection.
        Each sentence gets its own emotion detection → voice selection.
        """
        logger.info(
            f"Google TTS sequential: {len(sentences)} segments, "
            f"per-sentence emotion voices active"
        )
        all_audio = bytearray()
        for i, sentence in enumerate(sentences):
            s_emotion, _ = detect_sentence_emotion(sentence)
            voice_name, gender = voice_map.get(s_emotion.lower(), default_voice)

            # Build SSML with emphasis for this segment
            seg_ssml = _ssml.build_ssml_engine(sentence, emotion=s_emotion)

            logger.info(
                f"  Seg {i+1}/{len(sentences)}: emotion={s_emotion} "
                f"voice={voice_name}"
            )
            try:
                chunk = await self._call_api(
                    seg_ssml, voice_name, gender,
                    base_rate, base_pitch, base_vol,
                    lang_code, device_profile,
                )
                all_audio.extend(chunk)
            except Exception as e:
                logger.error(f"  Google TTS seg {i+1} failed: {e}. Skipping.")

        if not all_audio:
            raise TTSGenerationError("Google TTS: all sentence segments failed.")

        with open(filepath, "wb") as f:
            f.write(bytes(all_audio))
        logger.info(f"Google TTS multi-segment audio saved: {filepath} ({len(all_audio)} bytes)")
        return filepath

    async def _call_api(
        self,
        ssml: str,
        voice_name: str,
        gender: str,
        rate: float,
        pitch: float,
        volume_db: float,
        lang_code: str = "en-US",
        device_profile: str = "headphone-class-device",
    ) -> bytes:
        """Make a single REST call to the Google Cloud TTS API. Returns raw MP3 bytes."""
        payload = {
            "input": {"ssml": ssml},
            "voice": {
                "languageCode": lang_code,
                "name": voice_name,
                "ssmlGender": gender,
            },
            "audioConfig": {
                "audioEncoding":  "MP3",
                "speakingRate":   rate,
                "pitch":          pitch,
                "volumeGainDb":   volume_db,
                "effectsProfileId": [device_profile],
                "sampleRateHertz": 24000,  # higher quality audio
            },
        }

        try:
            async with httpx.AsyncClient(timeout=25.0) as client:
                response = await client.post(
                    TTS_REST_URL,
                    params={"key": self.api_key},
                    json=payload,
                )

            if response.status_code != 200:
                detail = response.text
                logger.error(f"Google TTS API error {response.status_code}: {detail}")
                if response.status_code in (401, 403):
                    self.available = False
                    logger.warning(
                        "Google TTS permanently disabled for this session "
                        f"(HTTP {response.status_code}). Will use next provider."
                    )
                raise TTSGenerationError(
                    f"Google TTS returned {response.status_code}: {detail}"
                )

            data = response.json()
            return base64.b64decode(data["audioContent"])

        except TTSGenerationError:
            raise
        except Exception as e:
            raise TTSGenerationError(f"Google TTS request failed: {e}")

    @staticmethod
    def _escape(text: str) -> str:
        """XML-escape plain text for embedding in SSML."""
        return (
            text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&apos;")
        )
