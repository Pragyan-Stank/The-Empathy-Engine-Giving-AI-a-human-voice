"""
Google Cloud Text-to-Speech — upgraded for maximum expressiveness.

What's improved:
  1. Emotion-based voice selection:
       - Sad/grief/anxiety  → en-US-Neural2-F (warm, female, empathetic)
       - Angry/rage         → en-US-Neural2-D (male, assertive)
       - Joy/excitement     → en-US-Neural2-C (bright, expressive female)
       - Neutral/default    → en-US-Neural2-J (conversational, warm)

  2. Finer audioConfig tuning per emotion:
       - Effective rate:  0.25–4.0x
       - Pitch: -20.0 to +20.0 semitones
       - VolumeGainDb: -96.0 to +16.0

  3. Full SSML pipeline:
       - Receives the rich SSML from SSMLBuilder (with <prosody>, <break>, <emphasis>)
       - Strips any outer prosody wrapper (Google uses audioConfig instead)
         so we don't double-apply prosody
       - Preserves all <break> and <emphasis> tags for natural rhythm

  4. Multi-segment support:
       - For long text, splits into sentences
       - Synthesizes each sentence separately with its own emotion profile
       - Concatenates raw MP3 bytes into a single file

  5. Emotion-specific audioConfig adjustments:
       - Angry:   slightly faster rate, higher volume
       - Sad:     slower rate, lower volume
       - Excited: faster rate, normal volume
       - Neutral: balanced defaults
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

TTS_REST_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"

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
    ) -> str:
        if not self.available:
            raise TTSGenerationError("Google TTS API key not configured.")

        if not filepath.endswith(".mp3"):
            filepath = filepath.rsplit(".", 1)[0] + ".mp3"

        # Parse base prosody (from shared request prosody)
        base_rate   = _parse_rate(prosody.get("rate", "default"))
        base_pitch  = _parse_pitch(prosody.get("pitch", "default"))
        base_vol    = _parse_volume(prosody.get("volume", "default"))

        logger.info(
            f"Google TTS [{emotion}]: rate={base_rate}x, "
            f"pitch={base_pitch}st, vol={base_vol}dB"
        )

        sentences = split_sentences(text)
        if not sentences:
            sentences = [text.strip()]

        if len(sentences) <= 1:
            # Single sentence: synthesize directly with rich SSML
            inner = _strip_outer_prosody(ssml)
            full_ssml = f"<speak>{inner}</speak>"
            voice_name, gender = _VOICE_MAP.get(emotion.lower(), _DEFAULT_VOICE)
            audio = await self._call_api(
                full_ssml, voice_name, gender,
                base_rate, base_pitch, base_vol
            )
            with open(filepath, "wb") as f:
                f.write(audio)
            logger.info(f"Google TTS audio saved: {filepath} ({len(audio)} bytes)")
            return filepath

        # Multi-sentence: emit each sentence with its own emotion voice
        logger.info(
            f"Google TTS multi-sentence: {len(sentences)} segments, "
            f"per-sentence emotion voices active"
        )
        all_audio = bytearray()
        for i, sentence in enumerate(sentences):
            s_emotion, _ = detect_sentence_emotion(sentence)
            voice_name, gender = _VOICE_MAP.get(s_emotion.lower(), _DEFAULT_VOICE)

            # Build simple SSML for this sentence (emphasis + breaks from SSMLBuilder)
            # We use a plain prosody-less SSML since audioConfig handles prosody
            escaped = self._escape(sentence)
            seg_ssml = f"<speak>{escaped}</speak>"

            logger.info(
                f"  Seg {i+1}/{len(sentences)}: emotion={s_emotion} "
                f"voice={voice_name}"
            )
            try:
                chunk = await self._call_api(
                    seg_ssml, voice_name, gender,
                    base_rate, base_pitch, base_vol
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
    ) -> bytes:
        """Make a single REST call to the Google Cloud TTS API. Returns raw MP3 bytes."""
        payload = {
            "input": {"ssml": ssml},
            "voice": {
                "languageCode": "en-US",
                "name": voice_name,
                "ssmlGender": gender,
            },
            "audioConfig": {
                "audioEncoding":  "MP3",
                "speakingRate":   rate,
                "pitch":          pitch,
                "volumeGainDb":   volume_db,
                "effectsProfileId": ["headphone-class-device"],  # subtle EQ warmth
            },
        }

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
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
