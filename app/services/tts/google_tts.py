from app.services.tts.base import TTSEngine
from app.core.exceptions import TTSGenerationError
from app.core.logging_config import logger
from app.core.config import settings
import os

try:
    from google.cloud import texttospeech
except ImportError:
    texttospeech = None

class GoogleCloudTTS(TTSEngine):
    def __init__(self):
        if not texttospeech or (not settings.GOOGLE_APPLICATION_CREDENTIALS and not settings.GOOGLE_API_KEY):
            logger.warning("Google Cloud TTS not configured or library missing.")
            self.client = None
        else:
            try:
                if settings.GOOGLE_API_KEY:
                    self.client = texttospeech.TextToSpeechAsyncClient(client_options={"api_key": settings.GOOGLE_API_KEY})
                else:
                    self.client = texttospeech.TextToSpeechAsyncClient()
            except Exception as e:
                logger.error(f"Failed to initialize Google TTS: {e}")
                self.client = None

    async def synthesize(self, text: str, ssml: str, filepath: str) -> str:
        if not self.client:
            raise TTSGenerationError("Google TTS Engine is not properly initialized.")

        try:
            synthesis_input = texttospeech.SynthesisInput(ssml=ssml)

            # Abstract language/voice choice; ideal to link emotion back to voice here
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-US",
                name="en-US-Journey-D" # robust, expressive voice
            )

            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )

            response = await self.client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )

            with open(filepath, "wb") as out:
                out.write(response.audio_content)

            return filepath
        except Exception as e:
            logger.error(f"Google Cloud TTS synthesis failed: {e}", exc_info=True)
            raise TTSGenerationError(f"Cloud API error: {e}")
