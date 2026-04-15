from app.services.tts.base import TTSEngine
from app.core.logging_config import logger
import pyttsx3
import asyncio
import os
import xml.etree.ElementTree as ET

class FallbackTTS(TTSEngine):
    """
    Uses pyttsx3. Does not actually interpret SSML properly, so we strip SSML tags for pyttsx3
    and loosely apply the prosody logic since pyttsx3 accepts rate and volume configurations globally.
    """
    def __init__(self):
        # We delay init because pyttsx3 can be blocky
        pass

    def _strip_tags(self, text: str) -> str:
        try:
            # A very naive XML strip, effectively getting text out of simple tags
            # Since SSML is well-formed XML in our builder, ElementTree handles it easily
            root = ET.fromstring(text)
            return "".join(root.itertext())
        except Exception:
            # fallback regex
            import re
            return re.sub(r'<[^>]+>', '', text)

    async def synthesize(self, text: str, ssml: str, filepath: str) -> str:
        def _run():
            logger.info("pyttsx3 synthesizing locally")
            engine = pyttsx3.init()
            
            # Extract plain text
            plain_text = self._strip_tags(ssml)
            
            # Just do basic set properties
            engine.setProperty('rate', 150)
            engine.save_to_file(plain_text, filepath)
            engine.runAndWait()
            return filepath
            
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _run)
