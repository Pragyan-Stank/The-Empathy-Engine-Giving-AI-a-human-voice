import hashlib
import os
from typing import Dict, Optional
from app.core.config import settings

class AudioStorageService:
    def __init__(self):
        self.output_dir = settings.OUTPUT_AUDIO_DIR
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    def generate_filename(
        self,
        text: str,
        emotion: str,
        intensity: float,
        prosody: Optional[Dict[str, str]] = None,
        extension: str = "mp3",
    ) -> str:
        """
        Deterministic filename that includes ALL relevant synthesis parameters.
        Different slider positions → different hashes → no stale cache.
        """
        prosody_key = ""
        if prosody:
            prosody_key = f"_{prosody.get('rate','')}_{prosody.get('pitch','')}_{prosody.get('volume','')}"
        content = f"{text}_{emotion}_{intensity}{prosody_key}".encode("utf-8")
        md5_hash = hashlib.md5(content).hexdigest()
        return f"{md5_hash}.{extension}"

    def get_filepath(self, filename: str) -> str:
        return os.path.join(self.output_dir, filename)

    def file_exists(self, filename: str) -> bool:
        return os.path.exists(self.get_filepath(filename))
