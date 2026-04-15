import hashlib
import os
from app.core.config import settings

class AudioStorageService:
    def __init__(self):
        self.output_dir = settings.OUTPUT_AUDIO_DIR
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    def generate_filename(self, text: str, emotion: str, intensity: float) -> str:
        """
        Creates a deterministic filename so identical text + params use cached audio.
        """
        content = f"{text}_{emotion}_{intensity}".encode('utf-8')
        md5_hash = hashlib.md5(content).hexdigest()
        return f"{md5_hash}.mp3"

    def get_filepath(self, filename: str) -> str:
        return os.path.join(self.output_dir, filename)

    def file_exists(self, filename: str) -> bool:
        return os.path.exists(self.get_filepath(filename))
