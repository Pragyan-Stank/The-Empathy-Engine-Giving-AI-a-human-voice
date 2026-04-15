from abc import ABC, abstractmethod
from typing import Dict


class TTSEngine(ABC):
    @abstractmethod
    async def synthesize(
        self,
        text: str,
        ssml: str,
        filepath: str,
        prosody: Dict[str, str],
    ) -> str:
        """
        Synthesize audio and save to filepath.
        prosody: {"rate": "+20%", "pitch": "+2.0st", "volume": "+2.0dB"}
        Returns the filepath on success.
        """
        pass
