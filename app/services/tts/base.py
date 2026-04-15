from abc import ABC, abstractmethod

class TTSEngine(ABC):
    @abstractmethod
    async def synthesize(self, text: str, ssml: str, filepath: str) -> str:
        """
        Synthesize audio and save to filepath. Return full filepath on success.
        """
        pass
