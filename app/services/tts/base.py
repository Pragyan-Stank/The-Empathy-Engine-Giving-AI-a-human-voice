from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class TTSEngine(ABC):
    @abstractmethod
    async def synthesize(
        self,
        text: str,
        ssml: str,
        filepath: str,
        prosody: Dict[str, str],
        emotion: str = "neutral",
        segment_deltas: list = None,
    ) -> str:
        """
        Synthesize audio and save to filepath. Return the saved filepath.

        Args:
            text:    Plain text (used by engines that don't parse SSML).
            ssml:    Structured SSML markup (used by engines that support it).
            filepath: Target output path.
            prosody: {"rate": "+20%", "pitch": "+2.0st", "volume": "+2.0dB"}
            emotion: Canonical emotion label for engines that use emotion-aware settings.
            segment_deltas: Per-segment prosody deltas from LLM speech analysis.
                Each dict contains: rate_delta_pct, pitch_delta_hz, volume_delta_db,
                pause_before_ms, emotion, emphasis_words.
        """
        pass
