"""
SSMLBuilder produces two flavours of SSML:
  - build_ssml_display(text, prosody)  → full SSML with <prosody> for UI preview
  - build_ssml_engine(text)            → emphasis/pauses only (Google uses audioConfig for prosody)

Both apply word-level <emphasis> and <break> after sentence-ending punctuation.
"""
import re
from typing import Dict


# Words that receive <emphasis level="strong">
EMPHASIS_WORDS = {
    "important", "urgent", "sorry", "great", "never", "always",
    "critical", "love", "hate", "terrible", "amazing", "horrible",
    "please", "help", "immediately", "now", "desperate", "incredible",
}


class SSMLBuilder:
    def build_ssml_display(self, text: str, prosody: Dict[str, str]) -> str:
        """Full SSML with <prosody> for UI preview panel."""
        inner = self._build_inner(text)
        rate = prosody.get("rate", "default")
        pitch = prosody.get("pitch", "default")
        volume = prosody.get("volume", "default")

        attrs = []
        if rate != "default":
            attrs.append(f'rate="{rate}"')
        if pitch != "default":
            attrs.append(f'pitch="{pitch}"')
        if volume != "default":
            attrs.append(f'volume="{volume}"')

        if attrs:
            inner = f'<prosody {" ".join(attrs)}>{inner}</prosody>'

        return f"<speak>{inner}</speak>"

    def build_ssml_engine(self, text: str) -> str:
        """
        SSML for the TTS engine — NO <prosody> wrapper (prosody is set via
        audioConfig / engine properties to avoid double-application).
        Contains emphasis and pause markup only.
        """
        inner = self._build_inner(text)
        return f"<speak>{inner}</speak>"

    # Keep backward-compatible alias used by old call sites
    def build_ssml(self, text: str, prosody: Dict[str, str]) -> str:
        return self.build_ssml_display(text, prosody)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_inner(self, text: str) -> str:
        escaped = self._escape_xml(text)
        with_emphasis = self._apply_word_emphasis(escaped)
        with_pauses = self._apply_pauses(with_emphasis)
        return with_pauses

    def _escape_xml(self, text: str) -> str:
        text = text.replace("&", "&amp;")
        text = text.replace("<", "&lt;")
        text = text.replace(">", "&gt;")
        text = text.replace('"', "&quot;")
        text = text.replace("'", "&apos;")
        return text

    def _apply_word_emphasis(self, text: str) -> str:
        words = text.split()
        out = []
        for w in words:
            clean_w = re.sub(r'[^\w]', '', w.lower())
            if clean_w in EMPHASIS_WORDS:
                out.append(f'<emphasis level="strong">{w}</emphasis>')
            else:
                out.append(w)
        return " ".join(out)

    def _apply_pauses(self, text: str) -> str:
        """Insert a short <break> after sentence-ending punctuation."""
        # After . ! ? that are NOT inside a tag
        text = re.sub(r'([.!?])(?!\s*<)', r'\1<break time="300ms"/>', text)
        return text
