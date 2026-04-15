import re
from typing import Dict, Any

class SSMLBuilder:
    def __init__(self):
        # We can add vendor-specific namespaces here later
        pass

    def build_ssml(self, text: str, prosody: Dict[str, str]) -> str:
        """
        Wrap text in SSML tags. applies <prosody> and extracts structural clues
        like exclamation points for pauses.
        """
        # A simple cleanup
        clean_text = self._escape_xml(text)
        
        # Word emphasis mock-up using <emphasis> for strong words
        clean_text = self._apply_word_emphasis(clean_text)
        
        # Build prosody tag
        rate = prosody.get("rate", "default")
        pitch = prosody.get("pitch", "default")
        volume = prosody.get("volume", "default")
        
        prosody_attrs = []
        if rate != "default": prosody_attrs.append(f'rate="{rate}"')
        if pitch != "default": prosody_attrs.append(f'pitch="{pitch}"')
        if volume != "default": prosody_attrs.append(f'volume="{volume}"')
        
        prosody_str = " ".join(prosody_attrs)
        
        if prosody_str:
            ssml_content = f"<prosody {prosody_str}>{clean_text}</prosody>"
        else:
            ssml_content = clean_text

        # Return standard SSML
        return f"<speak>{ssml_content}</speak>"

    def _escape_xml(self, text: str) -> str:
        text = text.replace("&", "&amp;")
        text = text.replace("<", "&lt;")
        text = text.replace(">", "&gt;")
        text = text.replace('"', "&quot;")
        text = text.replace("'", "&apos;")
        return text

    def _apply_word_emphasis(self, text: str) -> str:
        strong_words = {"important", "urgent", "sorry", "great", "never", "always", "critical"}
        words = text.split()
        emphasized = []
        for w in words:
            # strip punct to check
            clean_w = re.sub(r'[^\w\s]', '', w.lower())
            if clean_w in strong_words:
                emphasized.append(f'<emphasis level="strong">{w}</emphasis>')
            else:
                emphasized.append(w)
        return " ".join(emphasized)
