"""
Text Enhancement Layer — converts raw input into expressive, speech-ready text.

Pipeline:
  1. Normalize whitespace / basic cleanup
  2. Hinglish & code-mix awareness (preserve contractions, transliterations)
  3. Punctuation-as-emotion signals:
       "..." → hesitation / sadness break
       "!"   → excitement / urgency
       "?"   → curiosity / rising intonation
  4. Auto-inject natural punctuation where missing (ellipsis on trailing
     negative sentiment, "!" on all-caps words, etc.)
  5. Split long sentences into conversational chunks (≤18 words)
  6. Insert <break> markers that the SSML builder picks up

Output: enhanced plain text (NOT SSML) fed into the SSML builder.
"""
import re
from typing import List


# ── Constants ─────────────────────────────────────────────────────────────────

# Words that signal a natural trailing-off / hesitation
_HESITATION_WORDS = {
    "but", "however", "though", "although", "still", "yet",
    "anyway", "regardless", "maybe", "perhaps", "i think",
    "i guess", "not sure", "i don't know",
}

# Hinglish filler / discourse markers — preserve as-is, never split on them
_HINGLISH_FILLERS = {
    "yaar", "bhai", "bro", "sis", "acha", "haan", "nahi", "matlab",
    "toh", "bas", "chal", "arre", "oye", "suno", "dekho", "samjhe",
    "theek", "bilkul", "ekdum", "kya", "hai", "ho", "gaya", "karo",
}

# Emotion-to-pause length (ms) — injected as ||Xms|| markers
_PAUSE_MAP = {
    "grief":       700,
    "sadness":     500,
    "anxiety":     400,
    "frustration": 350,
    "anger":       300,
    "fear":        350,
    "neutral":     250,
    "contentment": 200,
    "joy":         150,
    "excitement":  100,
    "surprise":    200,
    "rage":        200,
}

# Split long sentences at these connector words
_SPLIT_CONNECTORS = re.compile(
    r"\b(and then|but then|because|so that|which means|therefore|however|although|"
    r"even though|in fact|what's more|on top of that|not only that)\b",
    re.IGNORECASE,
)

MAX_WORDS_PER_CHUNK = 18


# ── Public API ────────────────────────────────────────────────────────────────

def enhance_text(text: str, emotion: str = "neutral", intensity: float = 1.0) -> str:
    """
    Full text enhancement pipeline.

    Returns enhanced plain text with ||Xms|| pause markers embedded.
    The SSML builder converts those markers to <break time="Xms"/> tags.
    """
    text = _normalize(text)
    text = _inject_auto_punctuation(text, emotion)
    chunks = _split_to_chunks(text)
    chunks = _inject_pauses(chunks, emotion, intensity)
    return " ".join(chunks)


def get_pause_ms(emotion: str, intensity: float = 1.0) -> int:
    """Return the base pause length in ms for an emotion × intensity."""
    base = _PAUSE_MAP.get(emotion.lower(), 250)
    return int(base * max(0.5, intensity))


# ── Private Helpers ───────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Clean up whitespace, fix common encoding artifacts."""
    text = text.strip()
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    # Normalize unicode ellipsis → ASCII triple-dot
    text = text.replace("\u2026", "...")
    # Normalize unicode dashes
    text = re.sub(r"[\u2013\u2014]", " — ", text)
    # Fix missing space after punctuation (e.g. "hello.how" → "hello. how")
    text = re.sub(r"([.!?])([A-Za-z])", r"\1 \2", text)
    return text


def _inject_auto_punctuation(text: str, emotion: str) -> str:
    """
    Automatically inject expressive punctuation based on emotion signals.

    Rules:
    - Sentences ending in negative emotion words without punctuation → append "..."
    - ALL-CAPS words with positive/anger emotion → append "!" if not already
    - Questions phrased without "?" → add "?"
    """
    sentences = re.split(r"(?<=[.!?…])\s+", text)
    enhanced = []

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        lower = sent.lower()
        words = sent.split()
        last_word = re.sub(r"[^a-zA-Z]", "", words[-1]).lower() if words else ""

        # ALL-CAPS word mid-sentence (excitement / anger) → ensure !
        if re.search(r"\b[A-Z]{3,}\b", sent) and not sent.endswith(("!", "!!")):
            if emotion in ("excitement", "anger", "rage"):
                if not sent[-1] in ".!?":
                    sent += "!"

        # Trailing hesitation word without punctuation → ellipsis
        elif last_word in _HESITATION_WORDS and not sent[-1] in ".!?...":
            if emotion in ("sadness", "grief", "anxiety", "neutral"):
                sent += "..."

        # Question without "?" (simple heuristic: starts with wh-word / do-you)
        elif (
            not sent[-1] in ".!?"
            and re.match(r'^(who|what|where|when|why|how|do you|does|did|is it|are you|can you|would you)', lower)
        ):
            sent += "?"

        # Sentence with no terminal punctuation at all
        elif not sent[-1] in ".!?…":
            sent += "."

        enhanced.append(sent)

    return " ".join(enhanced)


def _split_to_chunks(text: str) -> List[str]:
    """
    Split text into short, conversational chunks.

    Strategy:
    1. Split at sentence boundaries first.
    2. If a sentence > MAX_WORDS_PER_CHUNK, split at connector words.
    3. If still too long, hard-split at comma boundaries.
    """
    # First pass: split at sentence boundaries
    sentences = re.split(r"(?<=[.!?…])\s+", text)
    chunks: List[str] = []

    for sent in sentences:
        words = sent.split()
        if len(words) <= MAX_WORDS_PER_CHUNK:
            if sent.strip():
                chunks.append(sent.strip())
            continue

        # Try splitting at connectors
        parts = _SPLIT_CONNECTORS.split(sent)
        # Re-attach connector words to following part
        rebuilt: List[str] = []
        i = 0
        while i < len(parts):
            part = parts[i].strip()
            if not part:
                i += 1
                continue
            # Check if this is a connector word
            if _SPLIT_CONNECTORS.fullmatch(part.strip()):
                # Attach connector to next part
                if i + 1 < len(parts):
                    rebuilt.append((part + " " + parts[i + 1]).strip())
                    i += 2
                    continue
            rebuilt.append(part)
            i += 1

        for part in rebuilt:
            part_words = part.split()
            if len(part_words) <= MAX_WORDS_PER_CHUNK:
                if part.strip():
                    # Ensure terminal punctuation
                    if part[-1] not in ".!?…,":
                        part += ","
                    chunks.append(part.strip())
            else:
                # Hard split at commas
                comma_parts = re.split(r",\s*", part)
                for cp in comma_parts:
                    cp = cp.strip()
                    if cp:
                        if cp[-1] not in ".!?…,":
                            cp += ","
                        chunks.append(cp)

    return [c for c in chunks if c.strip()]


def _inject_pauses(chunks: List[str], emotion: str, intensity: float) -> List[str]:
    """
    Inject ||Xms|| pause markers between chunks.

    Pause length scales with:
    - Emotion (grief = long, excitement = short)
    - Punctuation at end of chunk:
        "..."  → emotion pause × 1.5
        "."    → emotion pause × 1.0
        ","    → emotion pause × 0.4
        "!" "?"→ emotion pause × 0.6
    """
    base_ms = get_pause_ms(emotion, intensity)
    result: List[str] = []

    for i, chunk in enumerate(chunks):
        result.append(chunk)
        if i == len(chunks) - 1:
            break  # no pause after last chunk

        last_char = chunk.rstrip()[-1] if chunk.rstrip() else "."
        if chunk.rstrip().endswith("..."):
            pause_ms = int(base_ms * 1.5)
        elif last_char == ".":
            pause_ms = base_ms
        elif last_char in "!?":
            pause_ms = int(base_ms * 0.6)
        elif last_char == ",":
            pause_ms = int(base_ms * 0.4)
        else:
            pause_ms = int(base_ms * 0.8)

        # Clamp: 80ms min, 900ms max
        pause_ms = max(80, min(900, pause_ms))
        result.append(f"||{pause_ms}ms||")

    return result
