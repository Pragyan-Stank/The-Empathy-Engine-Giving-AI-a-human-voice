"""
Language Detection Utility — detects Hindi, Hinglish, and English input.

Used throughout the pipeline to ensure:
  1. The LLM never translates Hindi/Hinglish text to English.
  2. TTS engines select the correct voice (hi-IN vs en-US/en-IN).
  3. SSML xml:lang is set appropriately.

Detection Hierarchy:
  1. Devanagari script presence → "hi" (pure Hindi)
  2. High density of Romanized Hindi words → "hi-Latn" (Hinglish / Romanized Hindi)
  3. Default → "en" (English)

The returned language tag drives voice selection and prompt behaviour:
  - "hi"      → Use hi-IN voices, tell LLM to absolutely not translate
  - "hi-Latn" → Use hi-IN or en-IN voices, tell LLM to keep Hinglish intact
  - "en"      → Standard English pipeline
"""
import re
from typing import Tuple


# ── Devanagari Unicode range ──────────────────────────────────────────────────
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

# ── Romanized Hindi / Hinglish vocabulary ─────────────────────────────────────
# Comprehensive set of common Hindi words written in Roman script.
# These words almost never appear in natural English text.
_HINGLISH_WORDS = {
    # Pronouns & determiners
    "main", "mein", "mujhe", "mujhko", "hum", "humko", "humne",
    "tum", "tumko", "tumhara", "tumhari", "aap", "aapka", "aapki", "aapke",
    "woh", "yeh", "uska", "uski", "uske", "iska", "iski", "iske",
    "mera", "meri", "mere", "tera", "teri", "tere", "hamara", "hamari",
    "unka", "unki", "unke", "inhe", "unhe", "sabka",
    # Common verbs
    "hai", "hain", "tha", "thi", "the", "hoga", "hogi", "hoge",
    "karo", "karna", "karta", "karti", "karte", "kar", "kiya", "ki",
    "ho", "hona", "hota", "hoti", "hote", "hua", "hui", "hue",
    "ja", "jao", "jana", "jata", "jati", "jaate", "gaya", "gayi", "gaye",
    "aa", "aao", "aana", "aata", "aati", "aate", "aaya", "aayi", "aaye",
    "de", "do", "dena", "deta", "deti", "dete", "diya", "di", "diye",
    "le", "lo", "lena", "leta", "leti", "lete", "liya", "li", "liye",
    "bol", "bolo", "bolna", "bolta", "bolti", "bolte", "bola", "boli", "bole",
    "dekh", "dekho", "dekhna", "dekhta", "dekhti", "dekhte", "dekha", "dekhi", "dekhe",
    "sun", "suno", "sunna", "sunta", "sunti", "sunte", "suna", "suni", "sune",
    "samajh", "samjho", "samjhe", "samajhna", "samjha", "samjhi",
    "baith", "baitho", "baithna", "baitha", "baithi", "baithe",
    "rakh", "rakho", "rakhna", "rakha", "rakhi", "rakhe",
    "chal", "chalo", "chalna", "chala", "chali", "chale",
    "ruk", "ruko", "rukna", "ruka", "ruki", "ruke",
    "soch", "socho", "sochna", "socha", "sochi", "soche",
    "bana", "banao", "banana", "banata", "banati", "banate", "banaya", "banayi",
    "khao", "khana", "khata", "khati", "khate", "khaya", "khayi",
    "piyo", "peena", "peeta", "peeti", "peete", "piya", "piyi",
    "likhna", "likhta", "likhti", "likhte", "likha", "likhi", "likhe",
    "padhna", "padhta", "padhti", "padhte", "padha", "padhi", "padhe",
    "milna", "milta", "milti", "milte", "mila", "mili", "mile",
    "chalna", "chalta", "chalti", "chalte",
    "rona", "rota", "roti", "rote", "roya", "royi",
    "hasna", "hasta", "hasti", "haste", "hasa", "hasi",
    "sona", "sota", "soti", "sote", "soya", "soyi",
    "uthna", "uthta", "uthti", "uthte", "utha", "uthi",
    "baat", "batao", "batana", "batata", "batati", "bataya",
    "puchna", "puchta", "puchti", "pucha", "puchi",
    "rehna", "rehta", "rehti", "rehte", "raha", "rahi", "rahe",
    "chahna", "chahta", "chahti", "chahte", "chaha", "chahi",
    # Common nouns
    "kaam", "ghar", "dost", "pyaar", "dil", "zindagi", "duniya", "sapna",
    "raat", "din", "subah", "shaam", "waqt", "jagah", "taraf", "cheez",
    "baat", "khabar", "paisa", "log", "banda", "ladka", "ladki",
    "bachcha", "aadmi", "aurat", "bhai", "behen", "maa", "papa", "baap",
    "beta", "beti", "didi", "chacha", "mausi", "nana", "nani", "dada", "dadi",
    # Adjectives / adverbs
    "accha", "acha", "bura", "chhota", "bada", "naya", "purana", "khubsurat",
    "theek", "sahi", "galat", "mushkil", "aasaan", "zyada", "kam", "bahut",
    "bohot", "thoda", "bilkul", "ekdum", "sacchi", "sach", "jhooth",
    "pehle", "baad", "abhi", "kabhi", "hamesha", "firse", "phirse", "dubara",
    # Question words
    "kya", "kaise", "kaisa", "kyun", "kyu", "kyunki", "kab", "kahan",
    "kidhar", "kitna", "kitni", "kitne", "kaun", "kisko", "kiske",
    # Connectors / particles
    "toh", "par", "lekin", "magar", "isliye", "islye", "warna", "aur",
    "ya", "phir", "matlab", "matlab", "yaani", "jaise",
    # Fillers / exclamations
    "yaar", "arre", "arrey", "oye", "haan", "nahi", "nah", "na",
    "ji", "bas", "chal", "chalo", "achha", "suno", "dekho",
    # Postpositions
    "mein", "par", "pe", "se", "ko", "ka", "ki", "ke", "tak", "wala", "wali", "wale",
    # Miscellaneous common words
    "kuch", "koi", "sab", "idhar", "udhar", "wahan", "yahan",
    "agar", "jab", "tab", "toh", "bhi",
    "sirf", "hi", "nahi", "mat", "kabhi",
    "zaroor", "shayad", "pata",
}

# Words that appear in BOTH English and Hinglish — don't count these
_AMBIGUOUS_WORDS = {
    "hi", "par", "main", "ki", "tab", "mat", "is", "it", "the", "a",
    "an", "to", "in", "on", "at", "for", "of", "and", "or", "but",
    "not", "no", "so", "do", "go", "be", "am", "are", "was", "were",
    "has", "have", "had", "will", "can", "may", "did", "got",
}


def detect_language(text: str) -> Tuple[str, float]:
    """
    Detect the primary language of the input text.

    Returns:
        Tuple of (language_tag, confidence):
          - "hi"      — Devanagari Hindi (confidence 0.0–1.0)
          - "hi-Latn" — Romanized Hindi / Hinglish (confidence 0.0–1.0)
          - "en"      — English (confidence 0.0–1.0)
    """
    if not text or not text.strip():
        return "en", 1.0

    # ── Step 1: Check for Devanagari script ──────────────────────────────
    devanagari_chars = len(_DEVANAGARI_RE.findall(text))
    total_alpha = sum(1 for c in text if c.isalpha())

    if total_alpha == 0:
        return "en", 1.0

    devanagari_ratio = devanagari_chars / total_alpha

    if devanagari_ratio > 0.3:
        # Significant Devanagari → pure Hindi
        return "hi", min(1.0, devanagari_ratio + 0.3)

    # ── Step 2: Check for Romanized Hindi (Hinglish) ─────────────────────
    words = re.findall(r"[a-zA-Z]+", text.lower())
    if not words:
        if devanagari_chars > 0:
            return "hi", 0.8
        return "en", 1.0

    hindi_word_count = 0
    countable_words = 0

    for w in words:
        if len(w) < 2:
            continue
        countable_words += 1
        if w in _AMBIGUOUS_WORDS:
            continue
        if w in _HINGLISH_WORDS:
            hindi_word_count += 1

    if countable_words == 0:
        return "en", 1.0

    hindi_ratio = hindi_word_count / countable_words

    # Thresholds:
    #   > 0.30 → Hinglish (mix of Hindi and English words written in Roman)
    #   > 0.15 → Weak Hinglish signal — still treat as Hinglish to be safe
    if hindi_ratio > 0.30:
        return "hi-Latn", min(1.0, hindi_ratio + 0.3)
    elif hindi_ratio > 0.15:
        return "hi-Latn", hindi_ratio + 0.2

    return "en", 1.0 - hindi_ratio


def is_hindi_or_hinglish(text: str) -> bool:
    """Quick check: is the text Hindi or Hinglish?"""
    lang, _ = detect_language(text)
    return lang in ("hi", "hi-Latn")


def get_tts_language_code(text: str) -> str:
    """
    Return the appropriate TTS language code based on input text language.

    Returns:
        "hi-IN" for Hindi / Hinglish content
        "en-IN" for text with some Indian English markers
        "en-US" default
    """
    lang, confidence = detect_language(text)
    if lang == "hi":
        return "hi-IN"
    elif lang == "hi-Latn":
        # Hinglish — use en-IN for better Indian cadence
        # (Hindi TTS might struggle with English words in Hinglish)
        return "en-IN" if confidence < 0.6 else "hi-IN"
    return "en-US"
