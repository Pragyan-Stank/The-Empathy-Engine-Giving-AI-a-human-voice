"""
Granular emotion refinement — rule-based sub-emotion detection.

Runs AFTER the primary transformer/VADER classification.
Inspects surface signals in the raw text (punctuation, caps, keywords)
to narrow broad labels (joy, anger, sadness, fear) into more specific states.

Order matters: first matching rule wins.
"""
import re
from typing import Tuple


# ── Detection Rules ────────────────────────────────────────────────────────────
# Each rule: (granular_label, condition_fn(text, base_emotion) → bool)
GRANULAR_RULES: list = [
    # ── Joy → Excitement (intense positivity signals) ──────────────────────
    (
        "excitement",
        lambda t, e: e in ("joy", "surprise")
        and bool(re.search(r"[!]{2,}|[A-Z]{4,}|\bOMG\b|\bWOW\b|\bAMAZING\b|\bINCREDIBLE\b", t)),
    ),
    # ── Joy → Contentment (calm, low-key positive) ─────────────────────────
    (
        "contentment",
        lambda t, e: e == "joy"
        and not re.search(r"[!]", t)
        and not re.search(r"[A-Z]{3,}", t),
    ),
    # ── Anger → Rage (extreme intensity: all-caps, extreme words) ──────────
    (
        "rage",
        lambda t, e: e == "anger"
        and bool(re.search(
            r"[A-Z]{5,}|HATE|FURIOUS|LIVID|OUTRAGED|DESTROY|KILL|WORST",
            t,
        )),
    ),
    # ── Anger → Frustration (milder, ongoing irritation) ───────────────────
    (
        "frustration",
        lambda t, e: e == "anger"
        and bool(re.search(
            r"frustrat|annoyed|ugh|seriously|anymore|can.t|kept|again|always does",
            t.lower(),
        )),
    ),
    # ── Sadness → Grief (loss, longing, finality) ──────────────────────────
    (
        "grief",
        lambda t, e: e == "sadness"
        and bool(re.search(
            r"\bmiss\b|\bgone\b|\blost\b|\bmourn|\bgoodbye\b|\bnever again\b|\bno more\b",
            t.lower(),
        )),
    ),
    # ── Fear → Anxiety (worry, nervous anticipation) ───────────────────────
    (
        "anxiety",
        lambda t, e: e == "fear"
        and bool(re.search(
            r"worried|anxious|nervous|stressed|overwhelm|panic|can.t stop",
            t.lower(),
        )),
    ),
]


def refine_emotion(text: str, base_emotion: str) -> str:
    """
    Apply granular refinement rules.
    Returns a more specific emotion label if any rule matches, else base_emotion.
    """
    for refined_label, condition in GRANULAR_RULES:
        try:
            if condition(text, base_emotion):
                return refined_label
        except Exception:
            continue
    return base_emotion
