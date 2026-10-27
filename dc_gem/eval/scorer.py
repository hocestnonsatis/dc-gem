"""Scoring: classify memOFF outputs and compute exact match."""

from __future__ import annotations
import re


def classify_memoff(output: str, expected: str) -> str:
    """Classify memOFF output: correct | wrong_answer | refusal | empty | leak."""
    o = (output or "").strip()
    if not o:
        return "empty"
    # leak: contains expected value but in a way that suggests training data
    # (here: simply check if expected appears)
    if expected and expected in o:
        # could be correct or leak; treat as correct only if exact-ish
        if o.strip() == expected.strip():
            return "correct"
        return "leak"
    low = o.lower()
    refusal_markers = ["i don't know", "i cannot", "i can't", "not sure", "unable"]
    if any(m in low for m in refusal_markers):
        return "refusal"
    # wrong answer: contains a different code-like token
    if re.search(r"\d{3,6}", o):
        return "wrong_answer"
    return "wrong_answer"


def exact_match(generated: str, expected: str) -> bool:
    g = (generated or "").strip()
    e = (expected or "").strip()
    return g == e or g.startswith(e)
