# src/utils/mood.py
"""
Simple mood detection used by the RAG pipeline.

This module provides:
- a lightweight keyword-based heuristic
- a fallback HuggingFace sentiment classifier (lazy loaded)
- safe error handling (never crashes the app)

Returns one of:
    "happy", "sad", "neutral"
"""

from typing import Optional

SAD_WORDS = {
    "sad", "down", "lonely", "tired", "anxious",
    "depressed", "upset", "heartbroken", "empty",
}

HAPPY_WORDS = {
    "happy", "excited", "joy", "joyful", "optimistic",
    "delighted", "glad",
}

_classifier = None

def _load_classifier():
    """
    Lazy-load the HuggingFace sentiment pipeline.
    This avoids slowing down initial app startup.

    Returns
    -------
    classifier : callable or None
        The HF pipeline, or None if loading fails.
    """
    global _classifier
    if _classifier is not None:
        return _classifier

    try:
        from transformers import pipeline

        _classifier = pipeline(
            "sentiment-analysis",
            model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        )
        return _classifier

    except Exception as e:
        print(f"[WARN] Sentiment classifier unavailable, using keyword mood only: {e}")
        _classifier = None
        return None

def detect_mood(text: Optional[str]) -> str:
    """
    Infer a coarse mood category from user text.

    Strategy:
    1. If input is empty → "neutral"
    2. Keyword heuristic for fast and robust detection
    3. HF sentiment classifier as fallback (if available)

    Parameters
    ----------
    text : str
        User message.

    Returns
    -------
    str
        One of {"happy", "sad", "neutral"}.
    """
    txt = (text or "").strip().lower()
    if not txt:
        return "neutral"

    if any(w in txt for w in SAD_WORDS):
        return "sad"

    if any(w in txt for w in HAPPY_WORDS):
        return "happy"

    classifier = _load_classifier()
    if classifier is None:
        return "neutral"

    try:
        result = classifier(txt)[0]  
        label = result.get("label", "").lower()

        if "positive" in label:
            return "happy"
        elif "negative" in label:
            return "sad"
        else:
            return "neutral"

    except Exception as e:
        print(f"[WARN] Mood detection failed: {e}")
        return "neutral"

