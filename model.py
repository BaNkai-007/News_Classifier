import time
import streamlit as st
import torch
from transformers import pipeline


# CONSTANTS

ALL_LABELS = [
    "Business", "Macroeconomics", "Startups", "Stock Market", "Politics", "Geopolitics",
    "War & Conflict", "Human Rights", "Technology", "Artificial Intelligence",
    "Cybersecurity", "Science", "Healthcare", "Climate Change", "Entertainment",
    "Music", "Video Games", "Sports", "Crime", "Religion", "Psychology",
    "Social Issues", "Mental Health", "Youth & Development", "Education",
    "Finance & Banking", "Environment", "Space & Astronomy", "Food & Agriculture",
    "Law & Justice", "Transportation", "Energy", "Philosophy",
]

TIER_COLORS = {"HIGH": "#16a34a", "MEDIUM": "#d97706", "LOW": "#9ca3af"}


# HELPERS

def confidence_tier(score: float) -> str:
    """Return HIGH / MEDIUM / LOW based on score."""
    if score >= 0.80:
        return "HIGH"
    elif score >= 0.60:
        return "MEDIUM"
    return "LOW"


# MODEL

@st.cache_resource(show_spinner=False)
def load_model():
    """Load and cache the zero-shot classification pipeline."""
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "zero-shot-classification",
        model="cross-encoder/nli-distilroberta-base",
        device=device,
    )


# INFERENCE

def run_single(
    text: str,
    labels: list[str],
    threshold: float,
    top_n: int,
    multi_label: bool,
) -> tuple[list[tuple[str, float]], float]:
    """
    Run zero-shot classification on a single text.

    Returns:
        pairs   – list of (label, score) tuples, sorted descending, filtered by threshold
        elapsed – inference time in seconds (rounded to 3 dp)
    """
    classifier = load_model()
    t0 = time.time()
    result = classifier(text, labels, multi_label=multi_label)
    elapsed = round(time.time() - t0, 3)

    pairs = [
        (l, s)
        for l, s in zip(result["labels"], result["scores"])
        if s >= threshold
    ]

    if not pairs:  # always return at least one result
        pairs = [(result["labels"][0], result["scores"][0])]

    pairs = sorted(pairs, key=lambda x: x[1], reverse=True)[:top_n]
    return pairs, elapsed