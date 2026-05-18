import torch
from transformers import pipeline
from datetime import datetime

# Global classifier
classifier = None

# Labels
ALL_LABELS = [
    "Technology", "Business", "Politics", "Sports", "Entertainment",
    "Health", "Science", "Environment", "Education", "Finance",
    "Stock Market", "Artificial Intelligence", "Cryptocurrency",
    "War", "Economy", "Climate Change", "Startup", "Government",
    "International Relations", "Crime", "Culture", "Lifestyle"
]


def load_model():
    global classifier
    if classifier is None:
        print("🔄 Loading model...")
        classifier = pipeline(
            "multi-label-classification",
            model="facebook/bart-large-mnli",
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        )
    return classifier


def confidence_tier(score: float) -> str:
    if score >= 0.65:
        return "HIGH"
    elif score >= 0.40:
        return "MEDIUM"
    else:
        return "LOW"


def run_single(text: str, labels: list, threshold=0.3, top_n=8, multi_label=True):
    if not text or not text.strip():
        return [], 0.0

    start = datetime.now()
    
    model = load_model()
    
    result = model(
        text,
        candidate_labels=labels,
        multi_label=multi_label,
        hypothesis_template="This article is about {}.",
        truncation=True
    )
    
    # Sorted pairs
    pairs = list(zip(result['labels'], result['scores']))
    pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
    
    # Filtering by threshold
    if multi_label:
        pairs = [p for p in pairs if p[1] >= threshold]
    else:
        pairs = [pairs[0]] if pairs else []
    
    pairs = pairs[:top_n]
    
    elapsed = (datetime.now() - start).total_seconds()
    return pairs, round(elapsed, 3)
