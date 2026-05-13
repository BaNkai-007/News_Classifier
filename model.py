import torch
from transformers import pipeline
from datetime import datetime

# Global model
classifier = None

def load_model():
    global classifier
    if classifier is None:
        print("Loading zero-shot model... (this may take 10-20 seconds first time)")
        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",      # Best balance for class project
            # Alternative faster option (uncomment if too slow):
            # model="valhalla/distilbart-mnli-12-1",
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16 if torch.cuda.is_available() else None
        )
    return classifier


def confidence_tier(score):
    if score >= 0.65:
        return "HIGH"
    elif score >= 0.40:
        return "MEDIUM"
    else:
        return "LOW"


def run_single(text, labels, threshold=0.3, top_n=8, multi_label=True):
    if not text or not text.strip():
        return [], 0.0

    start_time = datetime.now()
    
    model = load_model()
    
    result = model(
        text,
        candidate_labels=labels,
        multi_label=multi_label,
        hypothesis_template="This example is about {}."   # Important for better scores
    )
    
    # Create pairs and sort
    pairs = list(zip(result['labels'], result['scores']))
    pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
    
    # Apply threshold
    if not multi_label:
        pairs = [pairs[0]] if pairs else []
    else:
        pairs = [p for p in pairs if p[1] >= threshold]
    
    pairs = pairs[:top_n]
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    return pairs, round(elapsed, 3)
