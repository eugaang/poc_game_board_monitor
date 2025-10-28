from .config import ISSUE_CATEGORIES, NEGATIVE_CUES, POSITIVE_CUES
from collections import Counter
import random
import pandas as pd

def word_importance(text: str, categories=None):
    tokens = text.split()
    scores = [random.uniform(0,1) for _ in tokens]  # Simulate LRP scores (replace with actual LRP)
    return tokens, scores

def aggregate_keywords(df: pd.DataFrame, period_start, period_end, top_n=10):
    filtered = df[(df['date'] >= period_start) & (df['date'] <= period_end) & df['is_issue']]
    all_tokens = []
    for text in filtered['text']:
        tokens, _ = word_importance(text)
        all_tokens.extend(tokens)
    counter = Counter(all_tokens)
    return counter.most_common(top_n)