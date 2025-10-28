from .config import ISSUE_CATEGORIES, NEGATIVE_CUES, POSITIVE_CUES

def word_importance(text: str):
    tokens = text.split()
    scores = [0.0]*len(tokens)
    for i, tok in enumerate(tokens):
        for _, kws in ISSUE_CATEGORIES.items():
            if any(kw in tok for kw in kws):
                scores[i] += 1.0
        if any(cue in tok for cue in NEGATIVE_CUES):
            scores[i] += 0.5
        if any(cue in tok for cue in POSITIVE_CUES):
            scores[i] += 0.3
    m = max(scores) if scores else 0.0
    if m > 0:
        scores = [s/m for s in scores]
    return tokens, scores