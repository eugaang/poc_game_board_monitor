from src.config import ISSUE_CATEGORIES, NEGATIVE_CUES, POSITIVE_CUES

def word_importance(text: str):
    tokens = text.split()          # 공백 단위 토큰화
    scores = [0.0]*len(tokens)
    for i, tok in enumerate(tokens):
        # 장애 유형 키워드 발견 시 +1.0
        for _, kws in ISSUE_CATEGORIES.items():
            if any(kw in tok for kw in kws):
                scores[i] += 1.0
        # 부정 감정 단서 발견 시 +0.5
        if any(cue in tok for cue in NEGATIVE_CUES):
            scores[i] += 0.5
        # 긍정 감정 단서 발견 시 +0.3
        if any(cue in tok for cue in POSITIVE_CUES):
            scores[i] += 0.3

    m = max(scores) if scores else 0.0
    if m > 0:
        scores = [s/m for s in scores]
    return tokens, scores