import re
import pandas as pd
import numpy as np
from typing import Tuple, List
# from datetime import datetime   # 사용 안하면 제거
from .config import ISSUE_CATEGORIES, NEGATIVE_CUES, POSITIVE_CUES, DEFAULT_THRESHOLD, ALPHA

# KoELECTRA 감정 분석 모델 (선택적 로딩)
USE_KOELECTRA = True  # True: KoELECTRA 사용, False: 규칙 기반 사용
_sentiment_model = None

if USE_KOELECTRA:
    try:
        from .koelectra_classifier import get_sentiment_classifier
        _sentiment_model = get_sentiment_classifier()
        print("✅ KoELECTRA 감정 분석 모듈 활성화")
    except Exception as e:
        print(f"⚠️ KoELECTRA 로딩 실패, 규칙 기반으로 대체: {e}")
        USE_KOELECTRA = False
        _sentiment_model = None

# --- 작은 유틸: 매우 가벼운 정규화(한글/영문/숫자만 남기고 소문자) ---
_word_re = re.compile(r"[가-힣a-z0-9]+")

def _light_normalize(text: str) -> str:
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text.lower().strip()
    # “안돼요/안 돼요” 같은 띄어쓰기 변이 최소화(선택)
    text = text.replace("안 돼", "안돼")
    return text

def _tokens(text: str) -> List[str]:
    return _word_re.findall(_light_normalize(text))

# -----------------------------
# 3.1 Load & Preprocess
# -----------------------------
def load_data(path_or_buf) -> pd.DataFrame:
    df = pd.read_csv(path_or_buf)
    # 안전하게 열 보장
    for col in ("title", "content"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    # 날짜 파싱
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        # NaT 제거 (resample 안전성)
        df = df.dropna(subset=["date"])
        # 필요시 타임존 고정 (옵션)
        # if df["date"].dt.tz is None:
        #     df["date"] = df["date"].dt.tz_localize("Asia/Seoul")
    else:
        raise ValueError("CSV에는 'date' 컬럼이 필요합니다 (예: 2025-10-12 01:23).")

    # 텍스트 결합
    df["text"] = (df["title"].astype(str) + " " + df["content"].astype(str)).str.strip()
    return df.reset_index(drop=True)

def simple_preprocess(text: str) -> str:
    # 아주 가벼운 공백 정리 + 소문자화
    return " ".join(_light_normalize(text).split())

# -----------------------------
# 3.2 Rule-based Classification
# -----------------------------
def rule_based_labels(text: str) -> Tuple[List[str], str]:
    """
    하이브리드 분류: 카테고리(규칙) + 감정(KoELECTRA)
    
    Args:
        text: 분석할 텍스트
    
    Returns:
        (카테고리 리스트, 감정)
        예: (["로그인", "결제"], "부정")
    """
    norm = simple_preprocess(text)
    toks = _tokens(norm)  # 약식 토큰 리스트
    joined = " ".join(toks)

    # 1️⃣ 카테고리 분류: 규칙 기반 (키워드 매칭)
    found = []
    for cat, kws in ISSUE_CATEGORIES.items():
        # 부분일치(in) 허용: "결제에러남" 같은 케이스 대응
        if any(kw in joined for kw in kws):
            found.append(cat)
    
    categories = found if found else ["일반"]
    
    # 2️⃣ 감정 분석: KoELECTRA 또는 규칙 기반
    if USE_KOELECTRA and _sentiment_model:
        # KoELECTRA 모델 사용
        sentiment = _sentiment_model.predict_sentiment(text)
    else:
        # 규칙 기반 (폴백)
        negative = any(cue in joined for cue in NEGATIVE_CUES)
        positive = any(cue in joined for cue in POSITIVE_CUES)
        sentiment = "부정" if negative and not positive else ("긍정" if positive and not negative else "중립")
    
    return categories, sentiment

def classify_posts(df: pd.DataFrame) -> pd.DataFrame:
    """
    게시글 분류 (하이브리드 방식)
    
    - 카테고리: 규칙 기반 키워드 매칭
    - 감정: KoELECTRA 사전학습 모델
    - 이슈: 부정 감정 + 특정 카테고리
    
    Args:
        df: 게시글 데이터프레임 (text 컬럼 필요)
    
    Returns:
        분류 결과가 추가된 데이터프레임
        - pred_categories: 카테고리 리스트
        - pred_sentiment: 감정 (부정/중립/긍정)
        - is_issue: 이슈 여부 (True/False)
        - classification_method: 사용된 분류 방법
    """
    cats, sentiments = [], []
    
    # 각 게시글 분류
    for t in df["text"]:
        c, s = rule_based_labels(t)
        cats.append(c)
        sentiments.append(s)
    
    out = df.copy()
    out["pred_categories"] = cats
    out["pred_sentiment"] = sentiments
    
    # 이슈 판단: 부정 감정 + 특정 카테고리
    out["is_issue"] = out["pred_sentiment"].eq("부정") & out["pred_categories"].apply(lambda x: x != ["일반"])
    
    # 분류 방식 표시
    classification_method = "Hybrid: Category(Rule-based) + Sentiment(KoELECTRA)" if USE_KOELECTRA else "Rule-based"
    out["classification_method"] = classification_method
    
    return out

# -----------------------------
# 3.3 EWMA-based Anomaly Detection
# -----------------------------
def ewma_anomaly_detection(
    df: pd.DataFrame,
    freq: str = "15min",
    alpha: float = ALPHA,
    z_thresh: float = DEFAULT_THRESHOLD
) -> pd.DataFrame:
    if "date" not in df.columns:
        raise ValueError("df에 'date' 컬럼이 필요합니다.")
    if "is_issue" not in df.columns:
        raise ValueError("df에 'is_issue' 컬럼이 필요합니다. 먼저 classify_posts()를 호출하세요.")

    ts = df.set_index("date").sort_index()
    # 빈 프레임 방어
    if ts.empty:
        return pd.DataFrame(columns=["count", "ewma", "resid", "zscore", "alert"])

    counts = ts["is_issue"].resample(freq).sum().fillna(0).astype(float)
    ewma = counts.ewm(alpha=alpha, adjust=False).mean()
    resid = counts - ewma

    # 표준편차: 최소 창 크기와 하한(ε) 설정
    window = max(3, int(round(1/alpha)))  # alpha 작을수록 창 크게
    std = resid.rolling(window=window, min_periods=max(3, window//2)).std()
    # 1차 보정: NaN 채우기
    global_std = resid.std(ddof=0)
    if not np.isfinite(global_std) or global_std == 0:
        global_std = 1.0
    std = std.fillna(global_std)
    # 2차 보정: 너무 작은 분모 방어
    std = std.clip(lower=1e-6)

    z = resid / std
    alerts = z.abs() >= z_thresh

    out = pd.DataFrame({
        "count": counts,
        "ewma": ewma,
        "resid": resid,
        "zscore": z,
        "alert": alerts
    })
    return out