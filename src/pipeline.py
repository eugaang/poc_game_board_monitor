import re
import pandas as pd
import numpy as np
from typing import Tuple, List
# from datetime import datetime   # 사용 안하면 제거
from .config import ISSUE_CATEGORIES, NEGATIVE_CUES, POSITIVE_CUES, DEFAULT_THRESHOLD, ALPHA

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
    norm = simple_preprocess(text)
    toks = _tokens(norm)  # 약식 토큰 리스트
    joined = " ".join(toks)

    found = []
    for cat, kws in ISSUE_CATEGORIES.items():
        # 부분일치(in) 허용: “결제에러남” 같은 케이스 대응
        if any(kw in joined for kw in kws):
            found.append(cat)

    negative = any(cue in joined for cue in NEGATIVE_CUES)
    positive = any(cue in joined for cue in POSITIVE_CUES)
    sentiment = "부정" if negative and not positive else ("긍정" if positive and not negative else "중립")
    return (found if found else ["일반"]), sentiment

def classify_posts(df: pd.DataFrame) -> pd.DataFrame:
    cats, sentiments = [], []
    for t in df["text"]:
        c, s = rule_based_labels(t)
        cats.append(c)
        sentiments.append(s)
    out = df.copy()
    out["pred_categories"] = cats
    out["pred_sentiment"] = sentiments
    out["is_issue"] = out["pred_sentiment"].eq("부정") & out["pred_categories"].apply(lambda x: x != ["일반"])
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