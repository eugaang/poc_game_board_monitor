import pandas as pd
import numpy as np
from datetime import datetime
from .config import ISSUE_CATEGORIES, NEGATIVE_CUES, POSITIVE_CUES, DEFAULT_THRESHOLD, ALPHA
from konlpy.tag import Okt

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 기본 전처리: 날짜 파싱
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # 텍스트 칼럼 결합(간단)
    df["text"] = (df.get("title", "").astype(str) + " " + df.get("content","").astype(str)).str.strip()
    return df

def simple_preprocess(text: str) -> str:
    okt = Okt()
    tokens = okt.morphs(text)  # Morpheme analysis
    cleaned = [t for t in tokens if len(t) > 1]  # Remove noise (short tokens)
    return " ".join(cleaned)

def rule_based_labels(text: str):
    text = str(text)
    found = []
    for cat, kws in ISSUE_CATEGORIES.items():
        for kw in kws:
            if kw in text:
                found.append(cat)
                break
    # 긍/부 판별(간단)
    negative = any(cue in text for cue in NEGATIVE_CUES)
    positive = any(cue in text for cue in POSITIVE_CUES)
    sentiment = "부정" if negative and not positive else ("긍정" if positive and not negative else "중립")
    return found, sentiment

def classify_posts(df: pd.DataFrame) -> pd.DataFrame:
    cats, sentiments = [], []
    for t in df["text"]:
        c, s = rule_based_labels(simple_preprocess(t))
        cats.append(c if c else ["일반"])
        sentiments.append(s)
    df = df.copy()
    df["pred_categories"] = cats
    df["pred_sentiment"] = sentiments
    df["is_issue"] = df["pred_sentiment"].eq("부정") & df["pred_categories"].apply(lambda x: x != ["일반"])
    return df

def ewma_anomaly_detection(df: pd.DataFrame, freq="15min", alpha: float = ALPHA, z_thresh: float = DEFAULT_THRESHOLD, slope_window=5, slope_thresh=0.1):
    ts = df.set_index("date").sort_index()
    counts = ts["is_issue"].resample(freq).sum().fillna(0).astype(float)
    ewma = counts.ewm(alpha=alpha).mean()
    
    # Add gradual increase detection
    ewma_diff = ewma.diff()
    rolling_slope = ewma_diff.rolling(window=slope_window).mean()
    gradual_alert = rolling_slope > slope_thresh
    
    resid = counts - ewma
    std = resid.rolling(window=max(3, int(1/alpha))).std().bfill().replace(0, resid.std() if resid.std()>0 else 1.0)
    z = resid / std
    alerts = (z.abs() >= z_thresh) | gradual_alert
    out = pd.DataFrame({"count": counts, "ewma": ewma, "resid": resid, "zscore": z, "alert": alerts, "gradual_alert": gradual_alert})
    return out