import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.pipeline import load_data, classify_posts, ewma_anomaly_detection
from src.explain import word_importance

st.set_page_config(page_title="Game Board Monitor PoC", layout="wide")
st.title("🕹️ Game Board Monitor PoC (3.1 ~ 3.4)")

st.markdown("""
본 데모는 다음 파이프라인을 구현합니다.
1) CSV 업로드(데이터 수집/전처리)
2) 규칙 기반 분류(운영환경에서는 KoELECTRA로 교체)
3) EWMA 이상치 탐지
4) 키워드 기반 설명(운영환경에서는 LRP/IG 대체)
""")

uploaded = st.file_uploader("CSV 파일을 업로드하거나 샘플을 사용하세요.", type=["csv"])
use_sample = st.checkbox("샘플 데이터 사용 (data/sample_game_posts.csv)", value=True)

if use_sample:
    path = "data/sample_game_posts.csv"
    df = load_data(path)
elif uploaded:
    df = pd.read_csv(uploaded)
    df = load_data(uploaded)
else:
    st.info("CSV를 업로드하거나 '샘플 데이터 사용'을 체크하세요.")
    st.stop()

st.subheader("① 데이터 미리보기")
st.dataframe(df.head(10))

st.subheader("② 분류 실행 (규칙 기반)")
pred_df = classify_posts(df)
st.dataframe(pred_df[["id","title","pred_categories","pred_sentiment","is_issue","date"]].head(20))

st.subheader("③ EWMA 이상치 탐지")
freq = st.selectbox("집계 주기", ["5min","10min","15min","30min","1H"], index=2)
alpha = st.slider("EWMA α (smoothing)", 0.05, 0.9, 0.3, 0.05)
zth = st.slider("임계치 (|z| ≥ 임계치 시 경보)", 1.0, 5.0, 2.0, 0.5)
an = ewma_anomaly_detection(pred_df, freq=freq, alpha=alpha, z_thresh=zth)

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(an.index, an["count"], label="count")
ax.plot(an.index, an["ewma"], label="ewma")
ax.scatter(an.index[an["alert"]], an["count"][an["alert"]], marker="o", s=60, label="ALERT")
ax.set_title("Issue count vs EWMA")
ax.legend()
st.pyplot(fig)

st.write(an.tail(20))

st.subheader("④ 설명(키워드 가중치 기반)")
row_ix = st.number_input("설명을 볼 행 index 선택", min_value=0, max_value=len(pred_df)-1, value=0, step=1)
text = pred_df.iloc[int(row_ix)]["text"]
tokens, scores = word_importance(text)

def colorize(tokens, scores):
    html = []
    for t, s in zip(tokens, scores):
        html.append(f'<span style="background-color: rgba(255,0,0,{s}); padding:2px; border-radius:3px; margin:1px;">{t}</span>')
    return " ".join(html)

st.markdown("**원문 텍스트**")
st.write(text)
st.markdown("**단어 중요도 히트맵 (간이 LRP 대체)**", unsafe_allow_html=True)
st.markdown(colorize(tokens, scores), unsafe_allow_html=True)

st.success("PoC 완료: 3.1~3.4의 최소 기능 체인을 한 화면에서 확인할 수 있습니다.")