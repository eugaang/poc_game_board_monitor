import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Streamlit Cloud용 백엔드 설정
import matplotlib.pyplot as plt
from src.pipeline import load_data, classify_posts, ewma_anomaly_detection
from src.explain import word_importance
from src.config import DEFAULT_THRESHOLD, ALPHA

# -----------------------------------
# Page setup
# -----------------------------------
st.set_page_config(page_title="Game Board Monitor PoC", layout="wide")
st.title("🕹️ Game Board Monitor PoC (3.1 ~ 3.4)")

st.markdown("""
본 데모는 다음 파이프라인을 구현합니다.

1) **데이터 수집/전처리**: CSV 업로드 또는 샘플 사용  
2) **🤖 하이브리드 분류**: 카테고리(규칙 기반) + 감정(KoELECTRA)  
3) **설명 가능성(간이)**: 키워드 가중치 기반 단어 중요도 하이라이트  
4) **EWMA 이상치 탐지**: 시간 버킷별 이슈 빈도 기반 경보  
""")

# -----------------------------------
# 3.1 데이터 로드 & 전처리
# -----------------------------------
SAMPLE_PATH = "data/sample_game_posts_100_realistic.csv"
st.subheader("① 데이터 로드 및 전처리")
st.caption(f"샘플 데이터 사용 중: {SAMPLE_PATH}")
df = load_data(SAMPLE_PATH)


# -----------------------------------
# 3.2 규칙 기반 분류
# -----------------------------------
st.subheader("② 분류 실행 (하이브리드: 규칙 + KoELECTRA)")

st.info("""
**🌟 KoELECTRA + 규칙 기반 하이브리드 분류 (Fine-tuned)**

- **카테고리 분류** (로그인, 결제, 렉 등): 규칙 기반 키워드 매칭  
- **감정 분석** (부정/중립/긍정): **Fine-tuned KoELECTRA** 모델 사용  
- **이슈 판단**: 부정 감정 + 특정 카테고리 → 이슈

📊 **게임 커뮤니티 데이터로 Fine-tuning 완료** (Validation 정확도: 100%)  
🎯 프로젝트 사전보고서에 명시된 **KoELECTRA를 실제 적용 및 최적화**한 구현입니다.
""")

# classify_posts()는 각 게시글(text)에 대해 하이브리드 분류를 수행합니다.
# 
# 🔹 카테고리 분류: config.py의 ISSUE_CATEGORIES 키워드로 규칙 기반 매칭
# 🔹 감정 분석: KoELECTRA 모델을 사용하여 부정/중립/긍정 예측
#
# 함수 실행 결과에는 다음 3개의 주요 컬럼이 추가됩니다:
#
#   1️⃣ pred_categories : 탐지된 문제 유형(category) 리스트
#        예) ["로그인"], ["결제"], ["일반"]
#        - "일반"은 장애 관련 키워드가 발견되지 않은 경우
#
#   2️⃣ pred_sentiment : 문장의 감정 경향 (KoELECTRA 예측)
#        예) "부정", "중립", "긍정"
#        - KoELECTRA 모델이 텍스트를 분석하여 감정을 예측
#        - 사전학습된 한국어 ELECTRA 모델 활용
#
#   3️⃣ is_issue : 실제 '이슈 게시글'로 간주되는지 여부 (True/False)
#        - 조건: pred_sentiment == "부정" 이면서 pred_categories != ["일반"]
#        - 즉, 부정적인 감정을 가진 장애 관련 게시글만 True로 표시
#
# 이렇게 분류된 데이터는 이후 EWMA 이상치 탐지(3.3) 단계의 입력으로 사용됩니다.
# 부정 게시글이 시간대별로 얼마나 발생했는지를 기반으로 이상 패턴을 감지하게 됩니다.


pred_df = classify_posts(df)

# 방어: is_issue 보장 (혹시 다른 버전의 classify_posts를 쓰더라도 안전하게)
if "is_issue" not in pred_df.columns:
    if {"pred_categories", "pred_sentiment"}.issubset(pred_df.columns):
        pred_df["is_issue"] = pred_df["pred_sentiment"].eq("부정") & \
                              pred_df["pred_categories"].apply(lambda x: x != ["일반"])
    else:
        st.error("분류 결과에 필요한 열이 없습니다. 'classify_posts()' 구현을 확인하세요.")
        st.stop()

# 핵심 단어 추출 함수
def extract_key_words(text):
    """텍스트에서 핵심 단어 추출 (간략 표시용)"""
    from src.explain import word_importance
    tokens, scores = word_importance(text)
    
    # 점수가 높은 상위 3개 단어만
    word_scores = [(t, s) for t, s in zip(tokens, scores) if s > 0]
    word_scores.sort(key=lambda x: x[1], reverse=True)
    
    if not word_scores:
        return "-"
    
    # 상위 3개만 표시
    top_words = word_scores[:3]
    result = ", ".join([f"{w}({s:.1f})" for w, s in top_words])
    return result

# 핵심 단어 컬럼 추가
pred_df["핵심_단어"] = pred_df["text"].apply(extract_key_words)

# 결과 미리보기 (핵심 단어 포함)
st.dataframe(
    pred_df[["id", "title", "content", "pred_categories", "pred_sentiment", "핵심_단어", "is_issue", "date"]],
    use_container_width=True
)

# -----------------------------------
# 💾 분류 결과 다운로드 버튼 추가
# -----------------------------------
import io

# CSV 버전 (선택)
st.download_button(
    label="② 분류 결과 CSV 다운로드",
    data=pred_df.to_csv(index=False).encode("utf-8-sig"),
    file_name="classification_results.csv",
    mime="text/csv",
)

# -----------------------------------
# 3.4 설명 가능성 (키워드 가중치 기반)
# -----------------------------------
st.subheader("③ 분류 근거 확인 (설명 가능성)")

st.markdown("""
**💡 왜 이 게시글이 "부정"으로 분류되었나요?**

아래에서 행 번호를 선택하면, 어떤 단어가 분류에 영향을 주었는지 확인할 수 있습니다.
""")

# 이 단계는 각 게시글의 문장 내에서 어떤 단어가 '이슈 탐지'에 영향을 주었는지를
# 시각적으로 표시하는 간단한 Explainability(설명 가능성) 모듈입니다.
#
# 내부적으로 src/explain.py의 `word_importance()` 함수가 호출되며,
# 그 함수는 다음 규칙으로 각 단어에 점수를 부여합니다:
#
#   1) ISSUE_CATEGORIES에 포함된 장애 키워드 발견 시 +1.0
#   2) NEGATIVE_CUES(부정 단서)가 포함되면 +0.5
#   3) POSITIVE_CUES(긍정 단서)가 포함되면 +0.3
#
# 이후 점수를 [0, 1] 범위로 정규화(normalization)하여
# Streamlit 화면에서 단어별로 빨간 음영(red heatmap intensity)으로 표현합니다.
#
#  • 진하게 표시된 단어일수록 모델(혹은 규칙)이 '이 문장은 문제 상황이다'라고
#    판단하는 데 기여한 비중이 큼
#  • 이 PoC에서는 간단한 규칙 기반 방식이지만,
#    실제 운영환경에서는 LRP(레이어별 관련성 전파, Layer-wise Relevance Propagation),
#    Integrated Gradients 등의 딥러닝 기반 방법으로 대체 가능


row_ix = st.number_input("설명을 볼 행 index 선택", min_value=0, max_value=len(pred_df)-1, value=0, step=1)
text = pred_df.iloc[int(row_ix)]["text"]
tokens, scores = word_importance(text)

def colorize_enhanced(tokens, scores):
    """개선된 시각화: 색상 + 레이블 + 점수"""
    from src.config import ISSUE_CATEGORIES, NEGATIVE_CUES, POSITIVE_CUES
    
    html = []
    for t, s in zip(tokens, scores):
        # 단어 종류 판단
        word_type = ""
        emoji = ""
        color = "red"
        
        # 카테고리 키워드 체크
        is_category = False
        for cat, keywords in ISSUE_CATEGORIES.items():
            if any(kw in t for kw in keywords):
                word_type = f"[{cat}]"
                emoji = "🎯"
                color = "red"
                is_category = True
                break
        
        # 부정어 체크
        if not is_category and any(cue in t for cue in NEGATIVE_CUES):
            word_type = "[부정]"
            emoji = "⚠️"
            color = "orange"
        
        # 긍정어 체크
        elif not is_category and any(cue in t for cue in POSITIVE_CUES):
            word_type = "[긍정]"
            emoji = "✅"
            color = "green"
        
        # HTML 생성
        if s > 0:  # 점수가 있는 단어만 강조
            html.append(
                f"<span style='background-color: rgba({'255,0,0' if color=='red' else '255,165,0' if color=='orange' else '0,200,0'},{s}); "
                f"padding:4px 8px; border-radius:5px; margin:2px; display:inline-block; "
                f"border: 2px solid {color};'>"
                f"{emoji} <strong>{t}</strong> "
                f"<small style='opacity:0.8;'>{word_type} {s:.1f}</small>"
                f"</span>"
            )
        else:
            html.append(f"<span style='margin:2px;'>{t}</span>")
    
    return " ".join(html)

def create_word_table(tokens, scores):
    """단어 분석 테이블 생성"""
    from src.config import ISSUE_CATEGORIES, NEGATIVE_CUES, POSITIVE_CUES
    import pandas as pd
    
    data = []
    for t, s in zip(tokens, scores):
        if s > 0:  # 점수가 있는 단어만
            word_type = "중립"
            reason = "-"
            
            # 카테고리 체크
            for cat, keywords in ISSUE_CATEGORIES.items():
                if any(kw in t for kw in keywords):
                    word_type = f"카테고리({cat})"
                    reason = f"'{cat}' 문제 키워드"
                    break
            
            # 부정어 체크
            if word_type == "중립" and any(cue in t for cue in NEGATIVE_CUES):
                word_type = "부정어"
                reason = "부정적 표현"
            
            # 긍정어 체크
            elif word_type == "중립" and any(cue in t for cue in POSITIVE_CUES):
                word_type = "긍정어"
                reason = "긍정적 표현"
            
            data.append({
                "단어": t,
                "종류": word_type,
                "중요도": f"{s:.2f}",
                "이유": reason
            })
    
    if data:
        return pd.DataFrame(data)
    return None

st.markdown("**원문 텍스트**")
st.write(text)

# 📘 "간이 LRP 대체"란?
#
#  • LRP(Layer-wise Relevance Propagation)는 딥러닝 모델(BERT, KoELECTRA 등)의
#    예측 결과가 입력의 어떤 단어(또는 특징)에서 기인했는지를 수학적으로 역추적하는
#    설명 가능한 AI(Explainable AI) 기법이다.
#
#  • 본 PoC는 규칙 기반 분류 방식으로, 모델의 가중치(weight)가 존재하지 않기 때문에
#    LRP 같은 수학적 역전파 방식은 사용할 수 없다.
#
#  • 대신 단어별로 간단한 가중치 규칙을 적용해 중요도를 계산하고,
#    이를 색상 강도로 시각화하여 "딥러닝의 LRP 결과처럼 보이는"
#    간이 형태의 설명 모듈로 구현하였다.
#
#  • 즉, "간이 LRP 대체"란 복잡한 딥러닝 설명 기법을
#    규칙 기반 점수 계산으로 단순 대체한 버전임.
#    (목적: PoC 단계에서 설명 가능성(Explainability) 개념을 시각적으로 확인)

st.markdown("**🎨 단어 중요도 시각화 (개선)**", unsafe_allow_html=True)
st.markdown(colorize_enhanced(tokens, scores), unsafe_allow_html=True)

# 상세 분석 테이블 추가
st.markdown("**📊 상세 단어 분석**")
word_table = create_word_table(tokens, scores)
if word_table is not None:
    st.dataframe(word_table, use_container_width=True)
else:
    st.info("중요 단어가 감지되지 않았습니다.")

# -----------------------------------
# 3.3 EWMA 이상치 탐지
# -----------------------------------
st.subheader("④ EWMA 이상치 탐지")

# EWMA(Exponentially Weighted Moving Average)는 시간 흐름에 따라
# 최신 데이터에 더 큰 가중치를 주는 '지수가중 이동평균'입니다.
# α(alpha)는 EWMA의 "민감도"를 조절하는 매개변수입니다.
#
#   • α ↓ (예: 0.1~0.3) → 오래된 데이터도 반영 → 곡선이 부드럽고 안정적
#                       → 변화에 둔감하지만 노이즈에 강함
#   • α ↑ (예: 0.5~0.9) → 최신 데이터 위주 반영 → 곡선이 민감하게 변동
#                       → 변화에 즉각 반응하지만 거짓 경보 많음
#
# z-threshold(임계치)는 경보 발생 기준이 되는 z-score(표준편차 배수) 값입니다.
#   • 임계치 ↓ (예: 1.5) → 작은 변화도 경보로 감지 (민감)
#   • 임계치 ↑ (예: 3.0) → 큰 변화만 경보로 감지 (보수적)
#
# 따라서 두 값을 조합하여 다음과 같은 동작 특성을 조절할 수 있습니다:
#   α=0.4~0.6, 임계치=2.0 → 권장값 (적당한 민감도와 안정성)
#   α 높음 + 임계치 낮음  → 매우 빠르지만 거짓 경보 많음
#   α 낮음 + 임계치 높음  → 느리지만 신뢰도 높은 경보만 탐지

freq = st.selectbox("집계 주기", ["5min", "10min", "15min", "30min", "1H"], index=2)
alpha = st.slider("EWMA α (smoothing)", 0.05, 0.9, ALPHA, 0.05)
zth = st.slider("임계치 (|z| ≥ 임계치 시 경보)", 1.0, 5.0, DEFAULT_THRESHOLD, 0.5)

try:
    an = ewma_anomaly_detection(pred_df, freq=freq, alpha=alpha, z_thresh=zth)
except Exception as e:
    st.error(f"이상치 탐지 중 오류: {e}")
    st.stop()

if an.empty:
    st.warning("집계 결과가 비어 있습니다. 기간/버킷 설정 또는 입력 데이터를 확인하세요.")
else:
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(an.index, an["count"], label="count")
    ax.plot(an.index, an["ewma"], label="ewma")

    alert_mask = an["alert"].fillna(False)
    ax.scatter(an.index[alert_mask], an.loc[alert_mask, "count"], marker="o", s=60, label="ALERT")

    ax.set_title("Issue count vs EWMA")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

    # st.markdown("**최근 20개 버킷**")
    # st.dataframe(an.tail(20))
    st.markdown("**전체 버킷 (CSV 전체 기준)**")
    st.dataframe(an, use_container_width=True)

    alerts_only = an[an["alert"].fillna(False)].copy()
    st.markdown("**감지된 ALERT 목록**")
    if alerts_only.empty:
        st.info("감지된 ALERT가 없습니다.")
    else:
        alerts_view = alerts_only[["count", "ewma", "zscore"]].assign(ts=alerts_only.index)
        st.dataframe(alerts_view.set_index("ts"))
        st.download_button(
            "ALERT CSV 다운로드",
            alerts_only.to_csv().encode("utf-8-sig"),
            file_name="alerts.csv",
            mime="text/csv",
        )

st.success("✅ PoC 완료: 3.1~3.4 전체 기능 체인을 한 화면에서 확인할 수 있습니다.")
