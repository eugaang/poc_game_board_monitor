# 🕹️ Game Board Monitor PoC (Sections 3.1 ~ 3.4)

이 저장소는 게임 커뮤니티 게시판 모니터링 시스템의 PoC(Proof of Concept)입니다.

## 📋 주요 기능

- **3.1 데이터 수집/전처리**: CSV 업로드 또는 샘플 데이터 사용
- **3.2 하이브리드 분류**: 카테고리(규칙 기반) + 감정(**KoELECTRA** 사전학습 모델)
- **3.3 EWMA 이상치 탐지**: 시간 버킷별 이슈 게시글 빈도 기반 경보 시스템
- **3.4 설명 가능성**: 키워드 기반 단어 중요도 시각화 (운영환경에선 LRP/IG/SHAP로 교체 가능)

## 🚀 로컬 실행

```bash
# 가상환경 생성 (선택사항)
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 앱 실행
streamlit run dashboard.py
```

브라우저가 자동으로 열리면 샘플 CSV를 선택하여 파이프라인을 실행하세요.

## ☁️ Streamlit Community Cloud 배포

### 1단계: GitHub 저장소 생성
```bash
git add .
git commit -m "Initial commit for Streamlit deployment"
git push origin main
```

### 2단계: Streamlit Community Cloud 배포
1. [share.streamlit.io](https://share.streamlit.io)에 접속
2. GitHub 계정으로 로그인
3. "New app" 클릭
4. 저장소, 브랜치, 메인 파일(`dashboard.py`) 선택
5. "Deploy!" 클릭

배포 완료! 🎉 몇 분 내에 공개 URL을 받게 됩니다.

### 배포 시 주의사항
- 샘플 데이터 파일(`data/sample_game_posts_150_realistic.csv`)이 저장소에 포함되어야 합니다
- KoELECTRA 모델 로딩에 약 3-5초 소요됩니다 (최초 실행 시)
- 무료 플랜 리소스 제한: 1GB RAM, 1 CPU (torch 사용으로 메모리 사용량 증가)
- Private 저장소도 배포 가능합니다

### KoELECTRA 모델 정보
- **모델**: monologg/koelectra-base-v3-discriminator
- **용도**: 한국어 텍스트 감정 분석 (부정/중립/긍정)
- **장점**: 사전학습된 모델로 라벨링 데이터 없이 즉시 사용 가능
- **처리 방식**: 하이브리드 (카테고리는 규칙, 감정은 KoELECTRA)

## 📁 프로젝트 구조

```
poc_game_board_monitor/
├── dashboard.py              # Streamlit 대시보드 (메인 앱)
├── requirements.txt          # Python 의존성
├── README.md                 # 이 파일
├── .streamlit/
│   └── config.toml          # Streamlit 설정
├── data/
│   └── sample_game_posts_150_realistic.csv  # 샘플 데이터
└── src/
    ├── config.py                # 설정 및 키워드 정의
    ├── pipeline.py              # 데이터 처리 파이프라인 (하이브리드 분류)
    ├── koelectra_classifier.py  # 🆕 KoELECTRA 감정 분석 모듈
    ├── explain.py               # 설명 가능성 모듈
    ├── classifier.py            # (미사용: 향후 Full Fine-tuning용)
    └── collector.py             # (미사용: 크롤링 모듈)
```

## 🛠️ 기술 스택

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib
- **ML Model**: KoELECTRA (한국어 사전학습 모델)
- **Deep Learning**: PyTorch, Transformers
- **Deployment**: Streamlit Community Cloud

## 📊 데이터 형식

CSV 파일은 다음 컬럼을 포함해야 합니다:
- `id`: 게시글 ID
- `title`: 게시글 제목
- `content`: 게시글 내용
- `date`: 작성 시간 (예: `2025-10-12 01:23`)

## 🔧 고급 설정

EWMA 파라미터 조정:
- `α (alpha)`: 0.05~0.9, 높을수록 최근 데이터에 민감
- `임계치`: 1.0~5.0, 낮을수록 더 많은 경보 발생

## 📝 라이선스

이 프로젝트는 PoC 목적으로 제작되었습니다.