# Game Board Monitor PoC (Sections 3.1 ~ 3.4)

이 저장소는 김동길 프로젝트의 3장(3.1~3.4)을 **PoC 형태**로 구현한 최소 동작 예시입니다.
- 3.1 데이터 수집/전처리: CSV 업로드(또는 data/sample_game_posts.csv) 후 간단 전처리
- 3.2 분류(대체): 규칙 기반 이진/다중 속성 분류(운영 환경에선 KoELECTRA로 교체)
- 3.3 이상치(EWMA) 탐지: 시간 버킷별 장애글 빈도에 EWMA 적용 + 임계치 경보
- 3.4 해석(대체): 키워드 기반 토큰 중요도 시각화(운영환경에선 LRP/IG/SHAP로 교체)

## 빠른 시작
```bash
pip install -r requirements.txt
streamlit run dashboard.py
```
브라우저가 열리면 샘플 CSV를 선택하여 파이프라인을 실행하세요.