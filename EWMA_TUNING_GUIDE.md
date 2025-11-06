# EWMA 이상치 탐지 튜닝 가이드

## 📊 현재 설정

```python
# src/config.py
ALPHA = 0.3              # EWMA smoothing factor
DEFAULT_THRESHOLD = 1.5  # Z-score threshold
```

---

## 🎯 문제별 해결법

### **문제 1: 연속 버스트에서 두 번째가 감지 안 됨** ✅ 해결됨

**원인:**
- EWMA가 첫 번째 버스트에 적응
- Std가 증가하여 Z-score 낮아짐

**해결:**
```python
# Option A: Threshold 낮추기 (현재 적용)
DEFAULT_THRESHOLD = 1.5  # 2 → 1.5

# Option B: Alpha 높이기 (빠른 복귀)
ALPHA = 0.5  # 0.3 → 0.5

# Option C: 둘 다
ALPHA = 0.4
DEFAULT_THRESHOLD = 1.8
```

---

### **문제 2: Alert이 너무 많이 뜸 (False Positive)**

**원인:**
- Threshold가 너무 낮음
- 정상적인 변동도 감지

**해결:**
```python
# Threshold 높이기
DEFAULT_THRESHOLD = 2.0  # 또는 2.5
```

---

### **문제 3: 작은 버스트는 감지 안 되고 큰 것만 감지됨**

**원인:**
- Threshold가 너무 높음

**해결:**
```python
# Threshold 낮추기
DEFAULT_THRESHOLD = 1.2  # 매우 민감
```

---

## 📈 파라미터 가이드

### **ALPHA (Smoothing Factor)**

| 값 | 반응 속도 | 용도 | 장단점 |
|----|----------|------|--------|
| 0.1 | 매우 느림 | 장기 트렌드 | 안정적 / 둔감 |
| 0.3 | 보통 | 균형 | **현재 설정** |
| 0.5 | 빠름 | 빠른 변화 감지 | 민감 / 잡음 |
| 0.8 | 매우 빠름 | 즉각 반응 | 너무 민감 |

**선택 가이드:**
```
게임 장애 특성:
- 장애는 갑자기 발생 → Alpha 0.4~0.5 추천
- 서서히 악화되는 경우 → Alpha 0.2~0.3
```

### **DEFAULT_THRESHOLD (Z-score)**

| 값 | 민감도 | 예상 빈도 | 용도 |
|----|--------|----------|------|
| 1.0 | 매우 높음 | 일 1~2회 | 모든 변화 감지 |
| 1.5 | 높음 | 주 2~3회 | **현재 설정** |
| 2.0 | 보통 | 월 2~4회 | 명확한 이상만 |
| 3.0 | 낮음 | 분기 1~2회 | 심각한 이상만 |

**선택 가이드:**
```
통계적 의미:
- 1.5σ: 상위 13% (민감)
- 2.0σ: 상위 5% (보통)
- 3.0σ: 상위 0.3% (보수적)
```

---

## 🧪 추천 조합

### **조합 1: 균형잡힌 설정 (추천)**
```python
ALPHA = 0.3
DEFAULT_THRESHOLD = 1.5
```
- 대부분의 버스트 감지
- False Positive 적당

### **조합 2: 매우 민감**
```python
ALPHA = 0.5
DEFAULT_THRESHOLD = 1.2
```
- 모든 변화 감지
- 빠른 대응 필요 시

### **조합 3: 보수적**
```python
ALPHA = 0.2
DEFAULT_THRESHOLD = 2.5
```
- 심각한 장애만
- False Positive 최소화

### **조합 4: 연속 버스트 최적화 (현재)**
```python
ALPHA = 0.3
DEFAULT_THRESHOLD = 1.5
```
- 짧은 간격 버스트 모두 감지
- 첫 번째 영향 최소화

---

## 🔬 실험 방법

### **1. Dashboard에서 슬라이더 조정**

대시보드에 이미 슬라이더가 있습니다:
```
EWMA α (smoothing): 0.05 ~ 0.90
임계치 (Z-score): 1.00 ~ 5.00
```

실시간으로 테스트 가능!

### **2. 최적 값 찾기**

```python
# 실험 절차
1. 기본값으로 시작 (0.3, 1.5)
2. 실제 데이터로 테스트
3. Alert 개수 확인
4. 조정 후 재테스트
5. 만족할 때까지 반복
```

### **3. 검증 기준**

```
좋은 설정:
✅ 실제 장애는 모두 감지
✅ False Positive < 10%
✅ Alert 후 30분 내 원인 확인 가능

나쁜 설정:
❌ 장애 놓침 (False Negative)
❌ 너무 많은 Alert (피로도)
❌ 연속 버스트 구분 안 됨
```

---

## 📝 변경 로그

### 2025-11-06
- DEFAULT_THRESHOLD: 2.0 → 1.5
- 이유: 연속 버스트 감지 개선
- 결과: 두 번째 버스트도 감지됨

---

## 🎓 참고 자료

### EWMA 공식
```
EWMA_t = α × Value_t + (1-α) × EWMA_{t-1}
```

### Z-score 공식
```
Z = (Value - EWMA) / Std
Alert = |Z| >= Threshold
```

### 통계적 의미
- Z = 1.5: 86.6% 신뢰구간
- Z = 2.0: 95% 신뢰구간
- Z = 3.0: 99.7% 신뢰구간

---

## 💡 Tip

**빠른 테스트:**
1. Dashboard 실행
2. "③ EWMA 이상치 탐지" 섹션 찾기
3. 슬라이더로 실시간 조정
4. 그래프에서 Alert 점 확인
5. 만족하면 config.py에 반영

