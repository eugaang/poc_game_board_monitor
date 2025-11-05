"""
KoELECTRA 기반 감정 분석 모듈
사전학습된 한국어 ELECTRA 모델을 사용하여 텍스트의 감정(부정/중립/긍정)을 분류합니다.
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

class KoElectraSentimentClassifier:
    """
    사전학습된 KoELECTRA를 사용한 감정 분석 분류기
    
    - 모델: monologg/koelectra-base-v3-discriminator (사전학습 모델)
    - 출력: 부정(0), 중립(1), 긍정(2)
    - 용도: 게임 커뮤니티 게시글의 감정 분석
    """
    
    def __init__(self, model_name="./koelectra-game-sentiment", use_finetuned=True):
        """
        KoELECTRA 모델 초기화
        
        Args:
            model_name: 모델 경로 (로컬 또는 Hugging Face)
            use_finetuned: Fine-tuned 모델 사용 여부
        """
        # Fine-tuned 모델 사용 여부 결정
        if use_finetuned:
            # Fine-tuned 모델 경로 확인
            import os
            if os.path.exists("./koelectra-game-sentiment"):
                model_name = "./koelectra-game-sentiment"
                print(f"🌟 Fine-tuned KoELECTRA 모델 로딩 중: {model_name}")
            else:
                print(f"⚠️ Fine-tuned 모델을 찾을 수 없습니다. 사전학습 모델 사용")
                model_name = "monologg/koelectra-base-v3-discriminator"
                print(f"🤖 사전학습 KoELECTRA 모델 로딩 중: {model_name}")
        else:
            print(f"🤖 사전학습 KoELECTRA 모델 로딩 중: {model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.eval()  # 평가 모드로 설정
            
            # GPU 사용 가능하면 GPU로, 아니면 CPU
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            
            print(f"✅ KoELECTRA 모델 로딩 완료 (device: {self.device})")
            
        except Exception as e:
            print(f"❌ KoELECTRA 로딩 실패: {e}")
            raise
    
    def predict_sentiment(self, text: str) -> str:
        """
        텍스트의 감정을 예측합니다.
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            "부정", "중립", "긍정" 중 하나
        """
        if not text or not text.strip():
            return "중립"
        
        try:
            # 토큰화
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # 예측 (gradient 계산 불필요)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                pred = torch.argmax(logits, dim=1).item()
            
            # 레이블 매핑
            # 0: 부정, 1: 중립, 2: 긍정
            labels = ["부정", "중립", "긍정"]
            return labels[pred]
            
        except Exception as e:
            print(f"⚠️ 감정 예측 중 오류 ({text[:30]}...): {e}")
            return "중립"  # 오류 시 중립으로 처리
    
    def predict_batch(self, texts: list) -> list:
        """
        여러 텍스트를 한 번에 예측 (배치 처리로 속도 향상)
        
        Args:
            texts: 텍스트 리스트
            
        Returns:
            감정 리스트 ["부정", "중립", "긍정", ...]
        """
        if not texts:
            return []
        
        try:
            # 배치 토큰화
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # 배치 예측
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            # 레이블 매핑
            labels = ["부정", "중립", "긍정"]
            return [labels[p] for p in preds]
            
        except Exception as e:
            print(f"⚠️ 배치 예측 중 오류: {e}")
            # 오류 시 개별 예측으로 대체
            return [self.predict_sentiment(text) for text in texts]


# 전역 모델 인스턴스 (싱글톤 패턴)
_global_model = None

def get_sentiment_classifier():
    """
    전역 KoELECTRA 모델 인스턴스를 반환합니다.
    모델을 한 번만 로딩하여 메모리 절약
    """
    global _global_model
    if _global_model is None:
        _global_model = KoElectraSentimentClassifier()
    return _global_model

