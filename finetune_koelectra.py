"""
KoELECTRA Fine-tuning Script
게임 커뮤니티 감정 분석을 위한 KoELECTRA 모델 Fine-tuning

사용법:
    python finetune_koelectra.py
"""

import os
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 설정
# ============================================

CONFIG = {
    "model_name": "monologg/koelectra-base-v3-discriminator",
    "train_file": "data/train_koelectra_120.csv",
    "val_file": "data/val_koelectra_30.csv",
    "output_dir": "./koelectra-finetuned",
    "max_length": 512,
    "num_labels": 3,  # 0=부정, 1=중립, 2=긍정
    
    # 학습 설정
    "num_train_epochs": 5,  # 전체 데이터를 5번 학습
    "batch_size": 8,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    
    # 기타
    "seed": 42,
    "use_gpu": torch.cuda.is_available()
}

print("="*60)
print("🤖 KoELECTRA Fine-tuning")
print("="*60)
print(f"📊 Train data: {CONFIG['train_file']}")
print(f"📊 Val data: {CONFIG['val_file']}")
print(f"🔧 Model: {CONFIG['model_name']}")
print(f"💻 Device: {'GPU' if CONFIG['use_gpu'] else 'CPU'}")
print(f"📝 Epochs: {CONFIG['num_train_epochs']}")
print(f"📦 Batch size: {CONFIG['batch_size']}")
print("="*60)

# ============================================
# 1. 데이터 로드 및 전처리
# ============================================

print("\n📂 데이터 로딩 중...")

# CSV 읽기
train_df = pd.read_csv(CONFIG["train_file"])
val_df = pd.read_csv(CONFIG["val_file"])

# 텍스트 결합 (title + content)
train_df["text"] = train_df["title"].fillna("") + " " + train_df["content"].fillna("")
val_df["text"] = val_df["title"].fillna("") + " " + val_df["content"].fillna("")

# 레이블 확인
print(f"✅ Train: {len(train_df)}개")
print(f"✅ Val: {len(val_df)}개")
print(f"\n📊 레이블 분포 (Train):")
print(train_df["sentiment_label"].value_counts().sort_index())
print(f"\n📊 레이블 분포 (Val):")
print(val_df["sentiment_label"].value_counts().sort_index())

# Dataset 변환
train_dataset = Dataset.from_dict({
    "text": train_df["text"].tolist(),
    "label": train_df["sentiment_label"].tolist()
})

val_dataset = Dataset.from_dict({
    "text": val_df["text"].tolist(),
    "label": val_df["sentiment_label"].tolist()
})

# ============================================
# 2. 토크나이저 로드 및 토큰화
# ============================================

print("\n🔤 토크나이저 로딩 중...")

tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

def preprocess_function(examples):
    """텍스트를 토큰화"""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=CONFIG["max_length"]
    )

print("🔤 토큰화 진행 중...")
train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)

print("✅ 토큰화 완료!")

# ============================================
# 3. 모델 로드
# ============================================

print(f"\n🤖 모델 로딩 중: {CONFIG['model_name']}")

model = AutoModelForSequenceClassification.from_pretrained(
    CONFIG["model_name"],
    num_labels=CONFIG["num_labels"],
    problem_type="single_label_classification"
)

print("✅ 모델 로드 완료!")

# ============================================
# 4. 평가 함수 정의
# ============================================

def compute_metrics(pred):
    """평가 지표 계산"""
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    
    # 정확도
    acc = accuracy_score(labels, preds)
    
    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )
    
    # 클래스별 정확도
    conf_matrix = confusion_matrix(labels, preds)
    class_acc = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "class_0_acc": class_acc[0] if len(class_acc) > 0 else 0,
        "class_1_acc": class_acc[1] if len(class_acc) > 1 else 0,
        "class_2_acc": class_acc[2] if len(class_acc) > 2 else 0,
    }

# ============================================
# 5. 학습 설정
# ============================================

print("\n⚙️ 학습 설정 중...")

training_args = TrainingArguments(
    # 출력 디렉토리
    output_dir=CONFIG["output_dir"],
    
    # 학습 파라미터
    num_train_epochs=CONFIG["num_train_epochs"],
    per_device_train_batch_size=CONFIG["batch_size"],
    per_device_eval_batch_size=CONFIG["batch_size"],
    learning_rate=CONFIG["learning_rate"],
    weight_decay=CONFIG["weight_decay"],
    
    # 평가 및 저장
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    
    # 로깅
    logging_dir=f"{CONFIG['output_dir']}/logs",
    logging_strategy="steps",
    logging_steps=5,
    
    # 기타
    seed=CONFIG["seed"],
    push_to_hub=False,
    report_to="none",  # wandb 등 비활성화
    no_cuda=True,  # CPU 사용 강제 (accelerate 호환성)
)

# Trainer 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

print("✅ 학습 준비 완료!")

# ============================================
# 6. Fine-tuning 실행!
# ============================================

print("\n" + "="*60)
print("🔥 Fine-tuning 시작!")
print("="*60)

try:
    train_result = trainer.train()
    
    print("\n✅ Fine-tuning 완료!")
    print(f"📊 최종 Loss: {train_result.training_loss:.4f}")
    
except Exception as e:
    print(f"\n❌ 학습 중 오류 발생: {e}")
    raise

# ============================================
# 7. 평가
# ============================================

print("\n📊 검증 데이터 평가 중...")

eval_results = trainer.evaluate()

print("\n" + "="*60)
print("📈 최종 평가 결과")
print("="*60)
print(f"정확도 (Accuracy): {eval_results['eval_accuracy']:.2%}")
print(f"정밀도 (Precision): {eval_results['eval_precision']:.2%}")
print(f"재현율 (Recall): {eval_results['eval_recall']:.2%}")
print(f"F1 Score: {eval_results['eval_f1']:.2%}")
print(f"\n클래스별 정확도:")
print(f"  - 부정 (0): {eval_results['eval_class_0_acc']:.2%}")
print(f"  - 중립 (1): {eval_results['eval_class_1_acc']:.2%}")
print(f"  - 긍정 (2): {eval_results['eval_class_2_acc']:.2%}")
print("="*60)

# ============================================
# 8. 모델 저장
# ============================================

print("\n💾 모델 저장 중...")

final_model_dir = "./koelectra-game-sentiment"
model.save_pretrained(final_model_dir)
tokenizer.save_pretrained(final_model_dir)

print(f"✅ 모델 저장 완료: {final_model_dir}")

# ============================================
# 9. 테스트
# ============================================

print("\n🧪 모델 테스트...")

test_texts = [
    "로그인이 안 돼요 계속 오류나요",
    "이벤트 기간이 언제까지인가요?",
    "업데이트 후 정말 좋아졌어요 감사합니다"
]

label_names = ["부정", "중립", "긍정"]

for text in test_texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    
    print(f"  '{text}'")
    print(f"  → 예측: {label_names[pred]}\n")

# ============================================
# 완료!
# ============================================

print("="*60)
print("🎉 Fine-tuning 완료!")
print("="*60)
print(f"📁 모델 위치: {final_model_dir}")
print(f"📊 정확도: {eval_results['eval_accuracy']:.2%}")
print("\n다음 단계:")
print("1. src/koelectra_classifier.py 수정")
print("2. Fine-tuned 모델 경로 변경")
print("3. dashboard.py 재실행")
print("="*60)

