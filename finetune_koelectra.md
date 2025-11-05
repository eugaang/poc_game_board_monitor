(poc_game_monitor) (base) eugaang@choeyugang-ui-MacBookPro poc_game_board_monitor % python finetune_koelectra.py
============================================================
🤖 KoELECTRA Fine-tuning
============================================================
📊 Train data: data/train_koelectra_120.csv
📊 Val data: data/val_koelectra_30.csv
🔧 Model: monologg/koelectra-base-v3-discriminator
💻 Device: CPU
📝 Epochs: 5
📦 Batch size: 8
============================================================

📂 데이터 로딩 중...
✅ Train: 120개
✅ Val: 30개

📊 레이블 분포 (Train):
sentiment_label
0    50
1    50
2    20
Name: count, dtype: int64

📊 레이블 분포 (Val):
sentiment_label
0     8
1    20
2     2
Name: count, dtype: int64

🔤 토크나이저 로딩 중...
🔤 토큰화 진행 중...
Map: 100%|████████████████████████████████████████████████████████████████████████████████| 120/120 [00:00<00:00, 8254.74 examples/s]
Map: 100%|██████████████████████████████████████████████████████████████████████████████████| 30/30 [00:00<00:00, 6718.77 examples/s]
✅ 토큰화 완료!

🤖 모델 로딩 중: monologg/koelectra-base-v3-discriminator
Some weights of ElectraForSequenceClassification were not initialized from the model checkpoint at monologg/koelectra-base-v3-discriminator and are newly initialized: ['classifier.out_proj.weight', 'classifier.out_proj.bias', 'classifier.dense.bias', 'classifier.dense.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
✅ 모델 로드 완료!

⚙️ 학습 설정 중...
✅ 학습 준비 완료!

============================================================
🔥 Fine-tuning 시작!
============================================================
{'loss': 1.1039, 'learning_rate': 1.866666666666667e-05, 'epoch': 0.33}
{'loss': 1.0814, 'learning_rate': 1.7333333333333336e-05, 'epoch': 0.67}
{'loss': 1.0231, 'learning_rate': 1.6000000000000003e-05, 'epoch': 1.0}
{'eval_loss': 0.9407457709312439, 'eval_accuracy': 0.8333333333333334, 'eval_precision': 0.779296066252588, 'eval_recall': 0.8333333333333334, 'eval_f1': 0.8024806201550388, 'eval_class_0_acc': 0.75, 'eval_class_1_acc': 0.95, 'eval_class_2_acc': 0.0, 'eval_runtime': 3.8264, 'eval_samples_per_second': 7.84, 'eval_steps_per_second': 1.045, 'epoch': 1.0}
{'loss': 0.9759, 'learning_rate': 1.4666666666666666e-05, 'epoch': 1.33}
{'loss': 0.9274, 'learning_rate': 1.3333333333333333e-05, 'epoch': 1.67}
{'loss': 0.8645, 'learning_rate': 1.2e-05, 'epoch': 2.0}
{'eval_loss': 0.7688537836074829, 'eval_accuracy': 0.9, 'eval_precision': 0.8466666666666666, 'eval_recall': 0.9, 'eval_f1': 0.8703703703703703, 'eval_class_0_acc': 1.0, 'eval_class_1_acc': 0.95, 'eval_class_2_acc': 0.0, 'eval_runtime': 3.8064, 'eval_samples_per_second': 7.881, 'eval_steps_per_second': 1.051, 'epoch': 2.0}
{'loss': 0.8391, 'learning_rate': 1.0666666666666667e-05, 'epoch': 2.33}
{'loss': 0.7407, 'learning_rate': 9.333333333333334e-06, 'epoch': 2.67}
{'loss': 0.7111, 'learning_rate': 8.000000000000001e-06, 'epoch': 3.0}
{'eval_loss': 0.5995989441871643, 'eval_accuracy': 1.0, 'eval_precision': 1.0, 'eval_recall': 1.0, 'eval_f1': 1.0, 'eval_class_0_acc': 1.0, 'eval_class_1_acc': 1.0, 'eval_class_2_acc': 1.0, 'eval_runtime': 3.795, 'eval_samples_per_second': 7.905, 'eval_steps_per_second': 1.054, 'epoch': 3.0}
{'loss': 0.6875, 'learning_rate': 6.666666666666667e-06, 'epoch': 3.33}
{'loss': 0.628, 'learning_rate': 5.333333333333334e-06, 'epoch': 3.67}
{'loss': 0.5936, 'learning_rate': 4.000000000000001e-06, 'epoch': 4.0}
{'eval_loss': 0.5042083859443665, 'eval_accuracy': 1.0, 'eval_precision': 1.0, 'eval_recall': 1.0, 'eval_f1': 1.0, 'eval_class_0_acc': 1.0, 'eval_class_1_acc': 1.0, 'eval_class_2_acc': 1.0, 'eval_runtime': 3.8626, 'eval_samples_per_second': 7.767, 'eval_steps_per_second': 1.036, 'epoch': 4.0}
{'loss': 0.6128, 'learning_rate': 2.666666666666667e-06, 'epoch': 4.33}
{'loss': 0.5443, 'learning_rate': 1.3333333333333334e-06, 'epoch': 4.67}
{'loss': 0.5452, 'learning_rate': 0.0, 'epoch': 5.0}
{'eval_loss': 0.470001757144928, 'eval_accuracy': 1.0, 'eval_precision': 1.0, 'eval_recall': 1.0, 'eval_f1': 1.0, 'eval_class_0_acc': 1.0, 'eval_class_1_acc': 1.0, 'eval_class_2_acc': 1.0, 'eval_runtime': 3.804, 'eval_samples_per_second': 7.887, 'eval_steps_per_second': 1.052, 'epoch': 5.0}
{'train_runtime': 465.1945, 'train_samples_per_second': 1.29, 'train_steps_per_second': 0.161, 'train_loss': 0.791907148361206, 'epoch': 5.0}
100%|████████████████████████████████████████████████████████████████████████████████████████████████| 75/75 [07:45<00:00,  6.20s/it]

✅ Fine-tuning 완료!
📊 최종 Loss: 0.7919

📊 검증 데이터 평가 중...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:02<00:00,  1.44it/s]

============================================================
📈 최종 평가 결과
============================================================
정확도 (Accuracy): 100.00%
정밀도 (Precision): 100.00%
재현율 (Recall): 100.00%
F1 Score: 100.00%

클래스별 정확도:
  - 부정 (0): 100.00%
  - 중립 (1): 100.00%
  - 긍정 (2): 100.00%
============================================================

💾 모델 저장 중...
✅ 모델 저장 완료: ./koelectra-game-sentiment

🧪 모델 테스트...
  '로그인이 안 돼요 계속 오류나요'
  → 예측: 부정

  '이벤트 기간이 언제까지인가요?'
  → 예측: 중립

  '업데이트 후 정말 좋아졌어요 감사합니다'
  → 예측: 중립

============================================================
🎉 Fine-tuning 완료!
============================================================
📁 모델 위치: ./koelectra-game-sentiment
📊 정확도: 100.00%

다음 단계:
1. src/koelectra_classifier.py 수정
2. Fine-tuned 모델 경로 변경
3. dashboard.py 재실행
============================================================