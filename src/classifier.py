import torch
from transformers import ElectraTokenizer, ElectraForSequenceClassification
from .config import ISSUE_CATEGORIES

class KoElectraClassifier:
    def __init__(self):
        self.tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
        self.models = {}
        for category in ISSUE_CATEGORIES:
            self.models[category] = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator", num_labels=2)  # Binary: positive/negative

    def classify(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        results = {}
        for cat, model in self.models.items():
            outputs = model(**inputs)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1).item()  # 0: negative, 1: positive
            results[cat] = "긍정" if pred == 1 else "부정"
        return results

# Note: This is a placeholder. Fine-tuning with labeled data is needed.
