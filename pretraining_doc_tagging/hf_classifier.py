import logging
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


class TextClassifierHf:
    def __init__(self, model_name: str, batch_size: int = 32):
        # NOTE: this is based on the fineweb-edu classifier but others may return things slightly differently
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        results = []

        for i in tqdm(range(0, len(texts), self.batch_size)):
            batch = texts[i : i + self.batch_size]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding="longest",
                truncation=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = outputs.logits.squeeze(-1).float().cpu().numpy()

            batch_results = [
                {
                    "raw_score": float(score),
                    "int_score": int(round(max(0, min(float(score), 5)))),
                }
                for score in scores
            ]
            results.extend(batch_results)

        return results
