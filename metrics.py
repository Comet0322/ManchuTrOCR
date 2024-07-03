from typing import Any, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from evaluate import load
from transformers.trainer import EvalPrediction


class Metrics:
    def __init__(self, processor):
        self.processor = processor
        self.cer_metric = load("cer")

    @torch.no_grad()
    def compute_metrics(
        self,
        evaluation_results: EvalPrediction,
    ) -> Mapping[str, float]:
        # exact match and character accuracy
        predictions = evaluation_results.predictions[0]
        references = evaluation_results.label_ids
        assert predictions.shape == references.shape
        references[references == -100] = self.processor.tokenizer.pad_token_id
        pred_str = self.processor.batch_decode(predictions, skip_special_tokens=True)
        label_str = self.processor.batch_decode(references, skip_special_tokens=True)
        pred_str = np.array(["".join(text.split()) for text in pred_str])
        label_str = np.array(["".join(text.split()) for text in label_str])
        # remove special tokens
        predicted_sequences = [
            [i for i in p if i not in self.processor.tokenizer.all_special_ids]
            for p in predictions
        ]
        reference_sequences = [
            [i for i in r if i not in self.processor.tokenizer.all_special_ids]
            for r in references
        ]
        predicted_sequences = [
            "".join([chr(i + 65) for i in p]) for p in predicted_sequences
        ]
        reference_sequences = [
            "".join([chr(i + 65) for i in r]) for r in reference_sequences
        ]
        results = {}
        results["accuracy"] = (pred_str == label_str).mean()
        results["CER"] = self.cer_metric.compute(
            predictions=predicted_sequences, references=reference_sequences
        )

        return results


if __name__ == "__main__":
    preds = [[0, 1, 47], [0, 2, 2]]
    labels = [[0, 1, 2], [0, 2, 0]]
    # transform to ascii, 'A' as offset
    preds = ["".join([chr(i + 65) for i in p]) for p in preds]
    labels = ["".join([chr(i + 65) for i in p]) for p in labels]
    # join list of chars to strings
    cer_metric = load("cer")
    print(preds, labels)
    print(cer_metric.compute(predictions=preds, references=labels))
