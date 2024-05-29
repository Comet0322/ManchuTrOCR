import logging
import os
import sys
from functools import partial
from typing import Any, List, Mapping, Optional, Tuple, Union

import torch
import transformers
from datasets import load_metric
from transformers import (AutoConfig, AutoTokenizer, HfArgumentParser, Trainer,
                          TrainingArguments, TrOCRProcessor,
                          VisionEncoderDecoderModel, ViTFeatureExtractor)
from transformers.trainer import EvalPrediction
from transformers.trainer_utils import get_last_checkpoint

from dataset import collate_fn, get_dataset


def calculate_cra(predicted_sequences, ground_truth_sequences):
    total_characters = 0
    correct_characters = 0

    for pred_seq, gt_seq in zip(predicted_sequences, ground_truth_sequences):
        total_characters += len(gt_seq)
        correct_characters += sum(p == g for p, g in zip(pred_seq, gt_seq))

    return correct_characters / total_characters if total_characters > 0 else 0


@torch.no_grad()
def compute_metrics(
    evaluation_results: EvalPrediction,
    processor: TrOCRProcessor,
) -> Mapping[str, float]:
    # exact match and character accuracy
    predictions = processor.batch_decode(evaluation_results.predictions)
    references = processor.batch_decode(evaluation_results.label_ids)
    exact_match_metric = load_metric("exact_match")
    results = exact_match_metric.compute(predictions=predictions, references=references)

    # remove special tokens
    predicted_sequences = [
        [i for i in p if i not in processor.tokenizer.all_special_ids]
        for p in predictions
    ]
    reference_sequences = [
        [i for i in r if i not in processor.tokenizer.all_special_ids]
        for r in references
    ]
    cer_metric = load_metric("cer")
    results["CRA"] = 100 * (
        1
        - cer_metric.compute(
            predictions=predicted_sequences, references=reference_sequences
        )
    )

    return results


if __name__ == "__main__":
    encoder = "google/vit-base-patch16-224-in21k"
    decoder = "xlm-roberta-base"
    training_args = TrainingArguments(
        output_dir="./output",
        fp16=True,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_dir="./logs",
        logging_strategy="steps",
        # logging_steps=10,
        # eval_steps=10,
        # save_steps=10,
        report_to="wandb",
        run_name="manchu-ocr",
        # torch_compile = True,
    )
    logger = logging.getLogger(__name__)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # set all seeds for reproducibility
    torch.manual_seed(training_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    feature_extractor = ViTFeatureExtractor.from_pretrained(encoder)
    tokenizer = AutoTokenizer.from_pretrained(decoder)
    processor = TrOCRProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    dataset = get_dataset(processor)
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(encoder, decoder)

    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.resize_token_embeddings(len(processor.tokenizer))
    assert model.config.decoder.is_decoder is True
    assert model.config.decoder.add_cross_attention is True
    eval_compute_metrics_fn = partial(compute_metrics, processor=processor)

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif (
        os.path.isdir(training_args.output_dir)
        and not training_args.overwrite_output_dir
    ):
        checkpoint = get_last_checkpoint(training_args.output_dir)
        if checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=processor,
        data_collator=collate_fn,
        compute_metrics=eval_compute_metrics_fn,
    )

    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Final evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(
            eval_dataset=dataset["test"], metric_key_prefix="test"
        )
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)
