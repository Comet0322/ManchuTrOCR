import logging
import os
import sys
from functools import partial
from typing import Any, List, Mapping, Optional, Tuple, Union

import torch
import transformers
from datasets import load_metric
from transformers import (AutoConfig, HfArgumentParser, Trainer,
                          TrainingArguments, TrOCRProcessor,
                          VisionEncoderDecoderModel)
from transformers.trainer import EvalPrediction
from transformers.trainer_utils import get_last_checkpoint

from dataset import collate_fn, get_dataset


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

    # accuracy_metric = load_metric("accuracy")
    # concatenate all predictions and references to compute character accuracy
    # remove special tokens
    # predictions = sum(predictions, [])
    # references = sum(references, [])
    # results.update(accuracy_metric.compute(predictions=predictions, references=references))

    return results


if __name__ == "__main__":
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
    dataset, processor = get_dataset()

    model = VisionEncoderDecoderModel.from_pretrained(
        "microsoft/trocr-base-handwritten"
    )
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

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
