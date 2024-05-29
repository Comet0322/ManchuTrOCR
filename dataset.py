import json
from functools import partial
from typing import Any, List, Mapping, Optional, Tuple, Union

import albumentations as A
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from transformers import TrOCRProcessor
from transformers.image_processing_utils import BatchFeature


def augment_and_transform_batch(
    examples: Mapping[str, Any],
    transform: A.Compose,
    processor: TrOCRProcessor,
) -> BatchFeature:

    images = []
    for image_id in examples["image_id"]:
        image = Image.open(image_id)
        image = np.array(image.convert("RGB").rotate(90, expand=True))
        output = transform(image=image)
        images.append(output["image"])

    examples["pixel_values"] = processor(
        images=images, return_tensors="pt"
    ).pixel_values
    examples["label"] = processor.tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=20,
        return_tensors="pt",
    ).input_ids

    return examples


def collate_fn(
    batch: List[BatchFeature],
) -> Mapping[str, Union[torch.Tensor, List[Any]]]:
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = torch.stack([x["label"] for x in batch])

    return data


def get_dataset(processor: TrOCRProcessor):
    with open("transliter.json") as f:
        manjurules = json.load(f)

    new_tokens = [i["roman"] for i in manjurules]
    processor.tokenizer.add_tokens(list(new_tokens))
    image_size = processor.feature_extractor.size["height"]
    train_augment_and_transform = A.Compose(
        [
            A.SmallestMaxSize(max_size=64, p=1.0),
            A.PadIfNeeded(min_height=64, min_width=192, p=1.0, value=(255, 255, 255)),
            A.LongestMaxSize(max_size=image_size, p=1.0),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                p=1.0,
                value=(255, 255, 255),
            ),
            # A.Rotate(limit=10, p=0.5),
            A.OneOf(
                [
                    A.Blur(blur_limit=7, p=0.5),
                    A.MotionBlur(blur_limit=7, p=0.5),
                    A.Defocus(radius=(1, 5), alias_blur=(0.1, 0.25), p=0.1),
                ],
                p=0.1,
            ),
            #     A.Perspective(p=0.1),
            #     A.HorizontalFlip(p=0.5),
            #     A.RandomBrightnessContrast(p=0.5),
            #     A.HueSaturationValue(p=0.1),
        ],
    )
    validation_transform = A.Compose(
        [
            A.SmallestMaxSize(max_size=64, p=1.0),
            A.PadIfNeeded(min_height=64, min_width=192, p=1.0, value=(255, 255, 255)),
            A.LongestMaxSize(max_size=image_size, p=1.0),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                p=1.0,
                value=(255, 255, 255),
            ),
        ],
    )

    train_transform_batch = partial(
        augment_and_transform_batch,
        transform=train_augment_and_transform,
        processor=processor,
    )
    validation_transform_batch = partial(
        augment_and_transform_batch, transform=validation_transform, processor=processor
    )

    dataset = load_dataset(
        "csv",
        data_files={
            "train": "data/train.csv",
            "validation": "data/val.csv",
            "test": "data/test.csv",
        },
    )

    dataset["train"] = dataset["train"].with_transform(train_transform_batch)
    dataset["validation"] = dataset["validation"].with_transform(
        validation_transform_batch
    )
    dataset["test"] = dataset["test"].with_transform(validation_transform_batch)

    return dataset
