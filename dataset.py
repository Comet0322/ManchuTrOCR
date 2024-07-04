import random
from functools import partial
from typing import Any, List, Mapping, Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import TrOCRProcessor


def get_black_channel(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = gray.astype(float) / 255.

    # Calculate channel K:
    gray = 1 - np.max(gray, axis=2)

    # Convert back to uint 8:
    gray = (255 * gray).astype(np.uint8)
    return gray

class ManchuDataset(Dataset):
    def __init__(self, df, transform, processor, max_len=30):
        if isinstance(df, str):
            self.df = pd.read_csv(df, keep_default_na=False)
        else:
            self.df = df
        self.processor = processor
        self.max_len = max_len
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df["image_id"][idx]
        img = Image.open(file_name).convert("RGB")
        img = np.array(img)
        # extract black pixels
        black = get_black_channel(img)
        _, img = cv2.threshold(black, 120, 255, cv2.THRESH_BINARY)
        img = cv2.bitwise_not(img)
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        target_height, target_width = 64, self.processor.image_processor.size["width"]
        if self.transform is not None:
            img = self.transform(image=img, target_width=target_width)
        else:
            pad = int(img.shape[0] * 0.15)
            img = cv2.copyMakeBorder(
                img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )
            h, w, _ = img.shape
            height = target_height
            width = int(w * height / h)
            if width >= target_width:
                img = cv2.resize(img, (target_width, target_height))
            else:
                img = cv2.resize(img, (width, height))

            img_pad = np.zeros((target_height, target_width, 3), dtype=img.dtype)
            img_pad.fill(255)
            img_pad[:height, :width] = img
            img = img_pad
        

        pixel_values = self.processor(
            images=img, return_tensors="pt"
        ).pixel_values.squeeze()

        if "text" in self.df:
            text = self.df["text"][idx]
            labels = self.processor.tokenizer(
                text, padding="max_length", max_length=self.max_len
            ).input_ids
            # important: make sure that PAD tokens are ignored by the loss function
            labels = [
                label if label != self.processor.tokenizer.pad_token_id else -100
                for label in labels
            ]
            encoding = {
                "pixel_values": pixel_values,
                "labels": torch.tensor(labels),
            }
        else:
            encoding = {
                "pixel_values": pixel_values,
            }

        return encoding

        

def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    if "labels" in batch[0]:
        labels = torch.stack([item["labels"] for item in batch])
        return {"pixel_values": pixel_values, "labels": labels}
    else:
        return {"pixel_values": pixel_values}


def aug2(image, target_width=480, **kwargs):
    image = Image.fromarray(image)
    random_rotate = random.uniform(-2, 2)
    image = image.rotate(random_rotate, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
    image = np.array(image)
    
    random_scale_x = random.uniform(0.7, 1.2)
    random_scale_y = random.uniform(0.7, 1.2)
    new_size = (
        int(image.shape[1] * random_scale_x),
        int(image.shape[0] * random_scale_y),
    )
    image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    pad = int(image.shape[0] * 0.15)
    image = cv2.copyMakeBorder(
        image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    # only augment 3/4th the images
    if image.shape[1] <= target_width:
        random_offset = random.randint(0, target_width - image.shape[1])
        # save
        image = cv2.copyMakeBorder(
            image.copy(),
            0,
            0,
            random_offset,
            target_width - image.shape[1] - random_offset,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )

    else:
        image = cv2.resize(image, (target_width, 64), interpolation=cv2.INTER_AREA)

    if random.randint(1, 4) > 3:
        return image


    # # morphological alterations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if random.randint(1, 5) == 1:
        # erosion because the image is not inverted
        image = cv2.dilate(image, kernel, iterations=1)

    transform = A.Compose(
        [
            A.OneOf(
                [
                    # add black pixels noise
                    A.OneOf(
                        [
                            A.RandomRain(
                                brightness_coefficient=1.0,
                                drop_length=1,
                                drop_width=1,
                                drop_color=(0, 0, 0),
                                blur_value=1,
                                rain_type="drizzle",
                                p=0.05,
                            ),
                            # A.RandomShadow(p=1),
                            A.PixelDropout(p=1),
                        ],
                        p=0.9,
                    ),
                    # add white pixels noise
                    A.OneOf(
                        [
                            A.PixelDropout(dropout_prob=0.5, drop_value=255, p=1),
                            A.RandomRain(
                                brightness_coefficient=1.0,
                                drop_length=2,
                                drop_width=2,
                                drop_color=(255, 255, 255),
                                blur_value=1,
                                rain_type=None,
                                p=1,
                            ),
                        ],
                        p=0.9,
                    ),
                ],
                p=1,
            ),
            A.Blur(blur_limit=3, p=0.25),
        ]
    )

    image = transform(image=image)["image"]
    return image


def get_dataset(
    processor: TrOCRProcessor,
    train_file=None,
    val_file=None,
    test_file=None,
    max_len=25,
):

    dataset = {}
    dataset["train"] = ManchuDataset(
        train_file, aug2, processor, max_len=max_len
    )
    dataset["validation"] = ManchuDataset(
        val_file, None, processor, max_len=max_len
    )
    if test_file is not None:
        dataset["test"] = ManchuDataset(
            test_file, None, processor, max_len=max_len
        )

    return dataset


if __name__ == "__main__":
    from transformers import TrOCRProcessor
    random.seed(999)
    from models import get_crnn_model
    model, processor = get_crnn_model(device="cuda")
    dataset = get_dataset(processor, "data/train.csv", "data/val.csv", "data/親征平定朔漠方略/crop.csv")
    print(dataset["train"][0])