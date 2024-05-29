import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def job(font_name, word, romanized, args):
    # Generate images
    base_img = Image.new("RGB", (args.width, args.height), color="white")
    font = ImageFont.truetype(f"{args.font_dir}/{font_name}", size=args.font_size)
    base_draw = ImageDraw.Draw(base_img)
    text_bbox = base_draw.textbbox(
        (0, 0), word, font=font, anchor="lt", language="ar-SA"
    )
    text_width = text_bbox[2] - text_bbox[0]

    img = Image.new("RGB", (text_width + 2 * args.padding, args.height), color="white")
    ImageDraw.Draw(img).text(
        (args.padding, args.height / 2),
        word,
        fill=(0, 0, 0),
        font=font,
        anchor="lm",
        language="ar-SA",
    )

    # rotate image 90 degree
    img = img.rotate(-90, expand=True)
    output_path = Path(f"{args.output_dir}/images/{font_name[:-4]}/{romanized}.jpg")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(str(output_path), np.array(img))

    return str(output_path), romanized


def main(args):
    # find all fonts
    fonts = [f for f in os.listdir(args.font_dir) if f.endswith(".ttf")]
    fonts = ["XM_BiaoHei.ttf"]
    with open(args.word_file, "r") as f:
        word_list = f.read().splitlines()
    with open(args.transliter_file) as f:
        manjurules = json.load(f)

    words_df = pd.DataFrame(word_list, columns=["roman"])
    words_df["roman"] = words_df["roman"].str.lower()
    words_df["manju"] = words_df["roman"]
    for rule in manjurules:
        words_df["manju"] = words_df["manju"].apply(
            lambda x: x.replace(rule["roman"], rule["manju"])
        )

    words_df = words_df[words_df["manju"].str.len() < args.max_word_length]

    result = []
    for i in tqdm(range(len(words_df))):
        romanized, word = words_df.iloc[i]
        for font_name in fonts:
            result.append(job(font_name, word, romanized, args))

    with open(f"{args.output_dir}/labels.csv", "w") as f:
        f.write("image_id,text\n")
        for path, label in result:
            f.write(f"{path},{label}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images from text.")
    parser.add_argument("--width", type=int, default=160, help="Width of the image.")
    parser.add_argument("--height", type=int, default=64, help="Height of the image.")
    parser.add_argument(
        "--padding", type=int, default=5, help="Padding around the text."
    )
    parser.add_argument(
        "--max_word_length", type=int, default=10, help="Maximum length of the manchu word."
    )
    parser.add_argument(
        "--font_size", type=int, default=32, help="Font size for the text."
    )
    parser.add_argument(
        "--font_dir", type=str, default="data/synth/fonts", help="Directory with fonts."
    )
    parser.add_argument(
        "--word_file", type=str, default="data/synth/AllWords.txt", help="Input file with words."
    )
    parser.add_argument(
        "--transliter_file",
        type=str,
        default="transliter.json",
        help="Input file with transliteration rules.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/synth", help="Directory to save the images."
    )
    args = parser.parse_args()

    main(args)
