import os
import argparse
import random
import pandas as pd

def main(args):
    dev = pd.read_csv("data/synth/labels.csv")
    test = pd.read_csv("data/金剛經/labels.csv")
    # id dir starts with "XM", then it is a dev image
    # dev = labels[labels["image_id"].str.contains("XM")]
    # the rest are test images
    # test = labels[~labels["image_id"].str.contains("XM")]
    # train, validation split
    random.seed(args.seed)
    train = dev.sample(frac=args.train_ratio, random_state=args.seed)
    validation = dev.drop(train.index)
    train.to_csv(os.path.join(args.output_dir, "train.csv"), index=False)
    validation.to_csv(os.path.join(args.output_dir, "val.csv"), index=False)
    test.to_csv(os.path.join(args.output_dir, "test.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data into train and test.")
    # parser.add_argument(
    #     "--metadata_file",
    #     type=str,
    #     default="images/all.csv",
    # )
    parser.add_argument(
        "--train_ratio",
        type=int,
        default=0.8,
        help="Ratio of data to use for training.",
    )
    parser.add_argument(
        "--seed", type=int, default=999, help="Seed for random number generator."
    )
    parser.add_argument(
        "--output_dir", type=str, default="data", help="Directory to save the split data."
    )
    args = parser.parse_args()
    main(args)
