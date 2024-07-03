#!/bin/bash
python3 data/金剛經/image_generate.py
# trdg -l mc -c 1000000 -rbl -b 1 -d2 -rk -f 64 --word_split -t 16 --output_dir data/synth2/images
python3 data/synth6/image_generate.py
python3 split_data.py