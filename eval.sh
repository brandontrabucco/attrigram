#!/bin/bash

python $SCRATCH/research/repos/attrigram/attrigram/evaluate.py \
--input_file_pattern="$SCRATCH/research/data/deepfashion_dataset/val-?????-of-00004" \
--checkpoint_dir="$SCRATCH/research/ckpts/attrigram/train/" \
--eval_dir="$SCRATCH/research/ckpts/attrigram/eval/"