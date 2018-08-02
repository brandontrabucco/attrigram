#!/bin/bash

python $SCRATCH/research/repos/attrigram/attrigram/run_inference.py \
--checkpoint_path="$SCRATCH/research/ckpts/attrigram/train/" \
--input_files="$SCRATCH/research/repos/attrigram/data/*.jpg" \