#!/bin/bash
python $SCRATCH/research/repos/attrigram/attrigram/train.py \
--input_file_pattern="$SCRATCH/research/data/deepfashion_dataset/train-?????-of-00256" \
--inception_checkpoint_file="$SCRATCH/research/ckpts/inception/inception_v3.ckpt" \
--train_dir="$SCRATCH/research/ckpts/attrigram/train/" \
--train_inception=false \
--number_of_steps=100000 \