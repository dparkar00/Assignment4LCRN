#!/bin/bash
# Log in once per machine with: wandb login
#
# TUNED: defaults below were adjusted after a 2-epoch smoke test showed loss
# dropping too slowly to reach a strong accuracy within a realistic training
# budget. learning_rate raised from 3e-5 -> 1e-4, freeze_backbone_until raised
# from 2 -> 4 (lets more of the backbone adapt), batch_size raised from 4 -> 8
# (better GPU utilization + more stable gradients), n_epochs raised to 35.
python run.py \
    --frame_dir HMDB51 \
    --train_size 0.75 \
    --test_size 0.15 \
    --model_type lrcn \
    --n_classes 51 \
    --fr_per_vid 16 \
    --batch_size 8 \
    --mode train \
    --n_epochs 35 \
    --learning_rate 1e-4 \
    --bidirectional True \
    --freeze_backbone_until 4 \
    --wandb_project hmdb51-lrcn \
    --run_name lrcn-tuned-run1