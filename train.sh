#!/bin/bash
# Log in once per machine with: wandb login
#
# TUNED: learning_rate raised from 3e-5 -> 1e-4, freeze_backbone_until raised
# from 2 -> 4, batch_size raised from 4 -> 8, n_epochs raised to 35 (see prior
# smoke tests). A 5-epoch check at these settings showed val loss plateauing
# by epoch 2-3 while train loss kept dropping (train 3.11->1.15, val
# 2.65->2.06) -- an early overfitting signature. Countered with higher dropout
# (0.1 -> 0.4) and a lower LR scheduler patience (5 -> 3) so the LR backs off
# sooner once val loss stalls, instead of continuing to overfit at a fixed LR.
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
    --dropout 0.4 \
    --lr_patience 3 \
    --bidirectional True \
    --freeze_backbone_until 4 \
    --wandb_project hmdb51-lrcn \
    --run_name lrcn-tuned-run2