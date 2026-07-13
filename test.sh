#!/bin/bash
python run.py \
    --ckpt ./models/best_model_wts.pt \
    --model_type lrcn \
    --n_classes 51 \
    --batch_size 4 \
    --mode eval \
    --wandb_project lrcn-baseline \
    --run_name lrcn-fixed-eval