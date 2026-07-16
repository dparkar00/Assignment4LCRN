#!/bin/bash
!python run.py \
    --frame_dir {frame_dir} \
    --ckpt ./models/best_model_wts.pt \
    --model_type x3d --n_classes 51 --batch_size 24 --mode eval \
    --tta_clips 1