#!/bin/bash
python run.py \
    --frame_dir {frame_dir} \
    --train_size 0.75 --test_size 0.15 \
    --model_type x3d --n_classes 51 --fr_per_vid 16 \
    --batch_size 24 --mode train --n_epochs 150 --learning_rate 2e-4 \
    --weight_decay 1e-3 --freeze_backbone_until 5 --backbone_lr_factor 1.0 \
    --grad_clip_norm 5.0 --lr_patience 8 --min_lr 1e-6 \
    --wandb_project hmdb51-x3d --run_name x3d-backbone-lr-1.0