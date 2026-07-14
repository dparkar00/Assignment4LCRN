#!/bin/bash
# Log in once per machine with: wandb login
#
# TUNED: learning_rate raised from 3e-5 -> 1e-4, freeze_backbone_until raised
# from 2 -> 4, batch_size raised from 4 -> 8, n_epochs raised to 35, dropout
# raised to 0.4, lr_patience lowered to 3 (see prior smoke tests).
#
# A full 35-epoch run at those settings plateaued around 51% test accuracy
# with severe overfitting (train loss 0.31 vs val loss 1.93 by the final
# epoch) -- the classifier head only ever saw the LSTM's final hidden state,
# and augmentation was too weak to prevent memorization on ~100 clips/class.
# Added: attention pooling over all LSTM timestep outputs (--use_attention),
# stronger augmentation (color jitter + random-resized crop, see utils.py),
# and label smoothing (see run.py) to directly attack that overfitting gap.
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
    --use_attention True \
    --freeze_backbone_until 4 \
    --wandb_project hmdb51-lrcn \
    --run_name lrcn-attention-run1