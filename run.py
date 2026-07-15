"""
Module: run.py

This module is the main entry point for training or evaluating a video classification model.
It uses command-line arguments to configure the experiment, including dataset paths, model
parameters, and training hyperparameters. Depending on the selected mode ('train' or 'eval'),
it performs the following:

- Train mode:
    - Loads the dataset from a directory structure.
    - Splits the dataset into training, validation, and test sets.
    - Creates PyTorch Datasets and DataLoaders for training and validation.
    - Initializes the model (e.g., LRCN) and defines the loss function, optimizer, and
      learning rate scheduler.
    - Runs the training loop and saves the best model weights.

- Eval mode:
    - Loads the pre-generated dataset splits.
    - Creates a DataLoader for the test set.
    - Loads the trained model checkpoint.
    - Evaluates the model on the test set and prints overall test accuracy.

The module also includes a helper function for parsing command-line arguments.
"""

import os
import glob
import argparse

import numpy as np
import torch
import wandb
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

from video_datasets import (VideoDataset, TwoStreamVideoDataset, load_dataset,
                             dataset_split, collate_fn_two_stream)
from utils import (transform_stats, compose_data_transforms, train_val_dloaders,
                    test_dloaders, compute_flow_frames, store_frames)
from models import LRCN, TwoStreamI3D
from train import train
from test import test  # pylint: disable=wrong-import-order
# (pylint mistakes this repo's local test.py for the Python stdlib 'test' package by name)


def args_parser():
    """
    Parse command-line arguments for configuring the video classification training or evaluation.

    Returns:
        argparse.Namespace: Parsed command-line arguments.

    Arguments include:
        -fd/--frame_dir: Directory for storing video frames.
        -flowd/--flow_dir: Directory for pre-computed optical flow frames (two-stream mode).
        -trs/--train_size: Proportion of data to use for training (default 0.7).
        -tss/--test_size: Proportion of data to use for testing (default 0.1).
        -fpv/--fr_per_vid: Number of frames per video to consider (default 16).
        -nc/--n_classes: Number of classes for the classification task (required).
        -c/--ckpt: Path for loading a trained model checkpoint.
        -mt/--model_type: Model type, '3dcnn', 'lrcn', or 'i3d_two_stream' (default 'lrcn').
        -cnn/--cnn_backbone: Backbone CNN for 2D feature extraction (default 'resnet34').
        -p/--pretrained: Whether to use a pretrained CNN backbone (default True).
        -rhs/--rnn_hidden_size: Number of neurons in the RNN/LSTM hidden layer (default 100).
        -rnl/--rnn_n_layers: Number of RNN/LSTM layers (default 1).
        -m/--mode: Mode of operation: 'train', 'eval', or 'preprocess_flow' (required).
        -bs/--batch_size: Mini-batch size (required).
        -d/--dropout: Dropout rate for regularization (default 0.1).
        -lr/--learning_rate: Learning rate for training (default 3e-5).
        -ne/--n_epochs: Number of training epochs (default 30).
        -wp/--wandb_project: Weights & Biases project name (default 'hmdb51-video-classification').
        -rn/--run_name: Weights & Biases run name (optional; wandb auto-generates one if omitted).
        --no_wandb: Disable Weights & Biases logging entirely (enabled by default in train mode).
    """
    parser = argparse.ArgumentParser(description='Video Classification Training')

    parser.add_argument('-fd', '--frame_dir', help='Directory for storing video frames')
    parser.add_argument('-flowd', '--flow_dir',
                         help='Directory for pre-computed optical flow frames '
                              '(required when model_type=i3d_two_stream)')
    parser.add_argument('-trs', '--train_size', type=float, default=0.7, help='Train set size')
    parser.add_argument('-tss', '--test_size', type=float, default=0.1, help='Test set size')
    parser.add_argument('-fpv', '--fr_per_vid', type=int, default=16,
                         help='Number of frames per video')
    parser.add_argument('-nc', '--n_classes', type=int, required=True,
                         help='Number of classes for the classification task')

    parser.add_argument('-c', '--ckpt', help='Path for loading trained model checkpoints')
    parser.add_argument('-mt', '--model_type', default='lrcn', help='3D CNN or LRCN')
    parser.add_argument('-cnn', '--cnn_backbone', default='resnet34',
                         help='2D CNN backbone - options: resnet18, resnet34, resnet50, '
                              'resnet101, resnet152')
    parser.add_argument('-p', '--pretrained', default=True,
                         help='Use pretrained 2D CNN backbone')
    parser.add_argument('-rhs', '--rnn_hidden_size', type=int, default=100,
                         help='Number of neurons in the RNN/LSTM hidden layer')
    parser.add_argument('-rnl', '--rnn_n_layers', type=int, default=1,
                         help='Number of RNN/LSTM layers')

    parser.add_argument('-m', '--mode', type=str, default='train', required=True,
                         help="Either 'train', 'eval', or 'preprocess_flow'")
    parser.add_argument('-bs', '--batch_size', type=int, required=True, help='Mini-batch size')
    parser.add_argument('-d', '--dropout', type=float, default=0.1,
                         help='Dropout rate for regularization')
    parser.add_argument('-lr', '--learning_rate', type=float, default=3e-5,
                         help='Learning rate for model training')
    parser.add_argument('-ne', '--n_epochs', type=int, default=30,
                         help='Number of training epochs')

    parser.add_argument('-wp', '--wandb_project', default='hmdb51-video-classification',
                         help='Weights & Biases project name')
    parser.add_argument('-rn', '--run_name', default=None,
                         help='Weights & Biases run name (optional)')
    parser.add_argument('--no_wandb', action='store_true',
                         help='Disable Weights & Biases logging')

    return parser.parse_args()


def build_model(args):
    """
    Instantiate the model specified by args: TwoStreamI3D for two-stream mode, otherwise the
    LRCN baseline.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        torch.nn.Module: The instantiated (untrained) model.
    """
    if args.model_type == 'i3d_two_stream':
        return TwoStreamI3D(n_classes=args.n_classes, pretrained=args.pretrained)
    return LRCN(hidden_size=args.rnn_hidden_size, n_layers=args.rnn_n_layers,
                dropout_rate=args.dropout, n_classes=args.n_classes,
                pretrained=args.pretrained, cnn_model=args.cnn_backbone)


def build_train_dataloaders(args, tr_split, val_split, tr_transforms, val_ts_transforms):
    """
    Build the training and validation DataLoaders, branching on model_type since two-stream
    I3D needs matching RGB/flow clips while the LRCN/3DCNN path needs a single RGB clip.

    Returns:
        dict: Dictionary with 'train' and 'val' DataLoaders.
    """
    if args.model_type == 'i3d_two_stream':
        tr_dataset = TwoStreamVideoDataset(tr_split, args.flow_dir, args.fr_per_vid,
                                            tr_transforms, tr_transforms)
        val_dataset = TwoStreamVideoDataset(val_split, args.flow_dir, args.fr_per_vid,
                                             val_ts_transforms, val_ts_transforms)
        train_dl = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_fn_two_stream)
        val_dl = DataLoader(val_dataset, batch_size=2 * args.batch_size, shuffle=False,
                             collate_fn=collate_fn_two_stream)
        return {'train': train_dl, 'val': val_dl}

    tr_dataset = VideoDataset(tr_split, args.fr_per_vid, tr_transforms)
    val_dataset = VideoDataset(val_split, args.fr_per_vid, val_ts_transforms)
    return train_val_dloaders(tr_dataset, val_dataset, args.batch_size, args.model_type)


def run_train(args, model, device, tr_transforms, val_ts_transforms):
    """
    Train mode: load and split the dataset, save the splits, build dataloaders, and run the
    training loop, logging metrics to Weights & Biases unless --no_wandb is set.
    """
    vid_dataset, _ = load_dataset(args.frame_dir)
    tr_split, val_split, ts_split = dataset_split(vid_dataset, args.train_size, args.test_size)

    # Save the splits for reproducibility and later use in evaluation
    splits = {'train': np.array(tr_split), 'val': np.array(val_split), 'test': np.array(ts_split)}
    np.save('./splits.npy', splits)

    dataloaders = build_train_dataloaders(args, tr_split, val_split,
                                           tr_transforms, val_ts_transforms)

    # TwoStreamI3D.forward returns log-probabilities (fusion happens in probability space), so
    # it pairs with NLLLoss rather than CrossEntropyLoss (which expects raw logits).
    if args.model_type == 'i3d_two_stream':
        loss_func = nn.NLLLoss(reduction='sum')
    else:
        loss_func = nn.CrossEntropyLoss(reduction='sum')

    opt = optim.Adam(model.parameters(), lr=args.learning_rate)
    lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    os.makedirs("./models", exist_ok=True)

    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))

    model.to(device)
    train(dataloaders, model, loss_func, opt, lr_scheduler, device, './models',
          args.n_epochs, use_wandb=use_wandb)

    if use_wandb:
        wandb.finish()


def run_eval(args, model, device, val_ts_transforms):
    """
    Eval mode: load the saved test split, build a test dataloader, load the checkpoint, and
    report overall test accuracy.
    """
    splits = np.load('./splits.npy', allow_pickle=True)
    ts_split = splits.item()['test']
    ts_split = [(sample[0], int(sample[1])) for sample in ts_split]

    if args.model_type == 'i3d_two_stream':
        ts_dataset = TwoStreamVideoDataset(ts_split, args.flow_dir, args.fr_per_vid,
                                            val_ts_transforms, val_ts_transforms)
        dataloaders = {'test': DataLoader(ts_dataset, batch_size=2 * args.batch_size,
                                           shuffle=False, collate_fn=collate_fn_two_stream)}
    else:
        ts_dataset = VideoDataset(ts_split, args.fr_per_vid, val_ts_transforms)
        dataloaders = test_dloaders(ts_dataset, args.batch_size, args.model_type)

    model.load_state_dict(torch.load(args.ckpt))
    model.to(device)
    _, _, accuracy = test(model, dataloaders['test'], device)

    print(f'The overall test accuracy is {100 * accuracy:.4f}%.')
    # For a detailed per-class breakdown, capture (targets, outputs) from test() above and pass
    # them to get_test_report / get_confusion_matrix from test.py.


def run_preprocess_flow(args):
    """
    Preprocess-flow mode: for every video already extracted as RGB frames under args.frame_dir,
    compute optical flow between its frames and write the result to args.flow_dir, mirroring
    the class/video subfolder structure so TwoStreamVideoDataset can pair the two directories up.
    This is a one-off step to run before training/evaluating model_type=i3d_two_stream.

    Idempotent/resumable: a video is skipped if its flow output directory already contains the
    same number of frames as its RGB input, so re-running this (in the same session, or after a
    session restart) does not redo work that already finished.
    """
    for vid_cat in sorted(os.listdir(args.frame_dir)):
        cat_path = os.path.join(args.frame_dir, vid_cat)
        if not os.path.isdir(cat_path):
            continue
        for vid in tqdm(sorted(os.listdir(cat_path)), desc=vid_cat):
            vid_path = os.path.join(cat_path, vid)
            fr_paths = sorted(glob.glob(vid_path + '/*.jpg'))
            if len(fr_paths) < 2:
                continue

            out_dir = os.path.join(args.flow_dir, vid_cat, vid)
            existing = glob.glob(out_dir + '/*.jpg')
            if len(existing) == len(fr_paths):
                # Flow already computed for this video (same frame count as the RGB clip) --
                # skip so re-running this step doesn't redo already-finished work.
                continue

            frames = [np.array(Image.open(p).convert('RGB')) for p in fr_paths]
            flow_frames = compute_flow_frames(frames)

            os.makedirs(out_dir, exist_ok=True)
            store_frames(flow_frames, out_dir)


def main(args):
    """
    Main function to execute training, evaluation, or flow preprocessing based on the parsed
    command-line arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    if args.mode == 'preprocess_flow':
        # Flow preprocessing needs no model/transforms, just frame_dir -> flow_dir.
        run_preprocess_flow(args)
        return

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load transformation statistics and create data augmentation transforms
    h, w, mean, std = transform_stats(args.model_type)
    tr_transforms, val_ts_transforms = compose_data_transforms(h, w, mean, std)

    model = build_model(args)

    if args.mode == 'train':
        run_train(args, model, device, tr_transforms, val_ts_transforms)
    elif args.mode == 'eval':
        run_eval(args, model, device, val_ts_transforms)
    else:
        raise ValueError("The mode argument must be 'train', 'eval', or 'preprocess_flow'.")


if __name__ == "__main__":
    cli_args = args_parser()
    main(cli_args)
