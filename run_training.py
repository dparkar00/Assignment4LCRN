"""
Module: trainer.py

This module provides an end-to-end training pipeline for a video classification model.
It parses command-line arguments to configure dataset paths, model parameters, and training
hyperparameters. The module performs the following steps:

1. Loads the video dataset from a specified directory.
2. Splits the dataset into training, validation, and test sets using stratified sampling.
3. Applies data augmentation transforms and creates PyTorch Datasets and DataLoaders using a custom
   function `compose_dataloaders`.
4. Initializes a video classification model (e.g., LRCN) with specified hyperparameters.
5. Sets up the loss function, optimizer, and learning rate scheduler.
6. Executes the training procedure and saves the best model weights.

Usage:
    Run the script from the command line with the required arguments. For example:

        python trainer.py -fd /path/to/frames -nc 51 -bs 8

Arguments include:
    -fd/--frame_dir: Directory where video frames are stored (required).
    -trs/--train_size: Proportion of data for training (default: 0.7).
    -tss/--test_size: Proportion of data for testing (default: 0.1).
    -fpv/--fr_per_vid: Number of frames per video to use (default: 16).
    -nc/--n_classes: Number of classes for classification (required).
    -mt/--model_type: Type of model to use, e.g., 'lrcn' (default: 'lrcn').
    -cnn/--cnn_backbone: CNN backbone for feature extraction (default: 'resnet34').
    -p/--pretrained: Whether to use a pretrained CNN backbone (default: True).
    -rhs/--rnn_hidden_size: Number of neurons in the RNN/LSTM hidden layer (default: 100).
    -rnl/--rnn_n_layers: Number of RNN/LSTM layers (default: 1).
    -bs/--batch_size: Mini-batch size (required).
    -d/--dropout: Dropout rate for regularization (default: 0.1).
    -lr/--learning_rate: Learning rate for training (default: 3e-5).
    -ne/--n_epochs: Number of training epochs (default: 30).
"""

# pylint: disable=duplicate-code
# run.py and run_training.py intentionally share a similar argparse setup
# (two working entry points, per project requirements); the overlap is
# expected, not an accidental duplication bug.

import os
import argparse

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import wandb

from video_datasets import VideoDataset, load_dataset, dataset_split
from utils import transform_stats, compose_data_transforms, compose_dataloaders
from models import LRCN, R3DClassifier, TwoStreamR3D
from train import train

def str2bool(value):
    """
    FIX: argparse's `default=True` with no `type=` meant --pretrained False stored
    the *string* "False", which is truthy, so this flag could never be disabled from
    the CLI. This helper parses common boolean string forms correctly.
    """
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def args_parser():
    """
    Parse command-line arguments for configuring the video classification training pipeline.

    Returns:
        argparse.Namespace: Parsed command-line arguments.

    Required Arguments:
        -fd/--frame_dir: Directory for storing video frames.
        -nc/--n_classes: Number of classes for the classification task.
        -bs/--batch_size: Mini-batch size.

    Optional Arguments (with defaults):
        -trs/--train_size: Proportion of the dataset used for training (default: 0.7).
        -tss/--test_size: Proportion of the dataset used for testing (default: 0.1).
        -fpv/--fr_per_vid: Number of frames per video (default: 16).
        -mt/--model_type: Model type, e.g., 'lrcn' (default: 'lrcn').
        -cnn/--cnn_backbone: CNN backbone for feature extraction (default: 'resnet34').
        -p/--pretrained: Use pretrained CNN backbone (default: True).
        -rhs/--rnn_hidden_size: Number of neurons in the RNN/LSTM hidden layer (default: 100).
        -rnl/--rnn_n_layers: Number of RNN/LSTM layers (default: 1).
        -d/--dropout: Dropout rate (default: 0.1).
        -lr/--learning_rate: Learning rate for training (default: 3e-5).
        -ne/--n_epochs: Number of training epochs (default: 30).
    """
    parser = argparse.ArgumentParser(description="Video Classification Training")

    parser.add_argument(
        "-fd", "--frame_dir", help="Directory for storing video frames", required=True
    )
    parser.add_argument("-trs", "--train_size", type=float, default=0.7, help="Train set size")
    parser.add_argument("-tss", "--test_size", type=float, default=0.1, help="Test set size")
    parser.add_argument(
        "-fpv", "--fr_per_vid", type=int, default=16, help="Number of frames per video"
    )
    parser.add_argument("-nc", "--n_classes", type=int, required=True, help="Number of classes")

    parser.add_argument(
        "-mt", "--model_type", default="lrcn",
        help="Model type: 'lrcn', '3dcnn' (R3D-18), or 'i3d' (two-stream RGB+flow)",
    )
    parser.add_argument(
        "-cnn", "--cnn_backbone", default="resnet34",
        help="2D CNN backbone - options: resnet18, resnet34, resnet50, resnet101, resnet152",
    )
    parser.add_argument(
        "-p", "--pretrained", type=str2bool, default=True, help="Use pretrained CNN backbone"
    )
    parser.add_argument("-rhs", "--rnn_hidden_size", type=int, default=100, help="LSTM hidden size")
    parser.add_argument("-rnl", "--rnn_n_layers", type=int, default=1, help="Number of LSTM layers")

    # Model-level improvements (see README for rationale).
    parser.add_argument(
        "-bi", "--bidirectional", type=str2bool, default=False,
        help="Use a bidirectional LSTM (model improvement)",
    )
    parser.add_argument(
        "-att", "--use_attention", type=str2bool, default=False,
        help="Use attention pooling over LSTM timestep outputs instead of only the "
             "final hidden state (model improvement)",
    )
    parser.add_argument(
        "-cls", "--use_conv_lstm", type=str2bool, default=False,
        help="Replace the standard LSTM with a ConvLSTM operating directly on CNN "
             "spatial feature maps, weaving recurrence throughout the model instead "
             "of applying it only after full pooling (model improvement). Mutually "
             "exclusive with --bidirectional/--use_attention.",
    )
    parser.add_argument(
        "-clsh", "--conv_lstm_hidden_channels", type=int, default=128,
        help="Number of hidden channels in the ConvLSTM cell (only used with "
             "--use_conv_lstm)",
    )
    parser.add_argument(
        "-fbu", "--freeze_backbone_until", type=int, default=None,
        help="Number of trailing ResNet child modules to keep trainable; earlier "
             "layers are frozen (model improvement, partial fine-tuning)",
    )
    parser.add_argument(
        "-fcd", "--flow_cache_dir", type=str, default="./flow_cache",
        help="PERFORMANCE IMPROVEMENT: directory to cache computed optical flow "
             "(only used with --model_type i3d). Flow is deterministic per clip, "
             "so caching it avoids recomputing it from scratch every epoch.",
    )

    parser.add_argument("-bs", "--batch_size", type=int, required=True, help="Mini-batch size")
    parser.add_argument(
        "-d", "--dropout", type=float, default=0.1, help="Dropout rate for regularization"
    )
    parser.add_argument("-lr", "--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument(
        "-blrf", "--backbone_lr_factor", type=float, default=1.0,
        help="Multiplier applied to --learning_rate for backbone (CNN) parameters "
             "only, e.g. 0.1 gives the backbone 1/10th the LR of the newly "
             "initialized head (model improvement -- prevents an LR tuned for "
             "fresh layers from being too aggressive for pretrained weights)",
    )
    parser.add_argument("-ne", "--n_epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument(
        "-lrp", "--lr_patience", type=int, default=5,
        help="Epochs with no val loss improvement before the LR scheduler reduces "
             "the learning rate (ReduceLROnPlateau patience)",
    )

    parser.add_argument(
        "-wb", "--wandb_project", type=str, default="hmdb51-lrcn", help="W&B project name"
    )
    parser.add_argument("-run", "--run_name", type=str, default=None, help="W&B run name")

    return parser.parse_args()

# pylint: disable=too-many-locals
def trainer(args):
    """
    Execute the training pipeline for video classification.

    This function performs the following steps:
      1. Loads the video dataset from the specified frame directory.
      2. Splits the dataset into training, validation, and test sets.
      3. Loads image transformation statistics and composes data augmentation transforms.
      4. Creates PyTorch Datasets for training, validation, and test splits.
      5. Constructs DataLoaders using a custom function `compose_dataloaders`.
      6. Initializes the LRCN model with specified hyperparameters.
      7. Sets up the loss function, optimizer, and learning rate scheduler.
      8. Executes the training loop and saves the best model weights.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    # Dataset parameters
    frame_dir = args.frame_dir
    tr_size = args.train_size
    ts_size = args.test_size
    fr_per_vid = args.fr_per_vid
    n_classes = args.n_classes

    # Model parameters
    model_type = args.model_type
    rnn_hidden_size = args.rnn_hidden_size
    rnn_n_layers = args.rnn_n_layers
    dropout = args.dropout
    pretrained = args.pretrained
    cnn_backbone = args.cnn_backbone

    # Training parameters
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    learning_rate = args.learning_rate
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))

    # Load dataset and generate train/validation/test splits
    # FIX: load_dataset/dataset_split now return group keys derived from
    # source-video identity, and dataset_split performs a group-aware stratified
    # split so clips from the same source video can never leak across splits.
    vid_paths, vid_labels, vid_groups, _label_dict = load_dataset(frame_dir)
    tr_split, val_split, ts_split = dataset_split(
        vid_paths, vid_labels, vid_groups, tr_size, ts_size
    )
    # Load image transformation statistics and compose data augmentation transforms
    # Model improvement: 'i3d' (two-stream RGB+flow) reuses the '3dcnn' resolution/
    # normalization stats for its RGB stream -- optical flow normalization is
    # handled separately inside VideoDataset/_compute_flow_stack, not through
    # this transform pipeline, so no separate branch is needed here.
    transform_model_type = '3dcnn' if model_type == 'i3d' else model_type
    h, w, mean, std = transform_stats(transform_model_type)
    tr_transforms, val_ts_transforms = compose_data_transforms(h, w, mean, std)

    # Create PyTorch Datasets for each split
    # Model improvement: compute_flow=True (only for --model_type i3d) makes
    # VideoDataset also compute and concatenate optical flow channels onto
    # each RGB frame -- see video_datasets.VideoDataset/_compute_flow_stack.
    tr_dataset = VideoDataset(
        tr_split, fr_per_vid, tr_transforms,
        compute_flow=(model_type == 'i3d'), flow_size=(h, w),
            flow_cache_dir=args.flow_cache_dir if model_type == 'i3d' else None,
    )
    val_dataset = VideoDataset(
        val_split, fr_per_vid, val_ts_transforms,
        compute_flow=(model_type == 'i3d'), flow_size=(h, w),
            flow_cache_dir=args.flow_cache_dir if model_type == 'i3d' else None,
    )
    ts_dataset = VideoDataset(
        ts_split, fr_per_vid, val_ts_transforms,
        compute_flow=(model_type == 'i3d'), flow_size=(h, w),
            flow_cache_dir=args.flow_cache_dir if model_type == 'i3d' else None,
    )

    # Compose DataLoaders for training, validation, and test using a custom function
    dataloaders = compose_dataloaders(tr_dataset, val_dataset, ts_dataset, batch_size, model_type)

    # Initialize the LRCN model with the specified parameters
    # FIX: --model_type was documented and accepted as a CLI arg (including a
    # '3dcnn' option) but never actually used to select which model gets built --
    # LRCN was constructed unconditionally regardless of --model_type. Model
    # improvement: R3DClassifier (Kinetics-400-pretrained 3D CNN) is now wired up
    # as a real alternative when --model_type 3dcnn is passed.
    if model_type == '3dcnn':
        model = R3DClassifier(
            n_classes=n_classes, pretrained=pretrained,
            freeze_backbone_until=args.freeze_backbone_until,
        )
    elif model_type == 'i3d':
        model = TwoStreamR3D(
            n_classes=n_classes, pretrained=pretrained,
            freeze_backbone_until=args.freeze_backbone_until,
        )
    else:
        model = LRCN(
            hidden_size=rnn_hidden_size,
            n_layers=rnn_n_layers,
            dropout_rate=dropout,
            n_classes=n_classes,
            pretrained=pretrained,
            cnn_model=cnn_backbone,
            freeze_backbone_until=args.freeze_backbone_until,
            bidirectional=args.bidirectional,
            use_attention=args.use_attention,
            use_conv_lstm=args.use_conv_lstm,
            conv_lstm_hidden_channels=args.conv_lstm_hidden_channels,
        )
    model = model.to(device)

    # Define the loss function, optimizer, and learning rate scheduler
    # Model improvement: label smoothing softens the target distribution (instead
    # of a hard one-hot target), discouraging the model from becoming
    # overconfident on training examples it has memorized.
    loss_func = nn.CrossEntropyLoss(reduction='sum', label_smoothing=0.1)
    # Model improvement: differential learning rates -- see run.py for full rationale.
    backbone_param_ids = {id(p) for p in model.base_model.parameters()}
    seen_ids, backbone_params, head_params = set(), [], []
    for param in model.parameters():
        if not param.requires_grad or id(param) in seen_ids:
            continue
        seen_ids.add(id(param))
        if id(param) in backbone_param_ids:
            backbone_params.append(param)
        else:
            head_params.append(param)
    opt = optim.Adam(
        [
            {"params": backbone_params, "lr": learning_rate * args.backbone_lr_factor},
            {"params": head_params, "lr": learning_rate},
        ],
        weight_decay=1e-4,
    )
    lr_scheduler = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=args.lr_patience)
    os.makedirs("./models", exist_ok=True)
    optim_model_dir = './models'

    # Execute the main training procedure and update the model
    model, _loss_hist, _acc_hist = train(
            dataloaders, model, loss_func, opt, lr_scheduler, device, optim_model_dir, n_epochs
        )
    return model

if __name__ == "__main__":
    trainer(args_parser())
