"""
Module: run.py

This module is the main entry point for training or evaluating a video classification
model. It uses command-line arguments to configure the experiment, including dataset
paths, model parameters, and training hyperparameters. Depending on the selected mode
('train' or 'eval'), it performs the following:

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
import argparse

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
import wandb

from models import LRCN
# NOTE: this project's own test.py module shares its name with Python's stdlib
# "test" package, which is what triggers pylint's import-order warning below for
# these two lines; the imports still resolve correctly to the local module.
from test import get_confusion_matrix, get_test_report  # pylint: disable=wrong-import-order
from test import test as evaluate  # pylint: disable=wrong-import-order
from train import train
from utils import compose_data_transforms, test_dloaders, train_val_dloaders, transform_stats
from video_datasets import VideoDataset, dataset_split, load_dataset

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
    Parse command-line arguments for configuring the video classification training or evaluation.

    Returns:
        argparse.Namespace: Parsed command-line arguments.

    Arguments include:
        -fd/--frame_dir: Directory for storing video frames.
        -trs/--train_size: Proportion of data to use for training (default 0.7).
        -tss/--test_size: Proportion of data to use for testing (default 0.1).
        -fpv/--fr_per_vid: Number of frames per video to consider (default 16).
        -nc/--n_classes: Number of classes for the classification task (required).
        -c/--ckpt: Path for loading a trained model checkpoint.
        -mt/--model_type: Model type, either '3dcnn' or 'lrcn' (default 'lrcn').
        -cnn/--cnn_backbone: Backbone CNN for 2D feature extraction (default 'resnet34').
        -p/--pretrained: Whether to use a pretrained CNN backbone (default True).
        -rhs/--rnn_hidden_size: Number of neurons in the RNN/LSTM hidden layer (default 100).
        -rnl/--rnn_n_layers: Number of RNN/LSTM layers (default 1).
        -m/--mode: Mode of operation: 'train' or 'eval' (required).
        -bs/--batch_size: Mini-batch size (required).
        -d/--dropout: Dropout rate for regularization (default 0.1).
        -lr/--learning_rate: Learning rate for training (default 3e-5).
        -ne/--n_epochs: Number of training epochs (default 30).
    """
    parser = argparse.ArgumentParser(description="Video Classification Training")

    parser.add_argument("-fd", "--frame_dir", help="Directory for storing video frames")
    parser.add_argument("-trs", "--train_size", type=float, default=0.7, help="Train set size")
    parser.add_argument("-tss", "--test_size", type=float, default=0.1, help="Test set size")
    parser.add_argument(
        "-fpv", "--fr_per_vid", type=int, default=16, help="Number of frames per video"
    )
    parser.add_argument("-nc", "--n_classes", type=int, required=True, help="Number of classes")

    parser.add_argument("-c", "--ckpt", help="Path for loading trained model checkpoints")
    parser.add_argument("-mt", "--model_type", default="lrcn", help="Model type")
    parser.add_argument(
        "-cnn", "--cnn_backbone", default="resnet34",
        help="2D CNN backbone: resnet18, resnet34, resnet50, resnet101, resnet152",
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
        "-fbu", "--freeze_backbone_until", type=int, default=None,
        help="Number of trailing ResNet child modules to keep trainable; earlier "
             "layers are frozen (model improvement, partial fine-tuning)",
    )

    parser.add_argument(
        "-m", "--mode", type=str, default="train", required=True, help="'train' or 'eval'"
    )
    parser.add_argument("-bs", "--batch_size", type=int, required=True, help="Mini-batch size")
    parser.add_argument("-d", "--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("-lr", "--learning_rate", type=float, default=3e-5, help="Learning rate")
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

# pylint: disable=too-many-locals,too-many-statements
def main(args):
    """
    Main function to execute training or evaluation based on the parsed command-line arguments.

    For training:
        - Loads the dataset from the specified frame directory.
        - Splits the dataset into training, validation, and test sets.
        - Saves the splits for later use.
        - Creates the training and validation DataLoaders.
        - Initializes the model (LRCN) with specified hyperparameters.
        - Defines the loss function, optimizer, and learning rate scheduler.
        - Runs the training loop and saves the best model weights.

    For evaluation:
        - Loads the saved dataset splits.
        - Creates the test DataLoader.
        - Loads the trained model checkpoint.
        - Evaluates the model on the test set and prints the overall test accuracy.

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
    mode = args.mode
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    learning_rate = args.learning_rate
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        config=vars(args),
    )
    # Load transformation statistics and create data augmentation transforms
    h, w, mean, std = transform_stats(model_type)
    tr_transforms, val_ts_transforms = compose_data_transforms(h, w, mean, std)

    # Initialize the model (LRCN)
    model = LRCN(
        hidden_size=rnn_hidden_size, n_layers=rnn_n_layers, dropout_rate=dropout,
        n_classes=n_classes, pretrained=pretrained, cnn_model=cnn_backbone,
        freeze_backbone_until=args.freeze_backbone_until,
        bidirectional=args.bidirectional,
    )

    if mode == 'train':
        # FIX: load_dataset/dataset_split now return group keys derived from
        # source-video identity, and dataset_split performs a group-aware stratified
        # split so clips from the same source video can never leak across splits.
        # Load dataset and split into train/validation/test
        vid_paths, vid_labels, vid_groups, label_dict = load_dataset(frame_dir)
        tr_split, val_split, ts_split = dataset_split(
            vid_paths, vid_labels, vid_groups, tr_size, ts_size
        )

        # Save the splits for reproducibility and later use in evaluation
        splits = {'train': np.array(tr_split),
                  'val': np.array(val_split),
                  'test': np.array(ts_split)}
        np.save('./splits.npy', splits)

        # Create PyTorch Datasets and DataLoaders for train and validation
        tr_dataset = VideoDataset(tr_split, fr_per_vid, tr_transforms)
        val_dataset = VideoDataset(val_split, fr_per_vid, val_ts_transforms)
        dataloaders = train_val_dloaders(tr_dataset, val_dataset, batch_size, model_type)

        # Define the loss function, optimizer, and learning rate scheduler
        loss_func = nn.CrossEntropyLoss(reduction='sum')
        # Model improvement: weight decay for regularization.
        opt = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        # FIX: verbose=1 is invalid/deprecated for ReduceLROnPlateau in recent
        # PyTorch versions (expects bool, and is removed entirely in newest versions).
        lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=args.lr_patience)
        os.makedirs("./models", exist_ok=True)
        optim_model_dir = './models'

        # Main training procedure
        model.to(device)
        model, _loss_hist, _acc_hist = train(
            dataloaders, model, loss_func, opt, lr_scheduler, device, optim_model_dir, n_epochs
        )
        # Evaluate on held-out test set immediately after training and log to W&B.
        ts_dataset = VideoDataset(ts_split, fr_per_vid, val_ts_transforms)
        test_dataloaders = test_dloaders(ts_dataset, batch_size, model_type)
        targets, outputs, test_accuracy = evaluate(model, test_dataloaders["test"], device)
        print(f"Final test accuracy: {100 * test_accuracy:.4f}%")
        wandb.summary["test_accuracy"] = test_accuracy

        all_cats = sorted(label_dict, key=label_dict.get)
        report = get_test_report(targets, outputs, all_cats)
        wandb.log({"test/accuracy": test_accuracy})
        wandb.summary["classification_report"] = report
        # FIX: main() must return consistently across branches (pylint
        # inconsistent-return-statements) -- the eval branch returns a confusion
        # matrix, so the train branch explicitly returns None here.
        return None

    if mode == 'eval':
        # Load saved dataset splits
        splits = np.load('./splits.npy', allow_pickle=True)
        ts_split = splits.item()['test']
        ts_split = [(sample[0], int(sample[1])) for sample in ts_split]

        # Create PyTorch Dataset and DataLoader for the test set
        ts_dataset = VideoDataset(ts_split, fr_per_vid, val_ts_transforms)
        dataloaders = test_dloaders(ts_dataset, batch_size, model_type)

        # Load the trained model checkpoint
        model.load_state_dict(torch.load(args.ckpt))
        model.to(device)
        targets, outputs, accuracy = evaluate(model, dataloaders["test"], device)

        print(f'The overall test accuracy is {100 * accuracy:.4f}%.')
        # Optionally, generate a detailed test report or confusion matrix:
        # print(get_test_report(targets, outputs, all_cats))
        # print(get_confusion_matrix(targets, outputs, labels_dict, all_cats))
        wandb.summary["test_accuracy"] = accuracy

        all_cats = sorted(label_dict, key=label_dict.get)
        report = get_test_report(targets, outputs, all_cats)
        conf_mat = get_confusion_matrix(targets, outputs, label_dict, all_cats)
        wandb.summary["classification_report"] = report
        print(report)
        return conf_mat

    raise ValueError('The mode argument must be either "train" or "eval".')

if __name__ == "__main__":
    main(args_parser())
