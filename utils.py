"""
Module: utils.py

This module provides helper functions for video processing and data transformations
for video classification tasks. It includes functions for:
    - Uniformly sampling frames from videos.
    - Storing extracted frames as JPEG images.
    - Retrieving image transformation statistics based on the model type.
    - Composing data transforms for training and validation/test datasets.
    - Creating DataLoaders for training, validation, and testing, using custom collate functions.
"""

import os
import cv2
import numpy as np

from torchvision import transforms
from torch.utils.data import DataLoader
from video_datasets import collate_fn_r3d_18, collate_fn_rnn


def get_frames(vid, n_frames=1):
    """
    Uniformly sample frames from a video file.

    Args:
        vid (str): Path to the video file.
        n_frames (int): Number of frames to sample from the video.

    Returns:
        tuple: (frames, v_len)
            - frames (list): List of sampled frames (as numpy arrays in RGB format).
            - v_len (int): Total number of frames in the video.

    Notes:
        - If the video cannot be opened or contains no frames, an empty list and 0 are returned.
        - Frames are sampled at uniformly spaced indices.
    """
    frames = []
    v_cap = cv2.VideoCapture(vid)  # pylint: disable=no-member
    if not v_cap.isOpened():
        print("Failed to open video:", vid)
        return frames, 0
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))  # pylint: disable=no-member
    if v_len <= 0:
        print("No frames found in video:", vid)
        v_cap.release()
        return frames, 0

    # FIX: use int64 to avoid overflow
    frame_idx = np.linspace(0, v_len - 1, n_frames, dtype=np.int64)
    frame_set = set(frame_idx)
    for idx in range(v_len):
        success, frame = v_cap.read()
        if not success:
            continue
        if idx in frame_set:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # pylint: disable=no-member
            frames.append(frame)
    v_cap.release()
    return frames, v_len


def store_frames(frames, store_path):
    """
    Save a list of frames as JPEG images to the specified directory.

    Each frame is converted from RGB to BGR format (as expected by OpenCV)
    before saving.

    Args:
        frames (list): List of frames (numpy arrays in RGB format) to save.
        store_path (str): Directory path where the frames will be stored.

    Returns:
        None
    """
    for idx, frame in enumerate(frames):
        print("processing")
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # pylint: disable=no-member
        path_to_frame = os.path.join(store_path, f"frame{idx:04d}.jpg")
        cv2.imwrite(path_to_frame, frame)  # pylint: disable=no-member

def transform_stats(model='lrcn'):
    """
    Retrieve transformation statistics based on the model type.

    For the 'lrcn' model, images are resized to 224x224; for '3dcnn', images are resized to 112x112.
    Also returns the mean and standard deviation values used for normalization.

    Args:
        model (str): Type of model ('lrcn' or '3dcnn').

    Returns:
        tuple: (h, w, mean, std)
            - h (int): Image height.
            - w (int): Image width.
            - mean (list): Mean values for normalization.
            - std (list): Standard deviation values for normalization.

    Raises:
        ValueError: If an undefined model type is provided.
    """
    if model == 'lrcn':
        h, w = 224, 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif model == '3dcnn':
        h, w = 112, 112
        mean = [0.43216, 0.394666, 0.37645]
        std = [0.22803, 0.22145, 0.216989]
    else:
        raise ValueError('model_type arg is undefined....')
    return h, w, mean, std


def compose_data_transforms(height, width, mean, std):
    """
    Compose and return data transforms for training and validation/test datasets.

    The training transforms include data augmentation such as random horizontal flipping
    and random affine transformations, while the validation/test transforms consist solely
    of resizing, converting to tensor, and normalizing.

    IMPROVEMENT: a full 35-epoch training run showed severe overfitting (train loss
    0.31 vs val loss 1.93 by the final epoch) with only flip + small affine
    augmentation. Training transforms now also include color jitter and a
    random-resized crop, which forces the model to rely on more than a few
    memorized pixel patterns per class. Validation/test transforms remain
    deterministic (resize, tensor, normalize only) so evaluation is reproducible.

    Args:
        height (int): Desired image height.
        width (int): Desired image width.
        mean (list): Mean values for normalization.
        std (list): Standard deviation values for normalization.

    Returns:
        tuple: (train_transforms, val_test_transforms)
            - train_transforms: Composed transforms for the training set.
            - val_test_transforms: Composed transforms for the validation/test set.
    """
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop((height, width), scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_test_transforms = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_transforms, val_test_transforms


def train_val_dloaders(train_dataset, val_dataset, batch_size, model='lrcn', num_workers=2):
    """
    Create DataLoaders for training and validation datasets.

    Selects the appropriate collate function based on the model type.
    For 'lrcn' (RNN-based models), uses collate_fn_rnn which pads sequences to equal lengths.
    Otherwise, uses collate_fn_r3d_18 for 3D CNN models.

    PERFORMANCE FIX: the original DataLoaders used the default num_workers=0, meaning
    every frame image is opened and transformed on the main process, blocking the GPU
    from getting new batches while it waits on disk/Drive I/O. Setting num_workers > 0
    lets multiple worker processes prefetch and transform frames in parallel with GPU
    compute; pin_memory speeds up the host-to-GPU transfer for CUDA devices.

    Args:
        train_dataset (Dataset): PyTorch Dataset for training data.
        val_dataset (Dataset): PyTorch Dataset for validation data.
        batch_size (int): Number of samples per batch.
        model (str): Model type; 'lrcn' for RNN-based models, otherwise for 3D CNNs.
        num_workers (int): Number of subprocesses used for data loading.

    Returns:
        dict: Dictionary with keys 'train' and 'val' mapping to their respective DataLoaders.
    """
    collate = collate_fn_rnn if model == "lrcn" else collate_fn_r3d_18
    train_dl = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate,
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
    )
    val_dl = DataLoader(
        val_dataset, batch_size=2 * batch_size, shuffle=False, collate_fn=collate,
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
    )
    return {"train": train_dl, "val": val_dl}


def test_dloaders(test_dataset, batch_size, model='lrcn', num_workers=2):
    """
    Create a DataLoader for the test dataset.

    Selects the appropriate collate function based on the model type.
    For 'lrcn' models, uses collate_fn_rnn; otherwise, uses collate_fn_r3d_18.

    PERFORMANCE FIX: see train_val_dloaders -- num_workers/pin_memory added so test
    evaluation isn't bottlenecked on single-process frame I/O either.

    Args:
        test_dataset (Dataset): PyTorch Dataset for test data.
        batch_size (int): Number of samples per batch.
        model (str): Model type; 'lrcn' for RNN-based models, otherwise for 3D CNNs.
        num_workers (int): Number of subprocesses used for data loading.

    Returns:
        dict: Dictionary with key 'test' mapping to the test DataLoader.
    """
    collate = collate_fn_rnn if model == "lrcn" else collate_fn_r3d_18
    test_dl = DataLoader(
        test_dataset, batch_size=2 * batch_size, shuffle=False, collate_fn=collate,
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
    )
    return {"test": test_dl}

def compose_dataloaders(train_dataset, val_dataset, test_dataset, batch_size, model="lrcn"):
    """
    FIX: this function was imported by run_training.py but never defined anywhere
    in the original codebase, silently breaking that entry point. Implemented here.
    """
    dloaders = train_val_dloaders(train_dataset, val_dataset, batch_size, model)
    dloaders.update(test_dloaders(test_dataset, batch_size, model))
    return dloaders
