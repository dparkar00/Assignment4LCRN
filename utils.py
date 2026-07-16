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
# pylint: disable=no-member
# cv2 is a compiled C extension; pylint cannot introspect its members (cv2.VideoCapture,
# cv2.cvtColor, etc.), so it flags every real cv2 attribute access as a false positive.

import os
import glob
import cv2
import numpy as np
import torch
from PIL import Image

from torchvision import transforms
from torch.utils.data import DataLoader
from video_datasets import collate_fn_r3d_18, collate_fn_rnn

# Parallel data loading matters a LOT here: video clips are loaded frame-by-frame from disk as
# individual JPEGs (doubled for two-stream, since RGB and flow are each loaded separately), which
# is I/O- and CPU-bound. With num_workers=0, this happens serially on one thread while the GPU
# sits idle waiting -- for two-stream I3D, back-of-envelope FLOP math suggests actual observed
# iteration time can run an order of magnitude past what GPU compute alone would predict, which
# points at data loading, not the model, as the dominant cost. This scales workers to the actual
# machine instead of a fixed guess; if Colab only grants 2 CPU cores, this won't manufacture more
# parallelism than exists, but it won't leave workers on the table either.
NUM_WORKERS = max(1, (os.cpu_count() or 2) - 1)
PIN_MEMORY = True


def _seed_worker(_worker_id):
    """
    Reseed NumPy's global RNG for a DataLoader worker process.

    PyTorch's default worker initialization reseeds torch's RNG and Python's built-in random
    module with a distinct, well-separated seed per worker -- but explicitly does NOT reseed
    NumPy's global RNG (this is a documented PyTorch behavior, not an oversight on their part).
    Since video_datasets._sample_frame_indices uses np.random.randint for temporal-sampling
    augmentation, and worker processes are forked from the same parent state, different workers
    could otherwise share near-identical initial NumPy random state -- producing correlated or
    duplicate "random" frame choices across workers instead of genuinely independent ones,
    quietly undermining the augmentation. torch.initial_seed() returns the per-worker seed
    PyTorch already assigned (base_seed + worker_id), so reusing it here keeps every worker's
    NumPy state distinct too.

    Args:
        _worker_id (int): DataLoader worker id (unused directly; part of the required
                           worker_init_fn signature).
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


def dataloader_kwargs():
    """
    Extra DataLoader kwargs for parallel, prefetching data loading, centralized here so every
    DataLoader in this project uses the same, valid combination. persistent_workers keeps worker
    processes alive between epochs instead of respawning them each time (meaningful with many
    short epochs); prefetch_factor keeps more batches queued ahead of the GPU so it's less
    likely to stall. worker_init_fn fixes NumPy's per-worker RNG correlation (see _seed_worker).
    persistent_workers, prefetch_factor, and worker_init_fn are only valid when num_workers > 0.

    Returns:
        dict: Kwargs to unpack into a DataLoader(...) call.
    """
    kwargs = {'num_workers': NUM_WORKERS, 'pin_memory': PIN_MEMORY}
    if NUM_WORKERS > 0:
        kwargs['persistent_workers'] = True
        kwargs['prefetch_factor'] = 4
        kwargs['worker_init_fn'] = _seed_worker
    return kwargs


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
    v_cap = cv2.VideoCapture(vid)
    if not v_cap.isOpened():
        print("Failed to open video:", vid)
        return frames, 0
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if v_len <= 0:
        print("No frames found in video:", vid)
        v_cap.release()
        return frames, 0
    frame_idx = np.linspace(0, v_len-1, n_frames, dtype=np.int64)
    for idx in range(v_len):
        success, frame = v_cap.read()
        if not success:
            continue
        if idx in frame_idx:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    v_cap.release()
    return frames, v_len


def compute_flow_frames(frames):
    """
    Compute dense optical flow (Farneback) between consecutive RGB frames and encode each flow
    field as a 3-channel pseudo-RGB image (x-flow, y-flow, magnitude), so the flow clip has the
    same (T, 3, H, W) shape as the RGB clip and can reuse the same RGB-pretrained I3D stem.

    Args:
        frames (list): List of RGB frames (numpy arrays in RGB format), as returned by get_frames.

    Returns:
        list: List of 3-channel flow frames (numpy arrays, dtype uint8), same length as the input.
              The first output frame duplicates the first computed flow field, since there is no
              flow defined before frame 0.
    """
    if len(frames) < 2:
        return [np.zeros_like(f) for f in frames]

    gray_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    flow_frames = []
    for i in range(1, len(gray_frames)):
        flow = cv2.calcOpticalFlowFarneback(
            gray_frames[i - 1], gray_frames[i], None,
            pyr_scale=0.5, levels=3, winsize=15, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0
        )
        fx = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
        fy = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        flow_img = np.stack([fx, fy, mag], axis=-1).astype(np.uint8)
        flow_frames.append(flow_img)

    # Duplicate the first computed flow field so the flow clip matches the RGB clip's length.
    return [flow_frames[0]] + flow_frames


def preprocess_video_flow(vid_path, out_dir):
    """
    Compute and store optical flow frames for a single video, skipping it if flow has already
    been computed (out_dir already has the same frame count as vid_path's RGB frames).

    This is a per-video unit of work, designed to be dispatched to a worker process by
    run.py's preprocess_flow mode (see ProcessPoolExecutor there) so flow computation across
    many videos can run in parallel across CPU cores instead of one video at a time. Optical
    flow between two consecutive frames of one video is independent of every other video, so
    this parallelizes cleanly with no correctness risk. cv2's own internal multi-threading is
    disabled here since many of these run concurrently in separate processes -- without this,
    each process would try to use all available cores for its own single video, oversubscribing
    the machine instead of letting the process pool divide the cores across videos.

    Args:
        vid_path (str): Path to the video's RGB frame directory.
        out_dir (str): Path to write this video's flow frames to.

    Returns:
        bool: True if flow was (re)computed, False if skipped (already done, or too few frames).
    """
    cv2.setNumThreads(1)
    fr_paths = sorted(glob.glob(vid_path + '/*.jpg'))
    if len(fr_paths) < 2:
        return False

    existing = glob.glob(out_dir + '/*.jpg')
    if len(existing) == len(fr_paths):
        return False

    frames = [np.array(Image.open(p).convert('RGB')) for p in fr_paths]
    flow_frames = compute_flow_frames(frames)

    os.makedirs(out_dir, exist_ok=True)
    store_frames(flow_frames, out_dir)
    return True


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
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        path_to_frame = os.path.join(store_path, f"frame{idx}.jpg")
        cv2.imwrite(path_to_frame, frame)


def transform_stats(model='lrcn'):
    """
    Retrieve transformation statistics based on the model type.

    For the 'lrcn' model, images are resized to 224x224; for '3dcnn', images are resized to
    112x112. Also returns the mean and standard deviation values used for normalization.

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
    elif model in ('i3d', 'i3d_two_stream', 'x3d'):
        # Kinetics-400 normalization stats shared by pytorchvideo's Kinetics-pretrained
        # checkpoints (I3D-R50, X3D, etc).
        h, w = 224, 224
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
    else:
        raise ValueError('model_type arg is undefined....')
    return h, w, mean, std


def compose_data_transforms(height, width, mean, std):
    """
    Compose and return data transforms for training and validation/test datasets.

    The training transforms include data augmentation such as random horizontal flipping and
    random affine transformations, while the validation/test transforms consist solely of
    resizing, converting to tensor, and normalizing.

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
        transforms.RandomResizedCrop((height, width), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        # degrees=0: this pipeline is also used for optical-flow pseudo-RGB images (see
        # video_datasets.TwoStreamVideoDataset), where pixel VALUES encode a motion direction.
        # Rotating the image would rotate pixel positions without rotating the encoded
        # direction values to match, producing physically inconsistent flow data. Translate
        # and scale don't have this problem (they change spatial position/extent, not
        # direction), so they're kept.
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_test_transforms = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_transforms, val_test_transforms


def compose_dataloaders(train_dataset, val_dataset, test_dataset, batch_size, model='lrcn'):
    """
    Create DataLoaders for training, validation, and test datasets in a single call.

    Selects the appropriate collate function based on the model type, matching the logic in
    train_val_dloaders/test_dloaders: 'lrcn' (RNN-based models) uses collate_fn_rnn, otherwise
    collate_fn_r3d_18 is used for 3D CNN models.

    Args:
        train_dataset (Dataset): PyTorch Dataset for training data.
        val_dataset (Dataset): PyTorch Dataset for validation data.
        test_dataset (Dataset): PyTorch Dataset for test data.
        batch_size (int): Number of samples per batch.
        model (str): Model type; 'lrcn' for RNN-based models, otherwise for 3D CNNs.

    Returns:
        dict: Dictionary with keys 'train', 'val', and 'test' mapping to their DataLoaders.
    """
    collate_fn = collate_fn_rnn if model == "lrcn" else collate_fn_r3d_18
    loader_kwargs = {'collate_fn': collate_fn, **dataloader_kwargs()}
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_dl = DataLoader(val_dataset, batch_size=2 * batch_size, shuffle=False, **loader_kwargs)
    test_dl = DataLoader(test_dataset, batch_size=2 * batch_size, shuffle=False, **loader_kwargs)
    return {'train': train_dl, 'val': val_dl, 'test': test_dl}


def train_val_dloaders(train_dataset, val_dataset, batch_size, model='lrcn'):
    """
    Create DataLoaders for training and validation datasets.

    Selects the appropriate collate function based on the model type.
    For 'lrcn' (RNN-based models), uses collate_fn_rnn which pads sequences to equal lengths.
    Otherwise, uses collate_fn_r3d_18 for 3D CNN models.

    Args:
        train_dataset (Dataset): PyTorch Dataset for training data.
        val_dataset (Dataset): PyTorch Dataset for validation data.
        batch_size (int): Number of samples per batch.
        model (str): Model type; 'lrcn' for RNN-based models, otherwise for 3D CNNs.

    Returns:
        dict: Dictionary with keys 'train' and 'val' mapping to their respective DataLoaders.
    """
    collate_fn = collate_fn_rnn if model == "lrcn" else collate_fn_r3d_18
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          collate_fn=collate_fn, **dataloader_kwargs())
    val_dl = DataLoader(val_dataset, batch_size=2 * batch_size, shuffle=False,
                        collate_fn=collate_fn, **dataloader_kwargs())
    dataloaders = {'train': train_dl, 'val': val_dl}
    return dataloaders


def test_dloaders(test_dataset, batch_size, model='lrcn'):
    """
    Create a DataLoader for the test dataset.

    Selects the appropriate collate function based on the model type.
    For 'lrcn' models, uses collate_fn_rnn; otherwise, uses collate_fn_r3d_18.

    Args:
        test_dataset (Dataset): PyTorch Dataset for test data.
        batch_size (int): Number of samples per batch.
        model (str): Model type; 'lrcn' for RNN-based models, otherwise for 3D CNNs.

    Returns:
        dict: Dictionary with key 'test' mapping to the test DataLoader.
    """
    collate_fn = collate_fn_rnn if model == "lrcn" else collate_fn_r3d_18
    test_dl = DataLoader(test_dataset, batch_size=2 * batch_size, shuffle=False,
                         collate_fn=collate_fn, **dataloader_kwargs())
    dataloaders = {'test': test_dl}
    return dataloaders
