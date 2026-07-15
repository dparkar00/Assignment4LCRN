"""
Module: video_datasets.py

This module provides classes and functions for loading and processing video datasets,
splitting them into training, validation, and test sets, and preparing data for video
classification models. It includes a custom PyTorch Dataset for videos stored as
directories of frame images, functions to load the dataset from a directory structure,
split the dataset using stratified sampling, and custom collate functions for handling
variable-length video sequences.
"""

import os
import glob

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import cv2
from PIL import Image
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def _compute_flow_stack(fr_imgs, size):
    """
    Model improvement: dense optical flow (Farneback), used for the two-stream
    RGB + optical-flow architecture (TwoStreamR3D, see models.py) in the spirit
    of I3D's two-stream design (Carreira & Zisserman, 2017) -- the model gets
    direct access to motion information instead of only inferring it indirectly
    from how appearance features change across a sequence of independent frames.

    Args:
        fr_imgs (list): List of PIL RGB Image frames (pre-transform).
        size (tuple): (height, width) to resize frames to before computing flow,
            matching the RGB stream's spatial resolution so both streams stay
            spatially aligned.

    Returns:
        list: List of (2, H, W) float32 tensors (x/y flow components, clipped
            and scaled to roughly [-1, 1]), one per input frame. The last
            frame's flow is duplicated from the final consecutive pair so the
            flow stack has the same length as the RGB stack (there is one fewer
            flow field than frames, since flow is computed between pairs).
    """
    if len(fr_imgs) == 0:
        return []
    gray_frames = [
        cv2.cvtColor(  # pylint: disable=no-member
            np.array(img.resize((size[1], size[0]))),
            cv2.COLOR_RGB2GRAY,  # pylint: disable=no-member
        )
        for img in fr_imgs
    ]
    flows = []
    for i in range(len(gray_frames) - 1):
        flow = cv2.calcOpticalFlowFarneback(  # pylint: disable=no-member
            gray_frames[i], gray_frames[i + 1], None,
            pyr_scale=0.5, levels=3, winsize=15, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0,
        )
        # Clip to a fixed range instead of per-clip min/max normalization, so
        # flow magnitude stays comparable across different clips/videos.
        flow = np.clip(flow, -20, 20) / 20.0
        flows.append(torch.from_numpy(flow).permute(2, 0, 1).float())  # (2, H, W)
    if flows:
        flows.append(flows[-1])
    else:
        flows = [torch.zeros(2, size[0], size[1])] * len(fr_imgs)
    return flows


def _flow_cache_path(cache_dir, clip_dir, n_frames, size):
    """
    Build a deterministic, filesystem-safe cache filename for a clip's computed
    optical flow, so repeated epochs can reuse it instead of recomputing.
    """
    safe_name = clip_dir.replace(os.sep, "_").replace(":", "_")
    return os.path.join(cache_dir, f"{safe_name}_n{n_frames}_{size[0]}x{size[1]}.npy")


class VideoDataset(Dataset):
    """
    PyTorch Dataset class for loading video data from directories of frame images.

    Each video is represented as a directory containing JPEG images of its frames.
    The dataset is provided as a dictionary mapping each video directory path to its label.

    Args:
        vid_dataset (dict): Dictionary where keys are video directory paths and values
            are integer labels.
        fr_per_vid (int): Number of frames per video to load (images are taken in order).
        transforms (callable, optional): A function/transform to apply to each frame
            image (e.g., resizing, normalization).
        compute_flow (bool, optional): Model improvement -- if True, also compute
            dense optical flow between consecutive frames and concatenate it onto
            each RGB frame's channels (3 RGB + 2 flow = 5 channels total), for the
            two-stream RGB+flow architecture (see models.TwoStreamR3D). Default False.
        flow_size (tuple, optional): (height, width) to resize frames to before
            computing flow, matching the RGB transform's output resolution so
            both streams stay spatially aligned. Only used when compute_flow=True.
        flow_cache_dir (str, optional): PERFORMANCE IMPROVEMENT -- if set, computed
            optical flow is cached to disk as .npy files and reused on later
            epochs instead of being recomputed every time. Flow is deterministic
            per clip (computed from the raw frames before RGB augmentation is
            applied), so recomputing it every epoch is pure wasted CPU work --
            caching cuts that cost to a one-time computation per clip. Only used
            when compute_flow=True.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(self, vid_dataset, fr_per_vid, transforms=None,
                 compute_flow=False, flow_size=(112, 112), flow_cache_dir=None):
        self.dataset = vid_dataset
        self.fpv = fr_per_vid
        self.transforms = transforms
        self.compute_flow = compute_flow
        self.flow_size = flow_size
        self.flow_cache_dir = flow_cache_dir
        if flow_cache_dir is not None:
            os.makedirs(flow_cache_dir, exist_ok=True)

    def __len__(self):
        """Return the number of video samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Load frames from the video directory corresponding to the given index, apply transforms,
        and return the stacked tensor of frames along with its label.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (frames_tensor, label) where frames_tensor is a tensor of shape (T, C, H, W)
                   with T being the number of frames (up to fr_per_vid) and label is an integer.
        """
        # Get all JPEG frame paths from the video directory and select up to fr_per_vid frames
        fr_paths = sorted(
                    glob.glob(os.path.join(self.dataset[idx][0], "*.jpg")),
                    key=lambda p: int("".join(filter(str.isdigit, os.path.basename(p)))
                                    or 0),
                )
        fr_paths = fr_paths[:self.fpv]

        # Open images using PIL
        # ROBUSTNESS FIX: a single corrupted/truncated JPEG (e.g. from an interrupted
        # download or unzip) previously crashed the entire DataLoader worker and
        # killed the whole training run with PIL.UnidentifiedImageError. Now a bad
        # frame is skipped and logged instead of taking down the run.
        fr_imgs = []
        for fr_path in fr_paths:
            try:
                fr_imgs.append(Image.open(fr_path).convert("RGB"))
            except (OSError, ValueError) as exc:
                print(f"Skipping unreadable frame: {fr_path} ({exc})")

        # Get the label associated with the video
        fr_label = self.dataset[idx][1]

        # Apply transforms to each frame if provided, else keep original images
        fr_imgs_trans = (
            [self.transforms(fr_img) for fr_img in fr_imgs] if self.transforms else fr_imgs
        )

        # Model improvement: two-stream RGB + optical flow. Concatenate a 2-channel
        # flow field onto each frame's 3 RGB channels (-> 5 channels total) before
        # stacking, so TwoStreamR3D can split them back apart in its forward pass.
        # PERFORMANCE IMPROVEMENT: flow is deterministic per clip (computed from
        # raw frames before RGB augmentation), so it's cached to disk on first
        # computation and reused on every subsequent epoch instead of being
        # recomputed from scratch each time -- see flow_cache_dir above.
        if self.compute_flow and len(fr_imgs) > 0:
            clip_dir = self.dataset[idx][0]
            cache_path = (
                _flow_cache_path(self.flow_cache_dir, clip_dir, len(fr_imgs), self.flow_size)
                if self.flow_cache_dir is not None else None
            )
            if cache_path is not None and os.path.exists(cache_path):
                flow_arr = np.load(cache_path)
                flow_stack = [torch.from_numpy(f).float() for f in flow_arr]
            else:
                flow_stack = _compute_flow_stack(fr_imgs, self.flow_size)
                if cache_path is not None:
                    np.save(cache_path, torch.stack(flow_stack).numpy())
            fr_imgs_trans = [
                torch.cat([rgb_t, flow_t], dim=0)
                for rgb_t, flow_t in zip(fr_imgs_trans, flow_stack)
            ]

        # Stack transformed images into a tensor if available
        if len(fr_imgs_trans) > 0:
            fr_imgs_trans = torch.stack(fr_imgs_trans)

        return fr_imgs_trans, fr_label

def _group_key(folder_name, vid_cat):
    """
    Recover the source-video identity from an HMDB51 clip folder name.

    HMDB51 folders are named `{source_video_name}_{class}_{condition_codes}_{clip_idx}`.
    Since the class label is known, splitting on `_{class}_` recovers the source video
    name shared by all clips cut from the same recording.
    """
    marker = f"_{vid_cat}_"
    if marker in folder_name:
        return folder_name.split(marker)[0]
    return folder_name

def load_dataset(frame_dir):
    """
    Load the full video dataset from the specified directory.

    Each subdirectory in frame_dir is assumed to correspond to a video category.
    The function builds a dictionary where keys are paths to video directories and
    values are integer labels corresponding to each category.

    Args:
        frame_dir (str): Path to the directory containing subdirectories for each video category.

    Returns:
        tuple: (vid_dataset, label_dict)
            - vid_dataset (dict): Dictionary mapping video directory paths to integer labels.
            - label_dict (dict): Dictionary mapping video category names to integer labels.
    """
    label_dict = {vid_cat: idx for idx, vid_cat in enumerate(sorted(os.listdir(frame_dir)))}

    paths, labels, groups = [], [], []
    print("Loading video dataset....")
    for vid_cat in sorted(os.listdir(frame_dir)):
        vid_cat_path = os.path.join(frame_dir, vid_cat)
        if not os.path.isdir(vid_cat_path):
            continue
        for vid in os.listdir(vid_cat_path):
            vid_path = os.path.join(vid_cat_path, vid)
            paths.append(vid_path)
            labels.append(label_dict[vid_cat])
            groups.append(_group_key(vid, vid_cat))

    return np.array(paths), np.array(labels), np.array(groups), label_dict


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def dataset_split(vid_paths, vid_labels, vid_groups, tr_ratio, ts_ratio, seed=0):
    """
    Split the dataset into training, validation, and test sets using *group-aware*
    stratified sampling, so that clips sharing a source video never cross the
    train/val/test boundary (see module docstring for why this matters).

    Args:
        vid_paths (np.ndarray): Video directory paths.
        vid_labels (np.ndarray): Integer class labels.
        vid_groups (np.ndarray): Source-video group identifiers.
        tr_ratio (float): Proportion of the data to use for training.
        ts_ratio (float): Proportion of the data to use for testing.
        seed (int, optional): Random seed for reproducibility. Default is 0.

    Returns:
        tuple: (tr_dataset, val_dataset, ts_dataset), each a list of (path, label) tuples.
    """

    print('Splitting train/validation/test datasets....')

    n_splits_test = max(2, round(1 / ts_ratio))
    sgkf_test = StratifiedGroupKFold(n_splits=n_splits_test, shuffle=True, random_state=seed)
    tr_val_idx, ts_idx = next(sgkf_test.split(vid_paths, vid_labels, groups=vid_groups))

    ts_dataset = [(vid_paths[i], int(vid_labels[i])) for i in ts_idx]

    tr_val_paths = vid_paths[tr_val_idx]
    tr_val_labels = vid_labels[tr_val_idx]
    tr_val_groups = vid_groups[tr_val_idx]

    val_ratio = 1 - tr_ratio - ts_ratio
    val_wt = val_ratio / (tr_ratio + val_ratio)
    n_splits_val = max(2, round(1 / val_wt))
    sgkf_val = StratifiedGroupKFold(n_splits=n_splits_val, shuffle=True, random_state=seed)
    tr_idx, val_idx = next(sgkf_val.split(tr_val_paths, tr_val_labels, groups=tr_val_groups))

    tr_dataset = [(tr_val_paths[i], int(tr_val_labels[i])) for i in tr_idx]
    val_dataset = [(tr_val_paths[i], int(tr_val_labels[i])) for i in val_idx]
    return tr_dataset, val_dataset, ts_dataset


def collate_fn_r3d_18(batch):
    """
    Collate function for 3D CNN models (e.g., R3D-18).

    Assumes each sample in the batch is a tuple (video_frames, label),
    where video_frames is a tensor of shape (T, C, H, W). This function filters out any samples
    with no frames, stacks the video frame tensors, transposes the tensor dimensions as needed,
    and stacks the labels.

    BUG FIX: the original implementation called torch.stack directly on imgs_batch,
    which requires every clip to have exactly the same number of frames (T). But
    VideoDataset can return fewer than fr_per_vid frames if a clip's source
    directory has fewer JPEGs than requested, or after a corrupted frame is
    skipped -- torch.stack then raises a RuntimeError on any batch containing
    clips of different lengths. Clips are now padded to the longest clip in the
    batch first (zero-padding extra frames), the same variable-length handling
    collate_fn_rnn already applies for the LSTM path. R3D-18 tolerates this fine
    since it pools over the time dimension at the end regardless of its length.

    Args:
        batch (list): List of samples, each as (video_frames, label).

    Returns:
        tuple: (imgs_tensor, labels_tensor, lengths)
            - imgs_tensor (Tensor): Stacked video frames tensor with shape adjusted for R3D-18.
            - labels_tensor (Tensor): Tensor of labels.
            - lengths: Always None -- R3D-18 takes a fixed-length clip directly and has
                no notion of padded/variable-length sequences the way the LSTM path
                does, but this keeps the same 3-tuple interface as collate_fn_rnn so
                the training/eval loops don't need model-type-specific unpacking.
    """
    imgs_batch, label_batch = list(zip(*batch))
    imgs_batch = [imgs for imgs in imgs_batch if len(imgs) > 0]
    label_batch = [torch.tensor(l) for l, imgs in zip(label_batch, imgs_batch) if len(imgs) > 0]
    padded_imgs = pad_sequence(imgs_batch, batch_first=True)  # (B, max_T, C, H, W)
    imgs_tensor = torch.transpose(padded_imgs, 2, 1)  # -> (B, C, max_T, H, W) for R3D-18
    labels_tensor = torch.stack(label_batch)
    return imgs_tensor, labels_tensor, None


def collate_fn_rnn(batch):
    """
    Collate function for RNN-based models.

    Handles variable-length video sequences by padding them to the length of the longest sequence
    in the batch. Each sample in the batch is expected to be a tuple (video_frames, label),
    where video_frames is a tensor of shape (T, C, H, W). The function returns a padded tensor
    of video frames with shape (batch_size, max_T, C, H, W) and a tensor of labels.

    Args:
        batch (list): List of samples, each as (video_frames, label).

    Returns:
        tuple: (padded_imgs, labels_tensor)
            - padded_imgs (Tensor): Padded tensor of video frames.
            - labels_tensor (Tensor): Tensor of labels.
    """
    # Unzip the batch into image tensors and labels
    imgs_batch, label_batch = list(zip(*batch))

    # Filter out any samples that have no frames
    valid_samples = [(imgs, label) for imgs, label in zip(imgs_batch, label_batch) if len(imgs) > 0]
    if not valid_samples:
        return None, None, None
    imgs_batch, label_batch = zip(*valid_samples)

    lengths = torch.tensor([imgs.shape[0] for imgs in imgs_batch])

    # Pad the video frame tensors along the time dimension (T)
    # Resulting shape: (batch_size, max_T, C, H, W)
    padded_imgs = pad_sequence(imgs_batch, batch_first=True)

    # Convert labels to a tensor
    labels_tensor = torch.tensor(label_batch)

    return padded_imgs, labels_tensor, lengths
