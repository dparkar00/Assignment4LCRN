"""
Module: video_datasets.py

This module provides classes and functions for loading and processing video datasets,
splitting them into training, validation, and test sets, and preparing data for video
classification models. It includes a custom PyTorch Dataset for videos stored as directories
of frame images, functions to load the dataset from a directory structure, split the dataset
using stratified sampling, and custom collate functions for handling variable-length video
sequences.
"""

import os
import glob

import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms as torch_transforms


class VideoDataset(Dataset):
    """
    PyTorch Dataset class for loading video data from directories of frame images.

    Each video is represented as a directory containing JPEG images of its frames.
    The dataset is provided as a dictionary mapping each video directory path to its label.

    Args:
        vid_dataset (dict): Dictionary where keys are video directory paths and values are
                             integer labels.
        fr_per_vid (int): Number of frames per video to load (images are taken in order).
        transforms (callable, optional): A function/transform to apply to each frame image
                                          (e.g., resizing, normalization).
    """
    def __init__(self, vid_dataset, fr_per_vid, transforms=None):
        self.dataset = vid_dataset
        self.fpv = fr_per_vid
        self.transforms = transforms

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
                   with T == fr_per_vid and label is an integer.
        """
        # Get all JPEG frame paths from the video directory and uniformly sample exactly
        # fpv of them. linspace is used regardless of whether the video has more or fewer
        # frames than fpv: when there are fewer, indices repeat so every clip in a batch still
        # comes out to exactly fpv frames (a fixed-size 3D-CNN batch can't stack clips of
        # different lengths).
        fr_paths = sorted(glob.glob(self.dataset[idx][0] + '/*.jpg'))
        if fr_paths:
            sample_idx = np.linspace(0, len(fr_paths) - 1, self.fpv).astype(int)
            fr_paths = [fr_paths[i] for i in sample_idx]

        # Open images using PIL
        fr_imgs = [Image.open(fr_path) for fr_path in fr_paths]

        # Get the label associated with the video
        fr_label = self.dataset[idx][1]

        # Apply transforms to each frame if provided, else keep original images.
        # The same random seed is reused for every frame in the clip so that stochastic
        # transforms (flip, affine) make one consistent decision per video instead of a
        # different one per frame, which would otherwise corrupt the temporal signal.
        if self.transforms:
            clip_seed = torch.seed()
            fr_imgs_trans = []
            for fr_img in fr_imgs:
                torch.manual_seed(clip_seed)
                fr_imgs_trans.append(self.transforms(fr_img))
        else:
            fr_imgs_trans = fr_imgs

        # Stack transformed images into a tensor if available
        if len(fr_imgs_trans) > 0:
            fr_imgs_trans = torch.stack(fr_imgs_trans)

        return fr_imgs_trans, fr_label


class TwoStreamVideoDataset(Dataset):
    """
    PyTorch Dataset that pairs an RGB clip with its corresponding pre-computed optical-flow clip,
    for two-stream video classification (e.g. two-stream I3D).

    Flow frames are expected to be pre-computed (see utils.compute_flow_frames) and stored under
    flow_dir using the same class/video subfolder names as the RGB frame directory.

    Args:
        vid_dataset (list): List of (rgb_video_path, label) tuples, as produced by dataset_split.
        flow_dir (str): Root directory holding the pre-computed flow frames.
        fr_per_vid (int): Number of frames per video to load.
        rgb_transforms (callable, optional): Transform applied to each RGB frame.
        flow_transforms (callable, optional): Transform applied to each flow frame.
    """
    def __init__(self, vid_dataset, flow_dir, fr_per_vid, rgb_transforms=None,
                 flow_transforms=None):
        self.dataset = vid_dataset
        self.flow_dir = flow_dir
        self.fpv = fr_per_vid
        self.rgb_transforms = rgb_transforms
        self.flow_transforms = flow_transforms

    def __len__(self):
        """Return the number of video samples in the dataset."""
        return len(self.dataset)

    def _load_clip(self, dir_path, transforms, clip_seed):
        """Load up to fpv uniformly-sampled, temporally-sorted frames from dir_path, applying
        the given random seed before every frame's transform so the whole clip gets one
        consistent augmentation decision (shared across streams via a common clip_seed)."""
        fr_paths = sorted(glob.glob(dir_path + '/*.jpg'))
        if fr_paths:
            sample_idx = np.linspace(0, len(fr_paths) - 1, self.fpv).astype(int)
            fr_paths = [fr_paths[i] for i in sample_idx]
        fr_imgs = [Image.open(fr_path) for fr_path in fr_paths]
        if not fr_imgs:
            return torch.empty(0)
        fr_imgs_trans = []
        for fr_img in fr_imgs:
            torch.manual_seed(clip_seed)
            if transforms:
                fr_imgs_trans.append(transforms(fr_img))
            else:
                fr_imgs_trans.append(torch_transforms.functional.to_tensor(fr_img))
        return torch.stack(fr_imgs_trans)

    def __getitem__(self, idx):
        """
        Load the matching RGB and flow clips for a video and return them with its label.

        Returns:
            tuple: (rgb_clip, flow_clip, label)
        """
        rgb_path = self.dataset[idx][0]
        label = self.dataset[idx][1]

        # Flow frames are stored under the same class/video subpath, rooted at flow_dir instead
        # of the RGB frame_dir, so swap the root while keeping the class/video suffix.
        rel_path = os.path.join(*rgb_path.rstrip('/').split(os.sep)[-2:])
        flow_path = os.path.join(self.flow_dir, rel_path)

        # Use the SAME seed for both streams so any random flip/affine is applied identically
        # to the RGB clip and its corresponding flow clip -- otherwise the two streams could
        # end up spatially misaligned (e.g. RGB flipped, flow not), which breaks the
        # correspondence between appearance and motion that two-stream fusion relies on.
        clip_seed = torch.seed()
        rgb_clip = self._load_clip(rgb_path, self.rgb_transforms, clip_seed)
        flow_clip = self._load_clip(flow_path, self.flow_transforms, clip_seed)

        return rgb_clip, flow_clip, label


def collate_fn_two_stream(batch):
    """
    Collate function for two-stream 3D CNN models (e.g. two-stream I3D).

    Filters out any samples missing either an RGB or flow clip, stacks both streams separately,
    and transposes each to the (batch, channels, time, H, W) layout expected by 3D CNNs.

    Args:
        batch (list): List of samples, each as (rgb_clip, flow_clip, label).

    Returns:
        tuple: ((rgb_tensor, flow_tensor), labels_tensor)
    """
    rgb_batch, flow_batch, label_batch = list(zip(*batch))
    valid = [(r, f, l) for r, f, l in zip(rgb_batch, flow_batch, label_batch)
             if len(r) > 0 and len(f) > 0]
    if not valid:
        return (None, None), None
    rgb_batch, flow_batch, label_batch = zip(*valid)

    rgb_tensor = torch.transpose(torch.stack(rgb_batch), 2, 1)
    flow_tensor = torch.transpose(torch.stack(flow_batch), 2, 1)
    labels_tensor = torch.tensor(label_batch)
    return (rgb_tensor, flow_tensor), labels_tensor


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
    vid_dataset = {}
    print('Loading video dataset....')
    for vid_cat in tqdm(sorted(os.listdir(frame_dir))):
        vid_cat_path = os.path.join(frame_dir, vid_cat)
        for vid in sorted(os.listdir(vid_cat_path)):
            vid_path = os.path.join(vid_cat_path, vid)
            vid_dataset[vid_path] = label_dict[vid_cat]
    return vid_dataset, label_dict


def _video_group_id(vid_path):
    """
    Extract a stable identifier for the SOURCE video a clip was taken from.

    HMDB51 (and similarly-structured action-recognition datasets) commonly include multiple
    clips cut from the same underlying source video, following the naming convention
    '<source_video_title>_<action_class>_<body_pos>_<camera_motion>_<num_people>_<camera_view>_
    <quality>_<clip_index>' -- e.g. 'April_09_brush_hair_u_nm_np1_ba_goo_0', '..._1', '..._2'
    are three different clips of the single source video 'April_09'. Splitting at the clip
    level (treating each as an independent sample) is a data leak: near-identical footage --
    same actor, room, lighting, camera -- ends up in both train and test, letting the model
    partially recognize the specific video instead of generalizing the action. Grouping clips
    by this source-video id (see dataset_split below) keeps all of one video's clips in a
    single split.

    Args:
        vid_path (str): Path to a video's frame directory, e.g.
                         '.../brush_hair/April_09_brush_hair_u_nm_np1_ba_goo_0'.

    Returns:
        str: The inferred source-video id (e.g. 'April_09'), or the full clip name if the
             expected '_<action_class>' marker isn't found in it (safe fallback: treats the
             clip as its own group rather than guessing wrong).
    """
    vid_path = vid_path.rstrip('/')
    vid_name = os.path.basename(vid_path)
    class_name = os.path.basename(os.path.dirname(vid_path))
    marker = '_' + class_name
    if marker in vid_name:
        return vid_name.split(marker, 1)[0]
    return vid_name


def _to_pairs(paths, labels):
    """Zip parallel path/label arrays into a list of (path, label) tuples."""
    return list(zip(paths, labels))


def dataset_split(vid_dataset, tr_ratio, ts_ratio, seed=0):  # pylint: disable=too-many-locals
    """
    Split the dataset into training, validation, and test sets, grouped by source video.

    Uses GroupShuffleSplit (grouped by _video_group_id) rather than a plain stratified split,
    so that every clip taken from the same source video ends up in exactly one of train/val/
    test -- never spread across more than one. A plain per-clip stratified split would leak:
    HMDB51-style datasets frequently contain multiple clips per source video, so randomly
    assigning individual clips to splits lets near-duplicate footage appear in both training
    and evaluation.

    Args:
        vid_dataset (dict): Dictionary mapping video paths to labels.
        tr_ratio (float): Proportion of the data to use for training.
        ts_ratio (float): Proportion of the data to use for testing.
        seed (int, optional): Random seed for reproducibility. Default is 0.

    Returns:
        tuple: (tr_dataset, val_dataset, ts_dataset)
            - tr_dataset (list): List of (video_path, label) tuples for the training set.
            - val_dataset (list): List of (video_path, label) tuples for the validation set.
            - ts_dataset (list): List of (video_path, label) tuples for the test set.
    """
    vid_paths = np.array(list(vid_dataset.keys()))
    vid_labels = np.array(list(vid_dataset.values()))
    groups = np.array([_video_group_id(p) for p in vid_paths])
    print('Splitting train/validation/test datasets (grouped by source video)....')

    # Test split using GroupShuffleSplit
    ts_spliter = GroupShuffleSplit(n_splits=1, test_size=ts_ratio, random_state=seed)
    for tr_val_idx, ts_idx in ts_spliter.split(vid_paths, vid_labels, groups=groups):
        ts_dataset = _to_pairs(vid_paths[ts_idx], vid_labels[ts_idx])
        tr_val_paths, tr_val_labels = vid_paths[tr_val_idx], vid_labels[tr_val_idx]
        tr_val_groups = groups[tr_val_idx]
        ts_groups = groups[ts_idx]

    # Train/validation split
    val_ratio = 1 - tr_ratio - ts_ratio
    val_wt = val_ratio / (tr_ratio + val_ratio)
    val_spliter = GroupShuffleSplit(n_splits=1, test_size=val_wt, random_state=seed)
    for tr_idx, val_idx in val_spliter.split(tr_val_paths, tr_val_labels, groups=tr_val_groups):
        tr_dataset = _to_pairs(tr_val_paths[tr_idx], tr_val_labels[tr_idx])
        val_dataset = _to_pairs(tr_val_paths[val_idx], tr_val_labels[val_idx])
        tr_groups = set(tr_val_groups[tr_idx])
        val_groups = set(tr_val_groups[val_idx])

    # Verification: confirm no source-video group spans more than one split. This is both a
    # safety check and evidence (for e.g. a README writeup) that the leak is actually fixed.
    ts_group_set = set(ts_groups)
    overlap = (tr_groups & val_groups) | (tr_groups & ts_group_set) | (val_groups & ts_group_set)
    print(f'Source-video groups -- train: {len(tr_groups)}, val: {len(val_groups)}, '
          f'test: {len(ts_group_set)}, cross-split overlap: {len(overlap)} (must be 0)')
    assert not overlap, f'Data leak: {len(overlap)} source video(s) span more than one split!'

    return tr_dataset, val_dataset, ts_dataset


def collate_fn_r3d_18(batch):
    """
    Collate function for 3D CNN models (e.g., R3D-18).

    Assumes each sample in the batch is a tuple (video_frames, label),
    where video_frames is a tensor of shape (T, C, H, W). This function filters out any samples
    with no frames, stacks the video frame tensors, transposes the tensor dimensions as needed,
    and stacks the labels.

    Args:
        batch (list): List of samples, each as (video_frames, label).

    Returns:
        tuple: (imgs_tensor, labels_tensor)
            - imgs_tensor (Tensor): Stacked video frames tensor with shape adjusted for R3D-18.
            - labels_tensor (Tensor): Tensor of labels.
    """
    imgs_batch, label_batch = list(zip(*batch))
    imgs_batch = [imgs for imgs in imgs_batch if len(imgs) > 0]
    label_batch = [torch.tensor(l) for l, imgs in zip(label_batch, imgs_batch) if len(imgs) > 0]
    imgs_tensor = torch.stack(imgs_batch)
    imgs_tensor = torch.transpose(imgs_tensor, 2, 1)
    labels_tensor = torch.stack(label_batch)
    return imgs_tensor, labels_tensor


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
        return None, None
    imgs_batch, label_batch = zip(*valid_samples)

    # Pad the video frame tensors along the time dimension (T)
    # Resulting shape: (batch_size, max_T, C, H, W)
    padded_imgs = pad_sequence(imgs_batch, batch_first=True)

    # Convert labels to a tensor
    labels_tensor = torch.tensor(label_batch)

    return padded_imgs, labels_tensor
