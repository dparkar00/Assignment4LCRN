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
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms as torch_transforms


def _sample_frame_indices(n_frames, fpv, random_sample):
    """
    Choose fpv frame indices out of n_frames total frames available for a clip.

    If random_sample is True (training), divides the video into fpv equal-width segments and
    picks a uniformly random frame within each segment (TSN-style random temporal sampling).
    This means the same video contributes a DIFFERENT temporal snapshot on every epoch, which
    is real data augmentation that spatial-only transforms (flip/crop) can't provide -- without
    this, every epoch loads the exact same fpv frames for a given video, so with a small number
    of independent training videos per class the model effectively only ever sees one temporal
    "view" of each one.

    If random_sample is False (validation/test), deterministically picks the frame closest to
    the center of each segment, so evaluation is reproducible across runs/checkpoints.

    Args:
        n_frames (int): Number of frames available for this video.
        fpv (int): Number of frames to sample.
        random_sample (bool): Sample randomly (train) or deterministically (val/test).

    Returns:
        np.ndarray: Array of fpv frame indices, each in [0, n_frames). Indices repeat if
                    n_frames < fpv, same as the previous fixed linspace behavior.
    """
    if n_frames <= 0:
        return np.array([], dtype=int)
    bounds = np.linspace(0, n_frames, fpv + 1)
    indices = []
    for i in range(fpv):
        lo = int(bounds[i])
        hi = max(lo, int(bounds[i + 1]) - 1)
        hi = min(hi, n_frames - 1)
        if random_sample:
            indices.append(np.random.randint(lo, hi + 1))
        else:
            indices.append((lo + hi) // 2)
    return np.array(indices)


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
        training (bool, optional): If True, sample frames randomly per segment each epoch
                                    (real temporal augmentation). If False, sample the
                                    deterministic center frame of each segment (reproducible
                                    evaluation). Default is False.
    """
    def __init__(self, vid_dataset, fr_per_vid, transforms=None, training=False):
        self.dataset = vid_dataset
        self.fpv = fr_per_vid
        self.transforms = transforms
        self.training = training

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
        # Get all JPEG frame paths from the video directory and sample exactly fpv of them:
        # randomly per segment during training (temporal augmentation), deterministically for
        # eval. Either way, indices repeat if the video has fewer frames than fpv, so every
        # clip in a batch still comes out to exactly fpv frames (a fixed-size 3D-CNN batch
        # can't stack clips of different lengths).
        fr_paths = sorted(glob.glob(self.dataset[idx][0] + '/*.jpg'))
        if fr_paths:
            sample_idx = _sample_frame_indices(len(fr_paths), self.fpv, self.training)
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
        training (bool, optional): If True, sample frames randomly per segment each epoch
                                    (real temporal augmentation). If False, sample the
                                    deterministic center frame of each segment (reproducible
                                    evaluation). Default is False.
    """
    def __init__(self, vid_dataset, flow_dir, fr_per_vid, rgb_transforms=None,  # pylint: disable=too-many-arguments,too-many-positional-arguments
                 flow_transforms=None, training=False):
        self.dataset = vid_dataset
        self.flow_dir = flow_dir
        self.fpv = fr_per_vid
        self.rgb_transforms = rgb_transforms
        self.flow_transforms = flow_transforms
        self.training = training

    def __len__(self):
        """Return the number of video samples in the dataset."""
        return len(self.dataset)

    def _load_clip(self, dir_path, transforms, clip_seed, sample_idx):
        """Load the frames at sample_idx (precomputed once per video and shared between the
        RGB and flow calls, so both streams describe the same temporal moments) from dir_path,
        applying the given random seed before every frame's transform so the whole clip gets
        one consistent spatial augmentation decision."""
        fr_paths = sorted(glob.glob(dir_path + '/*.jpg'))
        if not fr_paths or len(sample_idx) == 0:
            return torch.empty(0)
        fr_paths = [fr_paths[i] for i in sample_idx if i < len(fr_paths)]
        if not fr_paths:
            return torch.empty(0)
        fr_imgs = [Image.open(fr_path) for fr_path in fr_paths]
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

        # Pick ONE set of temporal indices for BOTH streams -- an index must refer to the same
        # underlying moment in the RGB and flow sequences (they're kept frame-count-matched by
        # preprocess_video_flow), so sampling each stream independently would pair an RGB frame
        # from one moment with flow describing a different moment entirely.
        n_frames = len(glob.glob(rgb_path + '/*.jpg'))
        sample_idx = _sample_frame_indices(n_frames, self.fpv, self.training)

        # Use the SAME seed for both streams so any random flip/affine is applied identically
        # to the RGB clip and its corresponding flow clip -- otherwise the two streams could
        # end up spatially misaligned (e.g. RGB flipped, flow not), which breaks the
        # correspondence between appearance and motion that two-stream fusion relies on.
        clip_seed = torch.seed()
        rgb_clip = self._load_clip(rgb_path, self.rgb_transforms, clip_seed, sample_idx)
        flow_clip = self._load_clip(flow_path, self.flow_transforms, clip_seed, sample_idx)

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


def dataset_split(vid_dataset, tr_ratio, ts_ratio, seed=0):
    # pylint: disable=too-many-locals,too-many-statements
    """
    Split the dataset into training, validation, and test sets, grouped by source video AND
    balanced per class.

    Grouping (see _video_group_id) ensures every clip from the same source video ends up in
    exactly one of train/val/test -- never spread across more than one, which is what prevents
    the classic HMDB51-style leak of near-duplicate footage appearing in both training and
    evaluation. A plain (unstratified) group split can still leave individual classes badly
    imbalanced purely by chance, though: HMDB51's clip-count-per-source-video varies a lot, so
    randomly assigning whole groups to splits can leave some classes with very few training
    examples if their clips happen to be concentrated in just a few large source videos.

    This balances in two phases:
      1. Every class is guaranteed at least one group in train first (its largest available
         group). This is a hard correctness floor, not a heuristic -- a class can never end up
         with zero training examples, however pathological its group-size distribution is.
      2. Remaining groups (largest first) are assigned greedily to whichever split has the
         largest PROPORTIONAL deficit (fraction of that split's own target still unmet, not
         the raw/absolute deficit) for the classes that group's clips belong to. Proportional
         deficit matters: comparing absolute deficits systematically starves small-target
         splits like val (e.g. 10%) in favor of train (e.g. 75%), since train's raw numbers
         are bigger even when val is proportionally far more under-served. Ties are broken by
         visiting splits in a random order each time, so they don't systematically favor
         whichever split happens to be checked first.

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
    print('Splitting train/validation/test datasets (grouped by source video, '
          'balanced per class)....')

    val_ratio = 1 - tr_ratio - ts_ratio
    target_ratio = {'train': tr_ratio, 'val': val_ratio, 'test': ts_ratio}

    # Each group holds ALL of its clips (across every class it happens to touch, in the rare
    # case a source video contributed clips to more than one class) -- it is always assigned
    # to exactly one split, whole, which is what actually prevents leakage.
    group_paths, group_labels = {}, {}
    for path, label, group in zip(vid_paths, vid_labels, groups):
        group_paths.setdefault(group, []).append(path)
        group_labels.setdefault(group, []).append(label)

    unique_groups = list(group_paths.keys())
    rng = np.random.RandomState(seed)  # pylint: disable=no-member
    rng.shuffle(unique_groups)
    # Largest groups first, so both phases have the most room to work with while every split
    # still has budget left.
    unique_groups.sort(key=lambda g: -len(group_paths[g]))

    all_labels = sorted(set(vid_labels.tolist()))
    cls_totals = {lbl: 0 for lbl in all_labels}
    for lbl in vid_labels:
        cls_totals[lbl] += 1
    cls_counts = {split: {lbl: 0 for lbl in all_labels} for split in target_ratio}
    split_lists = {'train': [], 'val': [], 'test': []}

    # Phase 1: guarantee every class has at least one group in train.
    assigned_groups = set()
    for lbl in all_labels:
        cls_groups = [g for g in unique_groups
                      if lbl in set(group_labels[g]) and g not in assigned_groups]
        if not cls_groups:
            continue
        chosen = cls_groups[0]  # largest remaining group touching this class
        for path, group_lbl in zip(group_paths[chosen], group_labels[chosen]):
            split_lists['train'].append((path, group_lbl))
            cls_counts['train'][group_lbl] += 1
        assigned_groups.add(chosen)

    # Phase 2: proportional greedy balancing for everything else.
    for group in unique_groups:
        if group in assigned_groups:
            continue
        g_paths, g_labels = group_paths[group], group_labels[group]
        split_order = list(target_ratio.keys())
        rng.shuffle(split_order)
        deficits = {}
        for split in split_order:  # pylint: disable=consider-using-dict-items
            deficits[split] = sum(
                (target_ratio[split] * cls_totals[lbl] - cls_counts[split][lbl])
                / max(target_ratio[split] * cls_totals[lbl], 1e-9)
                for lbl in set(g_labels)
            )
        chosen = max(deficits, key=deficits.get)
        for path, lbl in zip(g_paths, g_labels):
            split_lists[chosen].append((path, lbl))
            cls_counts[chosen][lbl] += 1

    tr_dataset, val_dataset, ts_dataset = (split_lists['train'], split_lists['val'],
                                            split_lists['test'])

    # Verification: every class must have at least one training example, and report the
    # per-class balance actually achieved as evidence (for e.g. a README writeup) that
    # balancing is working, not just assumed.
    train_counts = list(cls_counts['train'].values())
    val_counts = list(cls_counts['val'].values())
    test_counts = list(cls_counts['test'].values())
    zero_train = sum(1 for c in train_counts if c == 0)
    zero_val = sum(1 for c in val_counts if c == 0)
    zero_test = sum(1 for c in test_counts if c == 0)
    print(f'Per-class training examples -- min: {min(train_counts)}, '
          f'max: {max(train_counts)}, mean: {sum(train_counts)/len(train_counts):.1f}')
    print(f'Classes with zero examples -- train: {zero_train}, val: {zero_val}, '
          f'test: {zero_test} (val/test zeros can be unavoidable for classes with very '
          f'few source videos, since a whole group can never be split across splits)')
    assert zero_train == 0, f'{zero_train} class(es) have ZERO training examples!'

    # Group-overlap verification: structurally guaranteed by the atomic per-group assignment
    # above, but checked explicitly anyway as a safety net and as evidence the leak fix holds.
    train_groups = {_video_group_id(p) for p, _ in tr_dataset}
    val_groups = {_video_group_id(p) for p, _ in val_dataset}
    test_groups = {_video_group_id(p) for p, _ in ts_dataset}
    tv_overlap = train_groups & val_groups
    tt_overlap = train_groups & test_groups
    vt_overlap = val_groups & test_groups
    overlap = tv_overlap | tt_overlap | vt_overlap
    print(f'Source-video groups -- train: {len(train_groups)}, val: {len(val_groups)}, '
          f'test: {len(test_groups)}, cross-split overlap: {len(overlap)} (must be 0)')
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
    valid = [(imgs, label) for imgs, label in zip(imgs_batch, label_batch) if len(imgs) > 0]
    if not valid:
        return None, None
    imgs_batch, label_batch = zip(*valid)
    imgs_tensor = torch.stack(imgs_batch)
    imgs_tensor = torch.transpose(imgs_tensor, 2, 1)
    labels_tensor = torch.tensor(label_batch)
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
