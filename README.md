# HMDB51 Video Classification

This repository was originally a working-but-flawed video classification pipeline (LRCN: ResNet + LSTM, trained on HMDB51). It was explicitly given such that the codebase contained **a data leak** and **a critically significant bug**. This README documents what was found and fixed, the model-level improvements made on top of the corrected base, and how to run the
project.




## Dataset Preparation

### Step 0: Download and Unzip Dataset

1. **Download Dataset:**  
   Download the HMDB51 dataset from [Kaggle](https://www.kaggle.com/datasets/easonlll/hmdb51). This dataset contains videos of 51 different human action classes.

2. **Unzip and Organize:**  
   Unzip the downloaded dataset. The expected folder structure should be as follows:
   
        - HMDB51
            - Action_Class1
            - Action_Class2
            ... ... ... ...
            - Action_Class51

Each subdirectory represents a different action class.

---

## Environment Setup

1. **Python Version:**  
This project requires Python 3.7 or higher.

2. **Dependencies:**  
Install the required Python packages by running:

```bash
pip install -r requirements.txt
```

### Key Libraries

- **PyTorch**
- **torchvision**
- **OpenCV**
- **scikit-learn**
- **tqdm**
- **numpy**
- **Pillow**

## Hardware Requirements

A CUDA-enabled GPU is recommended for training. The code automatically detects GPU availability.

---

## Preprocessing and Frame Extraction

Before training, the raw video files must be converted into frame sequences. The preprocessing module includes functions for:

### Uniform Frame Sampling

- The `get_frames` function uses OpenCV to sample a fixed number of frames per video.

### Saving Frames to Disk

- The `store_frames` function writes the extracted frames as JPEG images.

Integrate these functions into a preprocessing script (e.g., `preprocess.py`) to convert all videos into folders of extracted frames. The resulting folder structure should mirror the original dataset structure:

## Table of Contents
- [Bugs Fixed and Data Leak](#bugs-fixed-and-data-leak)
- [Model-Level Improvements](#model-level-improvements)
- [Preprocessing and Frame Extraction](#preprocessing-and-frame-extraction)
- [Training the Model](#training-the-model)
- [Testing and Evaluation](#testing-and-evaluation)
- [Project Structure](#project-structure)
- [Customization and Hyperparameters](#customization-and-hyperparameters)

---

## Bugs Fixed and Data Leak

### The critical bug: LSTM `batch_first` misconfiguration

**File:** `models.py` (`LRCN.forward`)

`nn.LSTM` defaults to `batch_first=False`, expecting input shaped `(seq_len, batch,
features)`. The original code fed it `(batch, 1, features)`. With this, it meant **batch dimension was read as the sequence length**. **unrelated videos within the same training batch were treated as
consecutive frames of a single video**, with information leaking between them through the
recurrent hidden state. 

**Fix:** set `batch_first=True`, and restructured the forward pass to fold time into the
batch dimension for the CNN backbone, then reshape into a correct `(batch, time, features)`
sequence before the LSTM call, processing the whole clip in one call instead of a manual
per-frame loop.

### The data leak: clips from the same source video split across train/val/test

**Files:** `video_datasets.py` (`dataset_split`, `load_dataset`)

HMDB51 clip filenames follow the pattern
`<source_video_title>_<action_class>_<body_pos>_<camera_motion>_<num_people>_<camera_view>_
<quality>_<clip_index>`. The original `dataset_split` used `StratifiedShuffleSplit` on individual clips with no awareness that multiple clips could
share a source video, letting clips of the same video end up split across train and test. A model can partially "recognize the video" it already partially saw rather than genuinely generalizing the action. To check this, I checked file structure of the videos to confirm this bug and fix it in 2 stages:

1. **Group-aware splitting.** `_video_group_id()` extracts the source-video identifier from a
   clip's filename, and `dataset_split` assigns whole *groups* (all clips of one source video)
   to exactly one split. A hard `assert` runs on every training invocation and fails loudly if
   any group is ever found spanning more than one split — confirmed clean against the real
   dataset (`cross-split overlap: 0`).
2. **Class-balanced group allocation.** A plain group split can still leave individual classes
   badly imbalanced by chance. `dataset_split` uses a two-phase greedy algorithm: every class
   is guaranteed at least one group in train as a hard floor, then remaining groups go to
   whichever split has the largest *proportional* deficit relative to its target share. This
   went through two broken iterations first (one could leave a class with zero validation
   examples, a fix for that could leave a class with zero training examples) before landing on
   the current design, verified with an adversarial stress test.

### Other bugs found and fixed
(some were suggested by LLM, as I was stuck and needed help finding the bugs affecting training when I was stuck at 40% for accuracy; these were added over time, after multiple runs of debugging)

| Bug | File | Description |
|---|---|---|
| `resnet152` placeholder | `models.py` | Selecting `resnet152` silently built `resnet34` instead. |
| Unsorted frame order | `video_datasets.py` | `glob.glob()` does not guarantee order; frames could be fed to the model out of temporal sequence. |
| Non-reproducible video ordering | `video_datasets.py` | `load_dataset`'s inner loop used unsorted `os.listdir()`, so the exact split wasn't reliably reproducible even with a fixed seed. |
| `get_frames` off-by-one | `utils.py` | Sampled one frame too many; also fixed an `int16` overflow risk on long videos. |
| Per-frame independent augmentation | `video_datasets.py` | Random flip/affine transforms were re-rolled independently per frame instead of once per clip, corrupting temporal coherence. |
| Frame-count mismatch crash | `video_datasets.py` | Videos with fewer frames than requested crashed `torch.stack` when batched with full-length clips. Fixed by always sampling exactly `fr_per_vid` indices, repeating when necessary. |
| `run_training.py` broken import | `utils.py`, `run_training.py` | Imported a `compose_dataloaders` function that was never defined anywhere. Implemented it. |
| Deprecated `pretrained=` / `verbose=` APIs | `models.py`, `run.py` | Updated to current PyTorch/torchvision APIs. |
| Mislabeling bug in `collate_fn_r3d_18` | `video_datasets.py` | Filtered images and labels *separately* then re-zipped positionally; an invalid sample in a batch silently shifted every subsequent label onto the wrong clip. Fixed by filtering (image, label) pairs together. |
| Unguarded `None` batch | `train.py`, `test.py` | An entirely-invalid batch returns `(None, None)`; downstream code crashed on it. Added a guard. |
| NumPy RNG not reseeded per DataLoader worker | `utils.py` | PyTorch reseeds `torch`/`random` per worker but not NumPy's global RNG, undermining the temporal-sampling augmentation's diversity across workers. Fixed with an explicit `worker_init_fn`. |
| Test-set results never logged to Weights & Biases | `run.py` | Added a dedicated eval `wandb` run logging test loss and accuracy. |
| Optimizer momentum staleness on LR-drop reload | `train.py` | Reloading the best checkpoint after an LR drop reset the weights but not the optimizer's momentum state, applying stale momentum to gradients from different weights. Fixed by clearing optimizer state on every reload. |
| Weight decay applied to BatchNorm and bias parameters | `run.py` | `weight_decay` applied uniformly to every parameter; fixed by splitting into decay/no-decay parameter groups by tensor dimensionality. |
| LR scheduler could decay training into uselessness | `run.py` | Added a `min_lr` floor, because repeated unproductive LR drops could decay the rate. |

Also, this is not a bug or model level improvement; but I increased the num_workers to allow parallelization operations to make training faster for me; as before, I was not getting convergence within 3-3.5 hours of training.

---

## Model-Level Improvements

I tried several other architectures (a two-stream RGB+optical-flow I3D setup, and the original LRCN baseline) before landing on X3D, which was discussed in module 7's readings.

### LRCN → X3D architecture swap

The original LRCN never learns short-range motion directly; the LSTM has to reconstruct
motion from independently-extracted static-frame embeddings, a structurally weaker signal
than a 3D convolution over space *and* time jointly. I replaced this with **X3D**, a
Kinetics-400-pretrained 3D CNN loaded via `pytorchvideo`.  X3D (~3-4M parameters, an order of magnitude smaller than a single I3D-R50 stream) built for exactly this accuracy-per-parameter tradeoff, and produced the best results of any configuration tried.

### Training improvements

- **Backbone freezing with differential learning rates.** Trains only the classification head
  for the first few epochs, then unfreezes the backbone at a tunable fraction
  (`--backbone_lr_factor`) of the head's rate. This factor had been left at an untested
  default (`0.1`) through every early experiment and during runs, I found that (`1.0`) worked best for this hyperparameter/
- **Random temporal sampling augmentation.** Training now samples a different random temporal window per video per epoch, instead of the same fixed frames every time (deterministic center-sampling is kept for validation/test).

- **Gradient clipping**, added given optimizer momentum now resets on every LR drop; this stops massive weight updates. 

- **Weights & Biases integration**: full train/val/test loss and accuracy logging, plus
  best-checkpoint artifact upload.

---

## Preprocessing and Frame Extraction

Videos are expected as pre-extracted JPEG frame sequences on disk, one directory per video,
organized by class:

```
HMDB51/
  brush_hair/
    April_09_brush_hair_u_nm_np1_ba_goo_0/
      frame0.jpg
      frame1.jpg
      ...
  cartwheel/
    ...
```

`utils.get_frames()` handles the raw-video-to-frames extraction step if starting from `.avi`
source files (uniform sampling via `cv2.VideoCapture`). At load time, `video_datasets.py`'s
`_sample_frame_indices()` selects exactly `--fr_per_vid` frames per clip: TSN-style segmented
random sampling during training (a different temporal window each epoch), deterministic
center-of-segment sampling during validation/test.

Optical flow (only needed for `--model_type i3d_two_stream`) is precomputed once via
`--mode preprocess_flow` was used in earlier runs of the project when deciding which architecture to use; but X3D does not need this and I got better results with X3D than I3D.

---

## Training the Model

```bash
python run.py \
    --frame_dir HMDB51 \
    --train_size 0.75 --test_size 0.15 \
    --model_type x3d --n_classes 51 --fr_per_vid 16 \
    --batch_size 24 --mode train --n_epochs 150 --learning_rate 2e-4 \
    --weight_decay 1e-3 --freeze_backbone_until 5 --backbone_lr_factor 1.0 \
    --grad_clip_norm 5.0 --lr_patience 8 --min_lr 1e-6 \
    --wandb_project hmdb51-x3d --run_name my-run
```

Training loads the dataset and builds the class-balanced, group-aware train/val/test split
(saved to `splits.npy` so evaluation later reuses the exact same split). Parameters are split
into four AdamW groups -- backbone-decay, backbone-no-decay, head-decay, head-no-decay -- so
`--weight_decay` only applies to actual weight matrices, never BatchNorm scale/shift or
biases; the backbone's two groups use `learning_rate * backbone_lr_factor`, the head's two
use `learning_rate` directly. If `--freeze_backbone_until N` is set, the backbone is frozen
(and its BatchNorm layers held in eval mode) for the first `N` epochs, training only the
freshly-initialized classification head, then unfrozen. Each epoch runs a standard
forward/backward pass under AMP autocast, with gradients clipped to `--grad_clip_norm` before
the optimizer step. `ReduceLROnPlateau` watches validation loss and halves the learning rate
(down to a `--min_lr` floor) after `--lr_patience` epochs with no improvement; whenever it
does, the model reloads its best-so-far checkpoint and the optimizer's momentum state is
cleared, so the next step doesn't apply momentum computed on a different, more-overfit weight
trajectory to the reloaded weights. The checkpoint with the best validation accuracy is saved
to `./models/best_model_wts.pt` and re-uploaded to Weights & Biases every time it improves.
Pass `--no_wandb` to disable logging entirely.

I found that this technique worked the best, since my model's best checkpoint sometimes increased after long epoch runs, (15+ steps), but I think this can be optimized further, so that it targets the problem of slow gains. I also think my optical flow implementation for 2stream 3DCNNs were incorrect, so that is why I decided to forgo going this route; since I switched, I have better test accuracy with X3D.

---

## Testing and Evaluation

```bash
python run.py \
    --frame_dir HMDB51 --ckpt ./models/best_model_wts.pt \
    --model_type x3d --n_classes 51 --batch_size 24 --mode eval --tta_clips 1
```

Evaluation loads the saved test split and the specified checkpoint, then reports test loss
and accuracy, which is logged to a dedicated Weights & Biases eval run alongside the training run's
train/val history. `--tta_clips` controls multi-clip test-time averaging, which was found not to be helpful but i left it implemented for testing purposesin my case, so I left this at eval to be set to 1. 

---

## Project Structure

| File | Purpose |
|---|---|
| `run.py` | entry point for argument parsing, model/optimizer construction, train/eval orchestration. |
| `train.py` | Core training loop, loss/accuracy computation, gradient clipping, checkpointing. |
| `test.py` | Test-set evaluation, multi-clip probability averaging, classification report/confusion matrix helpers. |
| `models.py` | Model definitions: `LRCN`, `I3DStream`/`TwoStreamI3D`, `X3DStream`; we only called X3D in our case, which is single-stream architecture. |
| `video_datasets.py` | `VideoDataset`/`TwoStreamVideoDataset`, group-aware class-balanced `dataset_split`, collate functions. |
| `utils.py` | Frame extraction, optical flow computation (used for I3D two stream not for this), data transforms, DataLoader construction. |
| `run_training.py` | Simpler legacy entry point (LRCN-only, no wandb/freeze/clipping features). |
| `requirements.txt` | Python dependencies. |

---

## Customization and Hyperparameters

| Flag | Default | Description |
|---|---|---|
| `--frame_dir` | -- | Directory of extracted video frames. |
| `--train_size` / `--test_size` | `0.75` / `0.15` | Split proportions (remainder is validation). |
| `--fr_per_vid` | `16` | Frames sampled per clip. |
| `--n_classes` | `51` | Number of action classes. |
| `--ckpt` | -- | Checkpoint path (for `--mode eval`). |
| `--model_type` | `x3d` | -- |
| `--mode` | `train` | `train`, `eval`, or `preprocess_flow`. |
| `--batch_size` | `24` | Mini-batch size. |
| `--learning_rate` | `2e-4` | Head learning rate. |
| `--weight_decay` | `1e-3` | Applied only to weight matrices, not BatchNorm/bias (see Bugs Fixed). |
| `--n_epochs` | `150` | Training epochs. |
| `--wandb_project` / `--run_name` | `hmdb51-x3d` / `x3d-backbone-lr-1.0` | Weights & Biases naming. |
| `--no_wandb` | off | Disable W&B logging. |
| `--freeze_backbone_until` | `5` | Epochs to keep the backbone frozen (0 = never). |
| `--backbone_lr_factor` | `1.0` | Backbone LR = `learning_rate * backbone_lr_factor`. Confirmed optimal at `1.0` (see Model-Level Improvements). |
| `--grad_clip_norm` | `5.0` | Max gradient norm (0 disables). |
| `--lr_patience` | `8` | Epochs of no improvement before the scheduler halves the LR. |
| `--min_lr` | `1e-6` | Floor the scheduler won't decay the head LR below. |
| `--label_smoothing` | `0.1` | CrossEntropyLoss label smoothing (not applied to `i3d_two_stream`). |
| `--tta_clips` | `1` | Multi-clip test-time averaging at eval (was tested during this project, but found that it was best to just not implement it and left this variable at 1). |
| Loss function | `nn.CrossEntropyLoss(reduction='sum', label_smoothing=0.1)` | Selected in `run_train` based on `model_type`; `x3d` (and `lrcn`/`3dcnn`) use `CrossEntropyLoss`, while `i3d_two_stream` uses `NLLLoss` instead (since its forward pass outputs log-probabilities (this was tested but not used for this training), and `NLLLoss` doesn't support `label_smoothing`) |

# Conclusion

I did not achieve 85% test accuracy. I achieved 78.81; I was off by 6 ish points when I tested it. There could be some hyperparameters that I did not tune well, but I think this is the best performance I reached with these hyperparameters set in training script. Furthermore, throughout my code, I have remnants of classes and functions for different architectures I tested but did not end up choosing for evaluation. The ones listed here for X3D were used during training, as I found that it aimts to determine how many parameters to use for efficient video recognition. Random temporal subsampling also helped these gains, since sampling a different window of frames each epoch instead of the same fixed frames every time meant the model saw more of each video's temporal variation over the course of training, rather than memorizing one fixed snapshot of it. In one of the papers for Module 7, it was mentioned that X3D iteratively expands 6 dimensions to find the optimal tradeoff to make the model larger with the best tradeoff; it trains with 30 tiny models that require less multiply add operations than training one large networks; this beats my previous architecture's accuracy, but gains are slow. This is seen as I experimented with the code, I needed more training for convergence, but even with more training, gains are very slow and only kick in after many epoch steps after a few decay steps.

** Note: I have remnants of code from several different model architectures we discussed in class; they are not called during training or eval, they are just there in the repo, because I didn't want to risk deleting code that is currently working in case something breaks. This is why in some of my customization, I try to explain my code and why I chose certain hyperparameters to tune for this project.

For this code, I ran it for 150 epochs on High Ram on A100 GPUs on colab; I cloned the repo in colab and set up the frame directory, then ran the python bash scripts for training and evaluation. Logging screenshots are attached to submission. 