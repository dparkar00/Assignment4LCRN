# HMDB51 Video Classification

This repository was originally a working-but-flawed video classification pipeline (LRCN: ResNet + LSTM, trained on HMDB51). The assignment brief was explicit that the codebase contained **a data leak** and **a critically significant bug**. This README documents what was found and fixed, the model-level improvements made on top of the corrected base, and how to run the
project.

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
(some were suggested by LLM, as I was stuck and needed help finding the bugs affecting training)

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

Also, this is not a bug or model level improvement; but I increase the num_workers to allow parallelization operations to make training faster for me; as before, I was not getting convergence within 3-3.5 hours of training.

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

Training loads the dataset, builds the class-balanced group split, optionally freezes the
backbone for the first `--freeze_backbone_until` epochs, and runs the standard
train/validate loop with AdamW, mixed-precision (AMP), gradient clipping, and a
`ReduceLROnPlateau` scheduler with a `min_lr` floor. The best checkpoint (by validation
accuracy) is saved to `./models/best_model_wts.pt` and uploaded to Weights & Biases whenever
it improves. Pass `--no_wandb` to disable logging entirely.

---

## Testing and Evaluation

```bash
python run.py \
    --frame_dir HMDB51 --ckpt ./models/best_model_wts.pt \
    --model_type x3d --n_classes 51 --batch_size 24 --mode eval --tta_clips 1
```

Evaluation loads the saved test split and the specified checkpoint, then reports test loss
and accuracy -- logged to a dedicated Weights & Biases eval run alongside the training run's
train/val history. `--tta_clips` controls multi-clip test-time averaging (see Model-Level
Improvements above for why this project's final configuration uses `1`, not a higher value).

---

## Project Structure

| File | Purpose |
|---|---|
| `run.py` | CLI entry point: argument parsing, model/optimizer construction, train/eval/preprocess-flow orchestration. |
| `train.py` | Core training loop, loss/accuracy computation, gradient clipping, checkpointing. |
| `test.py` | Test-set evaluation, multi-clip probability averaging, classification report/confusion matrix helpers. |
| `models.py` | Model definitions: `LRCN`, `I3DStream`/`TwoStreamI3D`, `X3DStream`. |
| `video_datasets.py` | `VideoDataset`/`TwoStreamVideoDataset`, group-aware class-balanced `dataset_split`, collate functions. |
| `utils.py` | Frame extraction, optical flow computation, data transforms, DataLoader construction. |
| `run_training.py` | Simpler legacy entry point (LRCN-only, no wandb/freeze/clipping features). |
| `requirements.txt` | Python dependencies. |

---

## Customization and Hyperparameters

| Flag | Default | Description |
|---|---|---|
| `--frame_dir` | -- | Directory of extracted video frames. |
| `--train_size` / `--test_size` | `0.7` / `0.1` | Split proportions (remainder is validation). |
| `--fr_per_vid` | `16` | Frames sampled per clip. |
| `--n_classes` | required | Number of action classes. |
| `--ckpt` | -- | Checkpoint path (for `--mode eval`). |
| `--model_type` |  `x3d`. |
| `--mode` | required | `train`, `eval`, or `preprocess_flow`. |
| `--batch_size` | required | Mini-batch size. |
| `--learning_rate` | `2e4` | Head learning rate. |
| `--weight_decay` | `1e-3` | Applied only to weight matrices, not BatchNorm/bias (see Bugs Fixed). |
| `--n_epochs` | `30` | Training epochs. |
| `--wandb_project` / `--run_name` | -- | Weights & Biases naming. |
| `--no_wandb` | off | Disable W&B logging. |
| `--freeze_backbone_until` | `0` | Epochs to keep the backbone frozen (0 = never). |
| `--backbone_lr_factor` | `1.0` | Backbone LR = `learning_rate * backbone_lr_factor`. Confirmed optimal at `1.0` (see Model-Level Improvements). |
| `--grad_clip_norm` | `5.0` | Max gradient norm (0 disables). |
| `--lr_patience` | `5` | Epochs of no improvement before the scheduler halves the LR. |
| `--min_lr` | `1e-6` | Floor the scheduler won't decay the head LR below. |
| `--label_smoothing` | `0.1` | CrossEntropyLoss label smoothing (not applied to `i3d_two_stream`). |
| `--tta_clips` | `1` | Multi-clip test-time averaging at eval (was tested during this project, but found that it was best to just not implement it and leave this variable at 1). |

#Conclusion

I did not achieve 85% test accuracy, I was off by 8 points when I tested it. 