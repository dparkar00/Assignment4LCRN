"""
Module: train.py

This module provides functions to train and evaluate a video classification model.
It includes the main training loop that tracks loss and accuracy over epochs,
updates the best performing model based on validation accuracy, and provides helper
functions to compute the learning rate, batch loss, and epoch loss.

Functions:
    - train: Runs the training and validation loops over a specified number of epochs.
    - get_learning_rate: Retrieves the current learning rate from an optimizer.
    - batch_correct_preds: Computes the number of correct predictions in a mini-batch.
    - get_batch_loss: Computes the loss for a mini-batch and performs backpropagation.
    - get_epoch_loss: Computes the average loss and accuracy over an epoch.
"""

import os
import copy
from tqdm import tqdm
import torch
import wandb


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
# train() is the top-level training-loop entry point; each argument (data, model, loss,
# optimizer, scheduler, device, checkpoint dir, epoch count) is independently meaningful and
# bundling them into a config object would obscure the function's contract more than it helps.
def train(dataloaders, model, criterion, optimizer, scheduler, device,
          optim_model_wts_dir, n_epochs=30, use_wandb=False, freeze_backbone_until=0,
          grad_clip_norm=None):
    """
    Train and validate the model over a given number of epochs.
    
    This function performs the training loop for a video classification model.
    It iterates through the specified number of epochs, updates model weights
    using backpropagation, and evaluates model performance on a validation set.
    The model with the best validation accuracy is saved to disk.

    Args:
        dataloaders (dict): Dictionary containing 'train' and 'val' DataLoaders.
        model (torch.nn.Module): The video classification model to be trained. If
                                  freeze_backbone_until > 0, the caller is responsible for
                                  calling model.freeze_backbone() before passing it in.
        criterion (callable): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler, which adjusts the
                                               learning rate based on validation loss.
        device (torch.device): Device (CPU or GPU) on which to perform training.
        optim_model_wts_dir (str): Directory to save the best model weights.
        n_epochs (int, optional): Number of training epochs. Default is 30.
        use_wandb (bool, optional): If True, log per-epoch metrics and the best checkpoint to
                                     the currently active wandb run (the caller is responsible
                                     for wandb.init()/wandb.finish()). Default is False.
        freeze_backbone_until (int, optional): If > 0, the model's backbone is expected to
                                                already be frozen (see model.freeze_backbone())
                                                for the first freeze_backbone_until epochs; this
                                                function calls model.unfreeze_backbone() right
                                                before that many epochs have elapsed. 0 disables
                                                this (the model trains as given). Default is 0.

    Returns:
        tuple: (model, loss_hist, acc_hist)
            - model (torch.nn.Module): The trained model loaded with the best validation weights.
            - loss_hist (dict): Dictionary containing lists of training and validation losses
                                 for each epoch.
            - acc_hist (dict): Dictionary containing lists of training and validation accuracies
                                for each epoch.
    """
    loss_hist = {'train': [], 'val': []}
    acc_hist = {'train': [], 'val': []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0

    # Mixed-precision training: only enabled on CUDA, where tensor cores give a real speedup;
    # on CPU it has no benefit and GradScaler would just add overhead, so it's a no-op there.
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')

    for epoch in range(n_epochs):
        if freeze_backbone_until > 0 and epoch == freeze_backbone_until:
            model.unfreeze_backbone()
            print(f'Unfreezing backbone at epoch {epoch+1} '
                  f'(was frozen for the first {freeze_backbone_until} epochs).')

        current_lr = get_learning_rate(optimizer)
        all_lrs = ', '.join(f"{pg['lr']:.2e}" for pg in optimizer.param_groups)
        print(f'Epoch {epoch+1}/{n_epochs}; '
              f'Current learning rate(s) [backbone x2, head x2 (decay/no-decay)]: {all_lrs}')

        # Training phase
        model.train()
        train_loss, train_accuracy = get_epoch_loss(
            model, criterion, dataloaders['train'], device, optimizer, scaler,
            grad_clip_norm=grad_clip_norm)
        loss_hist['train'].append(train_loss)
        acc_hist['train'].append(train_accuracy)

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_loss, val_accuracy = get_epoch_loss(
                model, criterion, dataloaders['val'], device, scaler=scaler)
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
            best_model_name = 'best_model_wts.pt'
            best_model_path = os.path.join(optim_model_wts_dir, best_model_name)
            torch.save(best_model_wts, best_model_path)
            print(f'Best model weights are updated at epoch {epoch+1}!')
            if use_wandb:
                wandb.save(best_model_path)
        loss_hist['val'].append(val_loss)
        acc_hist['val'].append(val_accuracy)

        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        if current_lr != get_learning_rate(optimizer):
            print('Loading best model weights!')
            model.load_state_dict(best_model_wts)
            # Also reset the optimizer's per-parameter momentum state (AdamW's exp_avg/
            # exp_avg_sq). Without this, those running estimates -- computed from gradients
            # of the DIFFERENT (more overfit) weight trajectory that existed right before the
            # reload -- get applied to gradients from the newly-reloaded weights on the very
            # next step, a stale-momentum mismatch right when the lower LR is meant to
            # stabilize training, not destabilize it. Clearing optimizer.state (while leaving
            # param_groups, i.e. the LR/weight_decay settings, untouched) makes AdamW
            # reinitialize these from zero on the next step, as if starting fresh at the
            # reloaded weights.
            for param_group in optimizer.param_groups:
                for param in param_group['params']:
                    optimizer.state[param] = {}

        if use_wandb:
            log_dict = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'best_val_accuracy': best_val_acc,
            }
            # Log every optimizer param group's LR individually (e.g. backbone vs head when
            # --backbone_lr_factor is used), not just the first group -- logging only
            # current_lr here would show the same misleading single value we fixed in the
            # console print above.
            for group_idx, param_group in enumerate(optimizer.param_groups):
                log_dict[f'learning_rate_group_{group_idx}'] = param_group['lr']
            wandb.log(log_dict)

        print(f"train loss: {train_loss:.6f}, val loss: {val_loss:.6f}, "
              f"accuracy: {100*val_accuracy:.2f}")
        print("-" * 60)
        print()

    # Load the best model weights before returning the model
    model.load_state_dict(best_model_wts)
    return model, loss_hist, acc_hist


def get_learning_rate(optimizer):
    """
    Retrieve the current learning rate from the optimizer.
    
    Args:
        optimizer (torch.optim.Optimizer): The optimizer from which to get the learning rate.
    
    Returns:
        float: The current learning rate.
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def batch_correct_preds(output, target):
    """
    Compute the number of correct predictions for a mini-batch.
    
    Args:
        output (torch.Tensor): Model outputs (logits) with shape (batch_size, num_classes).
        target (torch.Tensor): True labels with shape (batch_size).
    
    Returns:
        int: Number of correct predictions in the mini-batch.
    """
    pred = output.argmax(dim=1, keepdim=True)
    correct_preds = pred.eq(target.view_as(pred)).sum().item()
    return correct_preds


def get_batch_loss(criterion, output, target, model=None, optimizer=None, scaler=None,
                    grad_clip_norm=None):
    """
    Compute the loss for a mini-batch and perform backpropagation (if optimizer is provided).
    
    Args:
        criterion (callable): Loss function.
        output (torch.Tensor): Model outputs for the mini-batch.
        target (torch.Tensor): True labels for the mini-batch.
        model (torch.nn.Module, optional): The model being trained; only needed (along with
                                            optimizer) if grad_clip_norm is set, since clipping
                                            needs model.parameters().
        optimizer (torch.optim.Optimizer, optional): Optimizer to update model weights. If
                                                       None, no backpropagation is performed.
        scaler (torch.cuda.amp.GradScaler, optional): Gradient scaler for mixed-precision
                                                        training. If None or disabled, falls
                                                        back to a plain backward()/step().
        grad_clip_norm (float, optional): If set, clips the global gradient norm to this value
                                           before the optimizer step -- a standard stabilizer
                                           against occasional large-gradient batches, and
                                           particularly relevant now that train() resets
                                           optimizer momentum on every LR drop (see train()),
                                           since a freshly-reset Adam state has less smoothing
                                           for its first few post-reset steps. Under AMP, the
                                           scaled gradients are explicitly unscaled first --
                                           clipping against the scaled values would clip at the
                                           wrong threshold entirely. None or 0 disables this.
    
    Returns:
        tuple: (loss_value, n_batch_correct_preds)
            - loss_value (float): Loss value for the mini-batch.
            - n_batch_correct_preds (int): Number of correct predictions in the mini-batch.
    """
    loss = criterion(output, target)
    with torch.no_grad():
        n_batch_correct_preds = batch_correct_preds(output, target)
    if optimizer:
        optimizer.zero_grad()
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
    return loss.item(), n_batch_correct_preds


def get_epoch_loss(model, criterion, dataloader, device, optimizer=None, scaler=None,
                    grad_clip_norm=None):
    """
    Compute the average loss and overall accuracy for an epoch.

    Iterates over the entire DataLoader, computes loss and accuracy for each mini-batch,
    and aggregates the results over the epoch.

    Args:
        model (torch.nn.Module): The video classification model.
        criterion (callable): Loss function.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        device (torch.device): Device (CPU or GPU) on which to perform computations.
        optimizer (torch.optim.Optimizer, optional): If provided, used to update model weights
                                                       during training.
        scaler (torch.cuda.amp.GradScaler, optional): Gradient scaler for mixed-precision
                                                        training; also controls whether the
                                                        forward pass runs under autocast.
        grad_clip_norm (float, optional): Passed through to get_batch_loss; see there for
                                           details. Only has an effect when optimizer is set
                                           (i.e. during training, not validation/eval).

    Returns:
        tuple: (loss, accuracy)
            - loss (float): Average loss over the epoch.
            - accuracy (float): Overall accuracy over the epoch.
    """
    running_loss, running_total_correct_preds = 0.0, 0.0
    len_dataset = len(dataloader.dataset)
    amp_enabled = scaler is not None and scaler.is_enabled()

    for x_batch, y_batch in tqdm(dataloader):
        if y_batch is None:
            # Every sample in this batch was invalid (e.g. a video with zero extractable
            # frames) -- the collate function returns (None, None) in that case. Skip rather
            # than crash on .to(device); this should be rare in practice given the frame
            # sampling always produces fpv frames whenever at least one exists.
            continue
        y_batch = y_batch.to(device)
        if isinstance(x_batch, (tuple, list)):
            # Two-stream input (e.g. RGB + optical flow): move each stream to device separately.
            x_batch = tuple(x.to(device) for x in x_batch)
        else:
            x_batch = x_batch.to(device)

        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            output = model(*x_batch) if isinstance(x_batch, tuple) else model(x_batch)

        batch_loss, n_batch_correct_preds = get_batch_loss(
            criterion, output, y_batch, model=model, optimizer=optimizer, scaler=scaler,
            grad_clip_norm=grad_clip_norm)

        running_loss += batch_loss
        running_total_correct_preds += n_batch_correct_preds

    loss = running_loss / float(len_dataset)
    accuracy = running_total_correct_preds / float(len_dataset)
    return loss, accuracy
