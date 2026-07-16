"""
Module: test.py

This module provides functions for evaluating a video classification model on test data.
It includes functions to compute predictions and accuracy, generate a detailed classification
report, and compute a multilabel confusion matrix for all classes.

Functions:
    - test: Evaluates the model on a test DataLoader and returns the ground truth labels,
      predicted labels, and overall accuracy.
    - predict_probs: Returns per-sample class probabilities (not just the argmax prediction),
      used for multi-clip test-time averaging (see run.py's run_eval).
    - get_test_report: Generates a classification report using scikit-learn's
      classification_report.
    - get_confusion_matrix: Computes a multilabel confusion matrix for each class.
"""

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, multilabel_confusion_matrix


def predict_probs(model, dataloader, device, is_log_prob):
    """
    Run the model over a DataLoader and return per-sample class PROBABILITIES (not just the
    argmax prediction) and targets, in dataloader iteration order.

    Used for multi-clip test-time averaging: each call samples one clip per video, and
    run.py's run_eval calls this multiple times (each pass sampling a different random
    temporal window of each test video) and averages the resulting probabilities before taking
    the final argmax -- standard practice for evaluating video classifiers, since judging
    accuracy on a single arbitrary clip per video is higher-variance than averaging over
    several.

    Args:
        model (torch.nn.Module): The trained model, already moved to device and in eval mode.
        dataloader (torch.utils.data.DataLoader): DataLoader to run inference over. Must NOT
                                                    shuffle, so sample order is consistent
                                                    across repeated calls (needed to align and
                                                    average probabilities across passes).
        device (torch.device): Device to run inference on.
        is_log_prob (bool): True if the model's forward() returns log-probabilities (e.g.
                             TwoStreamI3D, which fuses two streams in probability space), False
                             if it returns raw logits (e.g. LRCN, paired with CrossEntropyLoss).

    Returns:
        tuple: (probs, targets)
            - probs (np.ndarray): Array of shape (n_samples, n_classes) of class probabilities.
            - targets (list): Ground truth labels, same order as probs.
    """
    amp_enabled = device.type == 'cuda'
    batch_probs, targets = [], []
    with torch.no_grad():
        for x_batch, y_batch in tqdm(dataloader):
            if y_batch is None:
                # Every sample in this batch was invalid (e.g. a video with zero extractable
                # frames) -- the collate function returns (None, None) in that case.
                continue
            y_batch = y_batch.to(device)
            if isinstance(x_batch, (tuple, list)):
                x_batch = tuple(x.to(device) for x in x_batch)
            else:
                x_batch = x_batch.to(device)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                output = model(*x_batch) if isinstance(x_batch, tuple) else model(x_batch)
            probs = torch.exp(output) if is_log_prob else torch.softmax(output, dim=1)
            batch_probs.append(probs.detach().cpu().numpy())
            targets.extend(y_batch.detach().cpu().numpy().tolist())
    return np.concatenate(batch_probs, axis=0), targets


def test(model, dataloader, device):
    """
    Evaluate the model on the test dataset and compute overall accuracy.
    
    This function sets the model to evaluation mode and processes the test data
    from the provided DataLoader. It computes predictions for each batch, counts the number
    of correct predictions, and accumulates the true and predicted labels.
    
    Args:
        model (torch.nn.Module): The trained video classification model.
        dataloader (torch.utils.data.DataLoader): DataLoader containing the test dataset.
        device (torch.device): The device (CPU or GPU) on which to perform evaluation.
    
    Returns:
        tuple: (targets, outputs, accuracy)
            - targets (list): Ground truth labels for all samples.
            - outputs (list): Predicted labels for all samples.
            - accuracy (float): Overall accuracy computed as the ratio of correct predictions
                                to the total number of samples.
    """
    model.eval()
    with torch.no_grad():
        total_correct_preds = 0.0
        len_dataset = len(dataloader.dataset)
        targets, outputs = [], []
        amp_enabled = device.type == 'cuda'
        for x_batch, y_batch in tqdm(dataloader):
            if y_batch is None:
                # Every sample in this batch was invalid (e.g. a video with zero extractable
                # frames) -- the collate function returns (None, None) in that case.
                continue
            y_batch = y_batch.to(device)
            if isinstance(x_batch, (tuple, list)):
                # Two-stream input (e.g. RGB + optical flow): move each stream to device
                # separately.
                x_batch = tuple(x.to(device) for x in x_batch)
            else:
                x_batch = x_batch.to(device)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                output = model(*x_batch) if isinstance(x_batch, tuple) else model(x_batch)
            pred = output.argmax(dim=1, keepdim=True)
            correct_preds = pred.eq(y_batch.view_as(pred)).sum().item()
            total_correct_preds += correct_preds
            outputs.extend(pred.view(-1).detach().cpu().numpy().tolist())
            targets.extend(y_batch.detach().cpu().numpy().tolist())

        accuracy = total_correct_preds / float(len_dataset)

    return targets, outputs, accuracy


def get_test_report(target, output, target_names):
    """
    Generate a detailed classification report based on test results.
    
    This function uses scikit-learn's classification_report to produce a dictionary
    containing precision, recall, F1-score, and support for each class.
    
    Args:
        target (list): Ground truth labels.
        output (list): Predicted labels.
        target_names (list): List of class names corresponding to the labels.
    
    Returns:
        dict: A classification report as a dictionary.
    """
    return classification_report(target, output, output_dict=True, target_names=target_names)


def get_confusion_matrix(targets, outputs, labels_dict, all_cats):
    """
    Compute a multilabel confusion matrix for each class.
    
    This function converts numeric labels to their corresponding class names using the provided
    labels_dict, then computes a multilabel confusion matrix for each class using scikit-learn.
    
    Args:
        targets (list): Ground truth numeric labels.
        outputs (list): Predicted numeric labels.
        labels_dict (dict): Dictionary mapping class names to numeric labels.
        all_cats (list): List of all class names.
    
    Returns:
        dict: A dictionary where keys are class names and values are the corresponding
              confusion matrices.
    """
    # Create an inverse mapping from numeric label to class name
    inv_labels_dict = {label: cat for cat, label in labels_dict.items()}
    target_cats = [inv_labels_dict[target] for target in targets]
    output_cats = [inv_labels_dict[output] for output in outputs]
    confusion_mat = multilabel_confusion_matrix(target_cats, output_cats, labels=all_cats)
    return dict(zip(all_cats, confusion_mat))
