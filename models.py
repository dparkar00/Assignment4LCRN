"""
Module: models.py

This module defines the LRCN (Long-term Recurrent Convolutional Network) model for video
classification. The LRCN model combines a 2D CNN backbone (e.g., ResNet) for spatial feature
extraction from individual frames with an LSTM to capture temporal dynamics across frames.
An additional fully-connected layer is used to output the final class predictions.

Classes:
    Identity: A helper module that returns the input unchanged. It is used to replace the
              fully-connected layer of the ResNet backbone.
    LRCN: The main model that integrates the CNN backbone, an LSTM, dropout regularization,
          and a final fully-connected layer to produce class logits.
"""

import torch
from torch import nn

from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import models

class Identity(nn.Module):  # pylint: disable=too-few-public-methods
    """
    A placeholder identity operator that is argument-insensitive.

    This module is used to replace the fully-connected (fc) layer in the ResNet backbone,
    effectively making the backbone output the raw features before classification.

    Example:
        >>> identity = Identity()
        >>> output = identity(input_tensor)
    """

    def forward(self, x):
        """
        Forward pass that returns the input as is.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: The same tensor x.
        """
        return x

_BACKBONES = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
}

class LRCN(nn.Module):  # pylint: disable=too-few-public-methods
    """
    LRCN (Long-term Recurrent Convolutional Network) for video classification.

    This model uses a ResNet backbone as a 2D CNN to extract spatial features from each
    video frame. An LSTM network is then used to model the temporal dynamics across the
    sequence of frame features. Dropout is applied before the final fully-connected layer
    that produces class logits.

    Args:
        hidden_size (int): Number of features in the hidden state of the LSTM.
        n_layers (int): Number of recurrent layers in the LSTM.
        dropout_rate (float): Dropout rate applied before the final classification layer.
        n_classes (int): Number of output classes.
        pretrained (bool, optional): If True, uses a ResNet model pretrained on ImageNet.
            Default is True.
        cnn_model (str, optional): Specifies the ResNet variant to use as the backbone.
            Options: 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'.
            Default is 'resnet34'.
        freeze_backbone_until (int, optional): If set, freezes all backbone parameters
            except the last N children modules, for cheaper/more stable fine-tuning
            (model-level improvement).
        bidirectional (bool, optional): Use a bidirectional LSTM (model-level improvement).

    Raises:
        ValueError: If the specified cnn_model is not supported.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    def __init__(
        self,
        hidden_size,
        n_layers,
        dropout_rate,
        n_classes,
        pretrained=True,
        cnn_model="resnet34",
        freeze_backbone_until=None,
        bidirectional=False,
    ):
        super().__init__()

        if cnn_model not in _BACKBONES:
            raise ValueError(
                f"Unsupported cnn_model '{cnn_model}'. Choose from {list(_BACKBONES)}."
            )

        base_cnn = _BACKBONES[cnn_model](pretrained=pretrained)

        # Retrieve the number of features output by the CNN's original fully-connected layer.
        num_features = base_cnn.fc.in_features

        # Replace the original fc layer with an identity mapping so raw features are returned.
        base_cnn.fc = Identity()
        self.base_model = base_cnn

        # Model improvement: optional partial backbone freezing for more stable,
        # cheaper fine-tuning on a modest-sized dataset.
        if freeze_backbone_until is not None:
            children = list(self.base_model.children())
            n_freeze = max(0, len(children) - freeze_backbone_until)
            for child in children[:n_freeze]:
                for param in child.parameters():
                    param.requires_grad = False

        # Define the LSTM to process the sequence of frame features.
        # FIX: batch_first=True so the LSTM interprets input as (batch, seq, feat).
        self.rnn = nn.LSTM(
            num_features,
            hidden_size,
            n_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        # A bidirectional LSTM concatenates its forward and backward final hidden
        # states, doubling the feature width that reaches the classifier head.
        rnn_out_size = hidden_size * (2 if bidirectional else 1)
        # Define dropout for regularization.
        self.dropout = nn.Dropout(dropout_rate)
        # Final fully-connected layer to produce logits for each class.
        # FIX: this must consume rnn_out_size, not hidden_size -- using hidden_size
        # here caused a shape-mismatch RuntimeError whenever bidirectional=True,
        # since the LSTM's actual output width is hidden_size * 2 in that case.
        self.fc = nn.Linear(rnn_out_size, n_classes)

    def forward(self, x, lengths=None):  # pylint: disable=too-many-locals
        """
        Forward pass for the LRCN model.

        The input tensor x is expected to have the shape:
            (batch_size, time_steps, channels, height, width)

        The CNN backbone extracts features from all frames in a single batched call,
        then the full sequence is passed through the LSTM. The final hidden state(s)
        are passed through dropout and the final fully-connected layer to produce
        the class logits.

        Args:
            x (Tensor): Input tensor of shape (batch_size, time_steps, channels, height, width).
            lengths (Tensor, optional): True (unpadded) sequence length per sample. When
                provided, padded frames are excluded from the LSTM computation via
                pack_padded_sequence instead of being fed through as garbage input.

        Returns:
            Tensor: Output logits for each sample in the batch with shape (batch_size, n_classes).
        """
        bs, ts, c, h, w = x.shape  # batch_size, time_steps, channels, height, width

        # FIX: vectorize CNN feature extraction across all frames in one call
        # instead of a Python for-loop over time steps (also much faster on GPU).
        x = x.view(bs * ts, c, h, w)
        feats = self.base_model(x)
        feats = feats.view(bs, ts, -1)

        if lengths is not None:
            packed = pack_padded_sequence(
                feats, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _packed_out, (hn, _cn) = self.rnn(packed)
        else:
            _out, (hn, _cn) = self.rnn(feats)

        # hn shape: (num_layers * num_directions, batch, hidden_size).
        # Take the last layer's hidden state(s).
        if self.rnn.bidirectional:
            last_hidden = torch.cat([hn[-2], hn[-1]], dim=1)
        else:
            last_hidden = hn[-1]

        # Apply dropout to the final hidden state.
        out = self.dropout(last_hidden)
        # Pass the final output through the fully-connected layer to get class logits.
        out = self.fc(out)
        return out
