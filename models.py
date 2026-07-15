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
from torchvision import models


class I3DStream(nn.Module):
    """
    Single-stream I3D (ResNet-50 3D backbone, Kinetics-400 pretrained) classifier.

    Loaded via torch.hub from the pytorchvideo model zoo. This is the modern I3D-R50 variant
    (Kinetics-pretrained weights are readily available for it, unlike the original Inception-v1
    based I3D). Used as-is for either the RGB stream or the optical-flow stream, since both are
    encoded as 3-channel clips of shape (batch, 3, time, H, W).

    Args:
        n_classes (int): Number of output classes.
        pretrained (bool): If True, load Kinetics-400 pretrained weights.
    """
    def __init__(self, n_classes, pretrained=True):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/pytorchvideo', 'i3d_r50',
                                        pretrained=pretrained)
        # Replace the Kinetics-400 head (400 classes) with one sized for this task.
        in_features = self.backbone.blocks[-1].proj.in_features
        self.backbone.blocks[-1].proj = nn.Linear(in_features, n_classes)

    def forward(self, x):
        """
        Args:
            x (Tensor): Clip tensor of shape (batch, 3, time, H, W).
        Returns:
            Tensor: Class logits of shape (batch, n_classes).
        """
        return self.backbone(x)


class TwoStreamI3D(nn.Module):
    """
    Two-stream I3D model: an RGB stream and an optical-flow stream, combined via late fusion
    (weighted average of per-stream softmax probabilities) as in the standard two-stream action
    recognition design.

    Args:
        n_classes (int): Number of output classes.
        pretrained (bool): If True, initialize both streams from Kinetics-400 pretrained weights.
        flow_weight (float): Weight given to the flow stream in the fused prediction (0-1); the
                              RGB stream gets (1 - flow_weight). Flow is weighted slightly higher
                              than RGB by default, following the two-stream literature.
    """
    def __init__(self, n_classes, pretrained=True, flow_weight=0.6):
        super().__init__()
        self.rgb_stream = I3DStream(n_classes, pretrained=pretrained)
        self.flow_stream = I3DStream(n_classes, pretrained=pretrained)
        self.flow_weight = flow_weight

    def forward(self, rgb, flow):
        """
        Args:
            rgb (Tensor): RGB clip tensor of shape (batch, 3, time, H, W).
            flow (Tensor): Optical-flow clip tensor of shape (batch, 3, time, H, W).
        Returns:
            Tensor: Fused log-probabilities of shape (batch, n_classes). Returns
                    log-probabilities (not raw logits) so it pairs with nn.NLLLoss, since the
                    fusion has to happen in probability space (a weighted average of two
                    probability distributions is not equivalent to a weighted average of two
                    logit vectors).
        """
        rgb_probs = torch.softmax(self.rgb_stream(rgb), dim=1)
        flow_probs = torch.softmax(self.flow_stream(flow), dim=1)
        fused_probs = (1 - self.flow_weight) * rgb_probs + self.flow_weight * flow_probs
        return torch.log(fused_probs + 1e-8)


class Identity(nn.Module):
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


class LRCN(nn.Module):
    """
    LRCN (Long-term Recurrent Convolutional Network) for video classification.

    This model uses a ResNet backbone as a 2D CNN to extract spatial features from each video
    frame. An LSTM network is then used to model the temporal dynamics across the sequence of
    frame features. Dropout is applied before the final fully-connected layer that produces
    class logits.

    Args:
        hidden_size (int): Number of features in the hidden state of the LSTM.
        n_layers (int): Number of recurrent layers in the LSTM.
        dropout_rate (float): Dropout rate applied before the final classification layer.
        n_classes (int): Number of output classes.
        pretrained (bool, optional): If True, uses a ResNet model pretrained on ImageNet.
                                      Default is True.
        cnn_model (str, optional): Specifies the ResNet variant to use as the backbone.
                                   Options: 'resnet18', 'resnet34', 'resnet50', 'resnet101',
                                   'resnet152'. Default is 'resnet34'.

    Raises:
        ValueError: If the specified cnn_model is not supported.
    """
    def __init__(self, hidden_size, n_layers, dropout_rate, n_classes,
                 pretrained=True, cnn_model='resnet34'):
        super().__init__()

        # Set up the ResNet backbone as a 2D CNN feature extractor.
        weights = 'DEFAULT' if pretrained else None
        if cnn_model == 'resnet18':
            base_cnn = models.resnet18(weights=weights)
        elif cnn_model == 'resnet34':
            base_cnn = models.resnet34(weights=weights)
        elif cnn_model == 'resnet50':
            base_cnn = models.resnet50(weights=weights)
        elif cnn_model == 'resnet101':
            base_cnn = models.resnet101(weights=weights)
        elif cnn_model == 'resnet152':
            # Note: This example uses resnet34 for resnet152 option as a placeholder.
            base_cnn = models.resnet152(weights=weights)
        else:
            raise ValueError('The input CNN backbone is not supported, '
                              'please choose a valid ResNet variant.')

        # Retrieve the number of features output by the CNN's original fully-connected layer.
        num_features = base_cnn.fc.in_features

        # Replace the original fc layer with an identity mapping so that raw features are returned.
        base_cnn.fc = Identity()
        self.base_model = base_cnn

        # Define the LSTM to process the sequence of frame features.
        self.rnn = nn.LSTM(num_features, hidden_size, n_layers, batch_first=True)

        # Define dropout for regularization.
        self.dropout = nn.Dropout(dropout_rate)

        # Final fully-connected layer to produce logits for each class.
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        """
        Forward pass for the LRCN model.

        The input tensor x is expected to have the shape:
            (batch_size, time_steps, channels, height, width)

        For each time step (frame), the CNN backbone extracts features. These features are then
        passed through the LSTM sequentially. The output from the last time step is then passed
        through dropout and the final fully-connected layer to produce the class logits.

        Args:
            x (Tensor): Input tensor of shape (batch_size, time_steps, channels, height, width).

        Returns:
            Tensor: Output logits for each sample in the batch with shape (batch_size, n_classes).
        """
        bs, ts, c, h, w = x.shape  # batch_size, time_steps, channels, height, width

        # Extract CNN features for every frame in the clip in one pass (folding time into batch),
        # then feed the whole (batch, time, features) sequence to the LSTM in a single call so
        # batch and time are never conflated.
        y = self.base_model(x.reshape(bs * ts, c, h, w))
        y = y.view(bs, ts, -1)
        out, _ = self.rnn(y)

        # Apply dropout to the output of the final time step.
        out = self.dropout(out[:, -1])

        # Pass the final output through the fully-connected layer to get class logits.
        out = self.fc(out)
        return out
