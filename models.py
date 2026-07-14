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

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
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

class ConvLSTMCell(nn.Module):
    """
    Model improvement: a single ConvLSTM cell (Shi et al., 2015).

    A standard LSTM's gates are fully-connected, so it can only operate on flat
    feature vectors -- which means the CNN backbone must fully collapse each
    frame's spatial structure (via global average pooling) *before* any temporal
    modeling happens, and the LSTM only ever sees a sequence of already-pooled
    vectors "tacked on" after the CNN.

    ConvLSTM replaces the fully-connected gates with convolutions, so it can
    operate directly on (C, H, W) spatial feature maps. This lets recurrence be
    woven directly into the spatial feature representation at every timestep,
    instead of being applied only after the CNN has already discarded spatial
    structure via pooling.
    """
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        # A single convolution produces all four gates (input, forget, output,
        # candidate) at once, taking the concatenation of the current input map
        # and the previous hidden state map as its input.
        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.hidden_channels = hidden_channels

    def forward(self, x_t, h_prev, c_prev):
        """
        Args:
            x_t (Tensor): Input feature map for the current timestep, (B, C_in, H, W).
            h_prev (Tensor): Previous hidden state map, (B, C_hidden, H, W).
            c_prev (Tensor): Previous cell state map, (B, C_hidden, H, W).

        Returns:
            tuple: (h_t, c_t), the updated hidden and cell state maps.
        """
        combined = torch.cat([x_t, h_prev], dim=1)
        gates = self.conv(combined)
        in_gate, forget_gate, out_gate, cand_gate = torch.chunk(gates, 4, dim=1)
        in_gate = torch.sigmoid(in_gate)
        forget_gate = torch.sigmoid(forget_gate)
        out_gate = torch.sigmoid(out_gate)
        cand_gate = torch.tanh(cand_gate)

        c_t = forget_gate * c_prev + in_gate * cand_gate
        h_t = out_gate * torch.tanh(c_t)
        return h_t, c_t

class LRCN(nn.Module):  # pylint: disable=too-few-public-methods,too-many-instance-attributes
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
        use_attention (bool, optional): Model improvement -- instead of using only the
            LSTM's final hidden state, learn an attention weight over every timestep's
            output and pool them into a weighted context vector. Useful when the most
            discriminative motion in a clip doesn't occur at the very last frame.
        use_conv_lstm (bool, optional): Model improvement -- replace the standard
            fully-connected LSTM (applied only after full spatial pooling) with a
            ConvLSTM that operates directly on the CNN's spatial feature maps at
            every timestep, weaving recurrence throughout the model instead of
            tacking it on at the end. Mutually exclusive with bidirectional/
            use_attention, which apply to the standard LSTM path.
        conv_lstm_hidden_channels (int, optional): Number of hidden channels in the
            ConvLSTM cell, used only when use_conv_lstm=True. Default is 128.

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
        use_attention=False,
        use_conv_lstm=False,
        conv_lstm_hidden_channels=128,
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

        # Model improvement: use_conv_lstm needs the CNN's raw spatial feature maps
        # (C, H, W), not the fully pooled+flattened vector base_model produces. Build
        # a second view of the backbone that stops right before its own internal
        # average pool, so both paths below can share the same underlying weights.
        self.use_conv_lstm = use_conv_lstm
        if use_conv_lstm:
            # All backbone children except the final adaptive avg pool and fc layer.
            self.spatial_extractor = nn.Sequential(*list(base_cnn.children())[:-2])

        # Model improvement: optional partial backbone freezing for more stable,
        # cheaper fine-tuning on a modest-sized dataset.
        # BUG FIX: children() includes parameterless modules (avgpool, the Identity
        # fc replacement, etc.). Counting those toward "last N children" meant
        # freeze_backbone_until=N could leave only parameterless modules trainable,
        # silently freezing the entire backbone regardless of N. Only children with
        # at least one parameter are eligible to be counted/unfrozen.
        if freeze_backbone_until is not None:
            children = [c for c in self.base_model.children()
                        if any(True for _ in c.parameters())]
            n_freeze = max(0, len(children) - freeze_backbone_until)
            for child in children[:n_freeze]:
                for param in child.parameters():
                    param.requires_grad = False

        if use_conv_lstm:
            # Model improvement: ConvLSTM path. Recurrence is woven directly into
            # the spatial feature maps at every timestep instead of being applied
            # only after full pooling -- see ConvLSTMCell docstring for rationale.
            self.conv_lstm_hidden_channels = conv_lstm_hidden_channels
            self.conv_lstm_cell = ConvLSTMCell(num_features, conv_lstm_hidden_channels)
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.dropout = nn.Dropout(dropout_rate)
            self.fc = nn.Linear(conv_lstm_hidden_channels, n_classes)
        else:
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
            # Model improvement: optional attention pooling over LSTM timestep outputs,
            # instead of relying only on the final hidden state (see forward()).
            self.use_attention = use_attention
            if use_attention:
                self.attention = nn.Linear(rnn_out_size, 1)

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

        if self.use_conv_lstm:
            # Model improvement: ConvLSTM path. Extract spatial feature maps per
            # frame (no pooling yet), then run the ConvLSTM cell over time directly
            # on those maps -- recurrence is applied at every spatial location,
            # woven into the feature representation rather than only after the
            # CNN has already discarded spatial structure via pooling.
            x = x.view(bs * ts, c, h, w)
            feat_maps = self.spatial_extractor(x)
            _, fc_channels, fh, fw = feat_maps.shape
            feat_maps = feat_maps.view(bs, ts, fc_channels, fh, fw)

            h_t = torch.zeros(bs, self.conv_lstm_hidden_channels, fh, fw, device=x.device)
            c_t = torch.zeros(bs, self.conv_lstm_hidden_channels, fh, fw, device=x.device)
            for t in range(ts):
                h_t, c_t = self.conv_lstm_cell(feat_maps[:, t], h_t, c_t)

            pooled = self.global_pool(h_t).flatten(1)
            out = self.dropout(pooled)
            out = self.fc(out)
            return out

        # FIX: vectorize CNN feature extraction across all frames in one call
        # instead of a Python for-loop over time steps (also much faster on GPU).
        x = x.view(bs * ts, c, h, w)
        feats = self.base_model(x)
        feats = feats.view(bs, ts, -1)

        if lengths is not None:
            packed = pack_padded_sequence(
                feats, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, (hn, _cn) = self.rnn(packed)
            # Recover per-timestep outputs (needed for attention pooling below).
            # Padded timesteps come back as zeros and are masked out explicitly.
            rnn_out, out_lengths = pad_packed_sequence(packed_out, batch_first=True)
        else:
            rnn_out, (hn, _cn) = self.rnn(feats)
            out_lengths = torch.full((bs,), ts, dtype=torch.long, device=x.device)

        if self.use_attention:
            # Model improvement: attention pooling. Score every timestep's LSTM
            # output, mask out padded positions so they can never receive weight,
            # softmax over time, then take the weighted sum as the context vector
            # -- lets the model emphasize whichever frames are most discriminative
            # instead of always trusting only the final timestep.
            attn_scores = self.attention(rnn_out).squeeze(-1)  # (batch, seq)
            time_idx = torch.arange(rnn_out.size(1), device=rnn_out.device)
            mask = time_idx[None, :] < out_lengths.to(rnn_out.device)[:, None]
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
            attn_weights = torch.softmax(attn_scores, dim=1)
            last_hidden = torch.sum(rnn_out * attn_weights.unsqueeze(-1), dim=1)
        elif self.rnn.bidirectional:
            # hn shape: (num_layers * num_directions, batch, hidden_size).
            # Take the last layer's hidden state(s).
            last_hidden = torch.cat([hn[-2], hn[-1]], dim=1)
        else:
            last_hidden = hn[-1]

        # Apply dropout to the final hidden state.
        out = self.dropout(last_hidden)
        # Pass the final output through the fully-connected layer to get class logits.
        out = self.fc(out)
        return out
