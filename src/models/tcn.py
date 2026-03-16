"""
Temporal Convolutional Network (TCN) for SSS-TT.

Processes the per-frame spatial embeddings from ViT to model
temporal pain evolution (onset → peak → offset).

Design:
  - 4 dilated causal convolutional layers (rates 1, 2, 4, 8)
  - Receptive field: 31 frames — sufficient for the complete pain arc (~1 s at 30 FPS)
  - Causal padding: no future-frame leakage
  - Residual connections + WeightNorm for stable training
  - Input/output: (B, T, 768)

Reference: Bai et al. (2018) "An Empirical Evaluation of Generic
Convolutional and Recurrent Networks for Sequence Modeling"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class CausalConv1d(nn.Module):
    """
    1-D causal convolution: pads left only so output at time t
    depends only on inputs ≤ t (no future leakage).
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation          # left-only pad
        self.conv = weight_norm(
            nn.Conv1d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=0,                                    # manual below
            )
        )
        nn.init.kaiming_normal_(self.conv.weight_v,
                                mode='fan_in', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)
        Returns:
            out: (B, C, T)  — same length as input
        """
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class TCNResidualBlock(nn.Module):
    """
    One TCN residual block:
        x → CausalConv → ReLU → Dropout → CausalConv → ReLU → Dropout
        out = ReLU(residual_proj(x) + block_out)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, dilation: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels,
                                  kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels,
                                  kernel_size, dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Residual 1×1 projection if dimensions differ
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.dropout(self.relu(self.conv1(x)))
        out = self.dropout(self.relu(self.conv2(out)))
        if self.downsample is not None:
            residual = self.downsample(residual)
        return self.relu(out + residual)


class TCN(nn.Module):
    """
    Temporal Convolutional Network with dilated causal convolutions.

    Args:
        in_channels:  Feature dimension from ViT (768)
        out_channels: Output feature dimension (768)
        num_layers:   Number of dilated blocks (4)
        kernel_size:  Convolution kernel size (3)
        dropout:      Dropout rate (0.1)

    Dilation schedule: [1, 2, 4, 8]
    Receptive field: 1 + (3-1)*(1+2+4+8) = 1 + 2*15 = 31 frames

    Input/Output shape: (B, T, D) — handles channel-last format
    """

    def __init__(
        self,
        in_channels: int = 768,
        out_channels: int = 768,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Dilation schedule: powers of 2
        dilations = [2 ** i for i in range(num_layers)]      # [1,2,4,8]

        layers = []
        ch = in_channels
        for d in dilations:
            layers.append(TCNResidualBlock(ch, out_channels,
                                           kernel_size, d, dropout))
            ch = out_channels
        self.network = nn.Sequential(*layers)

        self.output_norm = nn.LayerNorm(out_channels)

        # Compute and report receptive field
        rf = 1 + (kernel_size - 1) * sum(dilations)
        self._receptive_field = rf

    @property
    def receptive_field(self) -> int:
        """Number of frames in the causal receptive field."""
        return self._receptive_field

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)  — sequence of per-frame ViT embeddings
        Returns:
            out: (B, T, D)  — temporally-contextualised embeddings
        """
        # TCN expects (B, C, T)
        x = x.transpose(1, 2)                                 # (B, D, T)
        out = self.network(x)                                  # (B, D, T)
        out = out.transpose(1, 2)                              # (B, T, D)
        out = self.output_norm(out)
        return out

    def get_clip_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return a single clip-level vector by taking the last time step
        (causal — contains information from all T frames).

        Args:
            x: (B, T, D)
        Returns:
            clip_feat: (B, D)
        """
        out = self.forward(x)
        return out[:, -1, :]                                   # last frame = full context
