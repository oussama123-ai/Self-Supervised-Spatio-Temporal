"""
Combined Loss Function for SSS-TT (Eq. 9 in paper):

  L_total = L_CORAL + 0.1 * L_temp + 0.5 * L_MAE * 1(t ≤ 10)

Where:
  L_CORAL  — ordinal regression loss (Eq. 8)
  L_temp   — temporal consistency regularisation
  L_MAE    — reconstruction loss, active only for first 10 epochs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.coral_head import coral_loss


class TemporalConsistencyLoss(nn.Module):
    """
    Penalises large prediction changes between consecutive time windows.
    Encourages smooth pain trajectories (onset → peak → offset).
    """

    def __init__(self):
        super().__init__()

    def forward(self, temporal_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            temporal_features: (B, T, D) TCN output
        Returns:
            loss: scalar
        """
        diff = temporal_features[:, 1:, :] - temporal_features[:, :-1, :]
        return diff.pow(2).mean()


class SSTTLoss(nn.Module):
    """
    Total SSS-TT training loss combining CORAL, temporal, and MAE components.

    Args:
        lambda_temp: Weight for temporal consistency loss (0.1)
        lambda_mae:  Weight for MAE reconstruction loss (0.5)
        mae_epochs:  Number of epochs MAE loss is active (10)
        num_classes: Number of pain levels (4)
    """

    def __init__(
        self,
        lambda_temp: float = 0.1,
        lambda_mae: float = 0.5,
        mae_epochs: int = 10,
        num_classes: int = 4,
    ):
        super().__init__()
        self.lambda_temp = lambda_temp
        self.lambda_mae = lambda_mae
        self.mae_epochs = mae_epochs
        self.num_classes = num_classes
        self.temporal_loss = TemporalConsistencyLoss()

    def forward(
        self,
        cumprobs: torch.Tensor,
        targets: torch.Tensor,
        temporal_features: torch.Tensor | None = None,
        mae_loss: torch.Tensor | None = None,
        epoch: int = 0,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            cumprobs:          (B, K)  CORAL cumulative probabilities
            targets:           (B,)   integer pain labels
            temporal_features: (B, T, D)  TCN features (optional)
            mae_loss:          scalar  MAE reconstruction loss (optional)
            epoch:             current training epoch
        Returns:
            dict with 'total', 'coral', 'temporal', 'mae' losses
        """
        # Main CORAL loss
        l_coral = coral_loss(cumprobs, targets, self.num_classes)

        # Temporal consistency
        l_temp = torch.tensor(0.0, device=cumprobs.device)
        if temporal_features is not None:
            l_temp = self.temporal_loss(temporal_features)

        # MAE reconstruction (active first 10 epochs)
        l_mae = torch.tensor(0.0, device=cumprobs.device)
        if mae_loss is not None and epoch <= self.mae_epochs:
            l_mae = mae_loss

        total = (
            l_coral
            + self.lambda_temp * l_temp
            + self.lambda_mae * l_mae
        )

        return {
            'total': total,
            'coral': l_coral,
            'temporal': l_temp,
            'mae': l_mae,
        }
