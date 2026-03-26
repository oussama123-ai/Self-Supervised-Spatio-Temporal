"""
CORAL Ordinal Regression Head for SSS-TT.

Pain intensity levels {0, 1, 2, 3} are ordinal: CORAL enforces rank-consistent
cumulative probabilities via shared weight vector and ordered thresholds.

Reference:
  Cao, Mirjalili & Raschka (2020). "Rank consistent ordinal regression for
  neural networks with application to age estimation." Pattern Recognition Letters.

Equations from paper:
  P(y ≤ k | x) = σ(b_k − wᵀ z_fused),  k ∈ {0, 1, 2}   (Eq. 7)

  L_CORAL = −(1/(K−1)) Σ_k [1(y>k) log P(y>k) + 1(y≤k) log P(y≤k)]  (Eq. 8)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CORALHead(nn.Module):
    """
    CORAL ordinal regression head.

    Args:
        in_features:  Input feature dimension (768)
        num_classes:  Number of ordinal levels (4 for pain 0-3)
        dropout:      Pre-classifier dropout (0.3)

    Architecture:
        z_fused → Dropout → Linear(D, 1) [shared w] + biases [b_0, b_1, b_2]
        → K-1 = 3 binary cumulative classifiers
        → P(y ≤ 0), P(y ≤ 1), P(y ≤ 2)
    """

    def __init__(self, in_features: int = 768,
                 num_classes: int = 4,
                 dropout: float = 0.3):
        super().__init__()
        self.num_classes = num_classes
        self.K = num_classes - 1                               # 3 binary tasks

        self.dropout = nn.Dropout(dropout)

        # Shared weight vector (one linear unit, no bias)
        self.fc = nn.Linear(in_features, 1, bias=False)

        # K-1 learnable bias thresholds (ordered not enforced by param,
        # but by the CORAL loss — see coral_loss below)
        self.biases = nn.Parameter(torch.zeros(self.K))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, D)  fused representation
        Returns:
            cumprobs: (B, K)  cumulative probabilities P(y ≤ k) for k in {0,1,2}
        """
        z = self.dropout(z)
        logit_base = self.fc(z)                                # (B, 1)
        # P(y ≤ k) = σ(b_k − wᵀz)
        logits = self.biases.unsqueeze(0) - logit_base        # (B, K)
        cumprobs = torch.sigmoid(logits)                       # (B, K)
        return cumprobs

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        """
        Convert cumulative probabilities to ordinal class predictions.
        Predicted class = Σ_k 1[P(y>k) > 0.5]

        Returns: (B,) integer predictions in {0, 1, 2, 3}
        """
        cumprobs = self.forward(z)                             # (B, K)
        exceed_half = (cumprobs < 0.5).float()                 # P(y>k)>0.5 ↔ P(y≤k)<0.5
        return exceed_half.sum(dim=1).long()                   # (B,)

    def predict_with_confidence(
        self, z: torch.Tensor, mc_dropout_passes: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns predicted class and entropy-based confidence.

        Args:
            z:                  (B, D)
            mc_dropout_passes:  if > 0, use MC Dropout for uncertainty

        Returns:
            pred:  (B,) int predictions
            conf:  (B,) float confidence in [0, 1]
        """
        if mc_dropout_passes > 0:
            # Monte Carlo Dropout: M stochastic passes
            self.train()                                       # enable dropout
            mc_cumprobs = torch.stack([
                self.forward(z) for _ in range(mc_dropout_passes)
            ], dim=0)                                          # (M, B, K)
            self.eval()

            # Convert to class probabilities via MC average
            cumprobs_mean = mc_cumprobs.mean(dim=0)            # (B, K)
            class_probs = cumprobs_to_class_probs(cumprobs_mean, self.num_classes)

            # MC variance as additional uncertainty signal
            mc_var = mc_cumprobs.var(dim=0).mean(dim=-1)       # (B,)
            mc_conf = 1.0 / (1.0 + mc_var * 10)               # (B,)
        else:
            self.eval()
            cumprobs_mean = self.forward(z)
            class_probs = cumprobs_to_class_probs(cumprobs_mean, self.num_classes)
            mc_conf = None

        # Entropy-based confidence
        eps = 1e-8
        H = -(class_probs * (class_probs + eps).log2()).sum(dim=-1)  # (B,)
        H_max = torch.log2(torch.tensor(float(self.num_classes)))
        entropy_conf = 1.0 - H / H_max                        # (B,) in [0,1]

        # Combined confidence (Section 3.9.1 of paper)
        if mc_conf is not None:
            conf = 0.6 * entropy_conf + 0.4 * mc_conf
        else:
            conf = entropy_conf

        pred = class_probs.argmax(dim=-1)                      # (B,)
        return pred, conf


def cumprobs_to_class_probs(cumprobs: torch.Tensor,
                             num_classes: int) -> torch.Tensor:
    """
    Convert cumulative probabilities P(y≤k) to class probabilities P(y=k).

    Args:
        cumprobs: (B, K)  where K = num_classes - 1
    Returns:
        probs:    (B, num_classes)
    """
    B = cumprobs.shape[0]
    # P(y=0) = P(y≤0)
    # P(y=k) = P(y≤k) − P(y≤k-1)  for k = 1..K-1
    # P(y=K) = 1 − P(y≤K-1)
    zeros = torch.zeros(B, 1, device=cumprobs.device)
    ones = torch.ones(B, 1, device=cumprobs.device)
    cum_aug = torch.cat([zeros, cumprobs, ones], dim=1)        # (B, K+2)
    probs = cum_aug[:, 1:] - cum_aug[:, :-1]                   # (B, num_classes)
    probs = probs.clamp(min=1e-8)                              # numerical stability
    return probs


def coral_loss(cumprobs: torch.Tensor, targets: torch.Tensor,
               num_classes: int = 4) -> torch.Tensor:
    """
    CORAL loss function (Eq. 8 in paper).

    Args:
        cumprobs:    (B, K)  predicted cumulative probabilities P(y≤k)
        targets:     (B,)   integer ordinal labels in {0, ..., num_classes-1}
        num_classes: total number of classes (4)

    Returns:
        loss: scalar
    """
    K = num_classes - 1
    B = targets.shape[0]

    # Encode target as K binary tasks: label_k = 1 if y > k else 0
    # Equivalently: label_k = 1 if y >= k+1
    rank_matrix = torch.zeros(B, K, device=targets.device)    # (B, K)
    for k in range(K):
        rank_matrix[:, k] = (targets > k).float()

    # Binary cross-entropy for each cumulative classifier
    # P(y > k) = 1 - P(y ≤ k) = 1 - cumprobs[:, k]
    p_gt = 1.0 - cumprobs                                     # P(y>k), (B, K)

    eps = 1e-7
    loss_pos = rank_matrix * torch.log(p_gt + eps)
    loss_neg = (1 - rank_matrix) * torch.log(1 - p_gt + eps)
    loss = -(loss_pos + loss_neg)                              # (B, K)

    return loss.mean()
