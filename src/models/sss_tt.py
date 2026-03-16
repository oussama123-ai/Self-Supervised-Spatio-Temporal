"""
SSS-TT: Full Self-Supervised Sequential Spatio-Temporal Transformer model.

Integrates:
  1. ViT-Base/16 encoder (MAE-pretrained)     — spatial features per frame
  2. TCN with dilated causal convolutions     — temporal pain dynamics
  3. Cross-Attention Fusion (CAF)            — adaptive multimodal integration
  4. CORAL ordinal regression head           — rank-consistent pain prediction

Pipeline (Section 3.3 of paper):
  Video (B, T, 3, 224, 224)
    → RetinaFace preprocessing (external, done in data pipeline)
    → ViTEncoderWithTemporalPE  → spatial_seq (B, T, 768)
    → TCN                       → temporal_repr (B, T, 768)
    → z_video = temporal_repr[:, -1, :]      (B, 768)
    → CrossAttentionFusion      → z_fused (B, 768)
    → CORALHead                 → cumprobs (B, 3) + pred (B,)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit_encoder import ViTEncoder, ViTEncoderWithTemporalPE
from .tcn import TCN
from .cross_attention import CrossAttentionFusion
from .coral_head import CORALHead, coral_loss, cumprobs_to_class_probs


class SSSTT(nn.Module):
    """
    SSS-TT: Self-Supervised Sequential Spatio-Temporal Transformer.

    Args:
        T:                Number of frames per clip (32)
        img_size:         Frame resolution (224)
        patch_size:       ViT patch size (16)
        embed_dim:        ViT embedding dimension (768)
        vit_depth:        ViT transformer layers (12)
        vit_heads:        ViT attention heads (12)
        tcn_layers:       TCN dilated layers (4)
        tcn_kernel:       TCN kernel size (3)
        caf_heads:        Cross-attention heads (4)
        num_pain_levels:  Ordinal pain levels (4: 0-3)
        dropout:          General dropout rate (0.1)
        missing_prob:     Modality dropout probability during training (0.3)
        modalities:       Optional modality names for CAF
        freeze_vit_epochs: Freeze ViT for first N epochs (10), then unfreeze
    """

    def __init__(
        self,
        T: int = 32,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        vit_depth: int = 12,
        vit_heads: int = 12,
        tcn_layers: int = 4,
        tcn_kernel: int = 3,
        caf_heads: int = 4,
        num_pain_levels: int = 4,
        dropout: float = 0.1,
        missing_prob: float = 0.3,
        modalities: list[str] | None = None,
    ):
        super().__init__()
        self.T = T
        self.embed_dim = embed_dim
        self.num_pain_levels = num_pain_levels
        self.modalities = modalities or ['thermal', 'physiology']
        self._current_epoch = 0

        # ---- Stage 3a: Spatial ViT encoder ----
        vit = ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=3,
            embed_dim=embed_dim,
            depth=vit_depth,
            num_heads=vit_heads,
            dropout=dropout,
            return_all_tokens=False,
        )
        self.vit_temporal = ViTEncoderWithTemporalPE(vit, T=T)

        # ---- Stage 3b: Temporal TCN ----
        self.tcn = TCN(
            in_channels=embed_dim,
            out_channels=embed_dim,
            num_layers=tcn_layers,
            kernel_size=tcn_kernel,
            dropout=dropout,
        )

        # ---- Stage 4: Cross-Attention Fusion ----
        self.caf = CrossAttentionFusion(
            embed_dim=embed_dim,
            num_heads=caf_heads,
            dropout=dropout,
            missing_prob=missing_prob,
            modalities=self.modalities,
        )

        # ---- Stage 5: Ordinal regression head ----
        self.coral_head = CORALHead(
            in_features=embed_dim,
            num_classes=num_pain_levels,
            dropout=dropout,
        )

    # ------------------------------------------------------------------
    # MAE weight loading
    # ------------------------------------------------------------------

    def load_mae_weights(self, mae_checkpoint: str,
                         strict: bool = False) -> None:
        """
        Load pretrained MAE encoder weights into self.vit_temporal.vit.

        Args:
            mae_checkpoint: Path to MAE .pth checkpoint
            strict:         Whether to require exact key match
        """
        ckpt = torch.load(mae_checkpoint, map_location='cpu')
        # Support both raw state_dict and {'model': state_dict} formats
        state = ckpt.get('model', ckpt.get('state_dict', ckpt))

        # Extract only encoder keys
        encoder_state = {
            k.replace('encoder.', ''): v
            for k, v in state.items()
            if k.startswith('encoder.')
        }
        missing, unexpected = self.vit_temporal.vit.load_state_dict(
            encoder_state, strict=strict
        )
        print(f"[MAE load] missing={len(missing)}, unexpected={len(unexpected)}")

    # ------------------------------------------------------------------
    # ViT freezing strategy (Section 3.5.4)
    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int, freeze_epochs: int = 10) -> None:
        """
        Freeze ViT for first `freeze_epochs`, then unfreeze with reduced lr.
        Call at the start of each epoch.
        """
        self._current_epoch = epoch
        frozen = (epoch < freeze_epochs)
        for p in self.vit_temporal.vit.parameters():
            p.requires_grad = not frozen
        if epoch == freeze_epochs:
            print(f"[Epoch {epoch}] ViT encoder unfrozen.")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        video: torch.Tensor,
        modality_signals: dict[str, torch.Tensor | None] | None = None,
        return_attn: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            video:            (B, T, 3, H, W) — preprocessed, normalised frames
            modality_signals: optional dict {'thermal': tensor|None,
                                              'physiology': tensor|None}
            return_attn:      whether to return attention maps

        Returns dict with keys:
            'cumprobs':      (B, K)   cumulative probabilities P(y≤k)
            'pred':          (B,)     integer pain level predictions
            'class_probs':   (B, 4)   class-level probabilities
            'attn_spatial':  attention from ViT last layer   [optional]
            'attn_caf':      attention from CAF               [optional]
        """
        B = video.shape[0]

        # ---- Spatial encoding: one ViT pass per frame ----
        spatial_seq = self.vit_temporal(video)                 # (B, T, 768)

        # ---- Temporal encoding: TCN over frame sequence ----
        temporal_out = self.tcn(spatial_seq)                   # (B, T, 768)
        z_video = temporal_out[:, -1, :]                       # (B, 768) — last=full context

        # ---- Cross-attention multimodal fusion ----
        z_fused, attn_caf = self.caf(
            z_video, modality_signals, return_attn=return_attn
        )                                                      # (B, 768)

        # ---- Ordinal regression ----
        cumprobs = self.coral_head(z_fused)                    # (B, K)
        pred = self.coral_head.predict(z_fused)                # (B,)
        class_probs = cumprobs_to_class_probs(
            cumprobs, self.num_pain_levels
        )                                                      # (B, 4)

        out = {
            'cumprobs': cumprobs,
            'pred': pred,
            'class_probs': class_probs,
            'z_fused': z_fused,
        }
        if return_attn:
            out['attn_caf'] = attn_caf
        return out

    def predict_with_confidence(
        self,
        video: torch.Tensor,
        modality_signals: dict | None = None,
        mc_passes: int = 10,
    ) -> dict[str, torch.Tensor]:
        """
        Inference with uncertainty estimation (Section 3.9.1).
        Returns pred, confidence, and clinical alert level.
        """
        out = self.forward(video, modality_signals)
        z_fused = out['z_fused']
        pred, conf = self.coral_head.predict_with_confidence(
            z_fused, mc_dropout_passes=mc_passes
        )
        # Clinical alert thresholds (Section 3.9.1)
        # High (≥0.8): auto-alert if pred ≥ 2
        # Medium (0.5-0.8): flag for review
        # Low (<0.5): require manual assessment
        alert = torch.zeros_like(pred, dtype=torch.long)
        alert = torch.where(conf >= 0.8,
                            torch.where(pred >= 2,
                                        torch.tensor(2),      # HIGH ALERT
                                        torch.tensor(0)),
                            alert)
        alert = torch.where((conf >= 0.5) & (conf < 0.8),
                             torch.tensor(1),                  # REVIEW
                             alert)
        return {
            'pred': pred,
            'confidence': conf,
            'alert_level': alert,          # 0=ok, 1=review, 2=high
            'class_probs': out['class_probs'],
        }


def build_sss_tt(config: dict | None = None) -> SSSTT:
    """
    Factory function building SSS-TT from a config dict.
    Default values match the paper's best configuration.
    """
    cfg = config or {}
    return SSSTT(
        T=cfg.get('T', 32),
        img_size=cfg.get('img_size', 224),
        patch_size=cfg.get('patch_size', 16),
        embed_dim=cfg.get('embed_dim', 768),
        vit_depth=cfg.get('vit_depth', 12),
        vit_heads=cfg.get('vit_heads', 12),
        tcn_layers=cfg.get('tcn_layers', 4),
        tcn_kernel=cfg.get('tcn_kernel', 3),
        caf_heads=cfg.get('caf_heads', 4),
        num_pain_levels=cfg.get('num_pain_levels', 4),
        dropout=cfg.get('dropout', 0.1),
        missing_prob=cfg.get('missing_prob', 0.3),
        modalities=cfg.get('modalities', ['thermal', 'physiology']),
    )
