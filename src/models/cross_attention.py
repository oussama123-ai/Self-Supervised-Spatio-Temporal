"""
Cross-Attention Fusion (CAF) module for SSS-TT.

Adaptively fuses RGB spatio-temporal features with optional
thermal and physiological modalities.

Key design choices (Section 3.7 of paper):
  - 4-head cross-attention: video queries multimodal embeddings
  - Learnable [MISSING] token replaces absent modalities
  - Residual connection: video representation always preserved
  - During training: modalities dropped randomly (p=0.3)
  - At inference: naturally assigns near-zero weights to [MISSING]
    tokens (α < 0.02, confident dismissal)

Equations from paper:
  α = softmax(W_Q z_video · [W_K m_enc]ᵀ / √192)   (Eq. 5)
  z_fused = z_video + Dropout(α W_V [m_enc])          (Eq. 6)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityEncoder(nn.Module):
    """
    Projects a raw modality signal to the shared 768-dim embedding space.

    Thermal: spatial feature map → flatten → MLP
    Physiology (HR, SpO2): time-series vector → MLP
    """

    def __init__(self, modality: str, embed_dim: int = 768):
        super().__init__()
        self.modality = modality

        if modality == 'thermal':
            # Thermal: expects (B, 1, H, W) or (B, thermal_feat_dim)
            # Use a small CNN backbone + linear projection
            self.encoder = nn.Sequential(
                nn.AdaptiveAvgPool2d((7, 7)),
                nn.Flatten(),
                nn.Linear(7 * 7, 256),
                nn.GELU(),
                nn.Linear(256, embed_dim),
            )
        elif modality in ('physiology', 'hr_spo2'):
            # HR + SpO2 time-series: (B, signal_len) or (B, 2, signal_len)
            self.encoder = nn.Sequential(
                nn.Linear(128, 256),            # 128-sample window by default
                nn.GELU(),
                nn.Linear(256, embed_dim),
            )
        else:
            raise ValueError(f"Unknown modality: {modality}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, embed_dim)."""
        if self.modality == 'thermal' and x.dim() == 4:
            # x: (B, C, H, W)
            return self.encoder[2:](
                self.encoder[0](x).flatten(1)
            )
        return self.encoder(x)


class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention Fusion with graceful degradation for missing modalities.

    Args:
        embed_dim:       Dimension of video and modality embeddings (768)
        num_heads:       Number of cross-attention heads (4)
        dropout:         Attention dropout rate (0.1)
        missing_prob:    Probability of randomly dropping a modality during training (0.3)
        modalities:      List of optional modality names ['thermal', 'physiology']
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 4,
        dropout: float = 0.1,
        missing_prob: float = 0.3,
        modalities: list[str] | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads                 # 192
        self.scale = self.head_dim ** -0.5
        self.missing_prob = missing_prob
        self.modalities = modalities or ['thermal', 'physiology']

        # Per-modality encoders
        self.mod_encoders = nn.ModuleDict({
            name: ModalityEncoder(name, embed_dim)
            for name in self.modalities
        })

        # Learnable [MISSING] token — one per modality
        self.missing_tokens = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1, 1, embed_dim))
            for name in self.modalities
        })
        for p in self.missing_tokens.values():
            nn.init.trunc_normal_(p, std=0.02)

        # Cross-attention projections
        self.W_Q = nn.Linear(embed_dim, embed_dim)
        self.W_K = nn.Linear(embed_dim, embed_dim)
        self.W_V = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(embed_dim)

    def _encode_modality(self, name: str,
                         signal: torch.Tensor | None,
                         B: int,
                         device: torch.device) -> torch.Tensor:
        """
        Encode one modality to (B, 1, embed_dim).
        Returns [MISSING] token if signal is None or randomly dropped.
        """
        use_missing = (signal is None)
        if not use_missing and self.training:
            # Stochastic dropout during training (p=0.3)
            use_missing = (torch.rand(1).item() < self.missing_prob)

        if use_missing:
            return self.missing_tokens[name].expand(B, 1, -1)  # (B,1,D)

        enc = self.mod_encoders[name](signal)                  # (B, D)
        return enc.unsqueeze(1)                                # (B, 1, D)

    def forward(
        self,
        z_video: torch.Tensor,
        modality_signals: dict[str, torch.Tensor | None] | None = None,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            z_video:          (B, embed_dim)  spatio-temporal video feature
            modality_signals: dict mapping modality name → raw signal tensor
                              or None if modality absent
            return_attn:      if True, also return attention weights

        Returns:
            z_fused:     (B, embed_dim)
            attn_weights: (B, heads, 1, J) or None
        """
        if modality_signals is None:
            modality_signals = {}

        B = z_video.shape[0]
        device = z_video.device

        # ---- Build key/value from all modalities ----
        mod_embeddings = []
        for name in self.modalities:
            sig = modality_signals.get(name, None)
            enc = self._encode_modality(name, sig, B, device)  # (B, 1, D)
            mod_embeddings.append(enc)

        # Keys/Values: (B, J, D)  where J = num modalities
        kv = torch.cat(mod_embeddings, dim=1)

        # ---- Cross-attention (Eq. 5-6) ----
        # Query: video feature (B, 1, D)
        query = z_video.unsqueeze(1)                           # (B, 1, D)
        Q = self.W_Q(query)                                    # (B, 1, D)
        K = self.W_K(kv)                                       # (B, J, D)
        V = self.W_V(kv)                                       # (B, J, D)

        # Multi-head reshape
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            B_, S, D = t.shape
            return t.reshape(B_, S, self.num_heads, self.head_dim).transpose(1, 2)

        Q = split_heads(Q)                                     # (B, h, 1, d)
        K = split_heads(K)                                     # (B, h, J, d)
        V = split_heads(V)                                     # (B, h, J, d)

        # Attention weights
        attn = (Q @ K.transpose(-2, -1)) * self.scale         # (B, h, 1, J)
        attn = F.softmax(attn, dim=-1)
        attn_weights = attn                                    # save for analysis

        attn = self.attn_dropout(attn)
        z_attn = (attn @ V)                                    # (B, h, 1, d)
        z_attn = z_attn.transpose(1, 2).reshape(B, 1, self.embed_dim)
        z_attn = self.out_proj(z_attn).squeeze(1)              # (B, D)

        # ---- Residual fusion (Eq. 6) ----
        z_fused = z_video + self.out_dropout(z_attn)
        z_fused = self.norm(z_fused)

        if return_attn:
            return z_fused, attn_weights
        return z_fused, None

    def get_modality_weights(
        self,
        z_video: torch.Tensor,
        modality_signals: dict[str, torch.Tensor | None] | None = None,
    ) -> dict[str, float]:
        """
        Utility: return per-modality attention weight (averaged over heads).
        Useful for clinical transparency and debugging.
        """
        self.eval()
        with torch.no_grad():
            _, attn_w = self.forward(
                z_video, modality_signals, return_attn=True
            )
        if attn_w is None:
            return {}
        # attn_w: (B, h, 1, J)
        avg_weights = attn_w.mean(dim=[0, 1, 2])               # (J,)
        return {
            name: avg_weights[i].item()
            for i, name in enumerate(self.modalities)
        }
