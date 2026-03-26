"""
Masked Autoencoder (MAE) for self-supervised pretraining on NICU video.

Architecture (He et al., 2022 — MAE):
  - Asymmetric encoder-decoder
  - Encoder: ViT-Base/16 (12 layers, 768-dim)  processes only visible 25% of patches
  - Decoder: lightweight (8 layers, 512-dim)   reconstructs masked patches
  - Masking ratio: 75%  (147 of 196 patches masked per frame)
  - Loss: per-pixel MSE on masked patches (Eq. 2 in paper)

Pretraining scope:
  - RGB frames only; no audio or thermal modalities
  - 5,000 unlabeled NICU clips × 30 FPS × 30 s = 150,000 frames
  - 800 epochs, AdamW lr=1.5e-4, weight_decay=0.05
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .vit_encoder import ViTEncoder, PatchEmbedding


class MAEDecoder(nn.Module):
    """
    Lightweight MAE decoder: 8 Transformer layers, 512-dim.
    Projects encoder tokens → decoder tokens → reconstructed pixels.
    """

    def __init__(self, encoder_dim: int = 768, decoder_dim: int = 512,
                 depth: int = 8, num_heads: int = 16,
                 patch_size: int = 16, in_channels: int = 3):
        super().__init__()
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)

        # Learnable mask token (replaces each masked patch)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, 196 + 1, decoder_dim)           # 196 patches + CLS
        )
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

        # Decoder Transformer blocks
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=num_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=0.0,
            activation='gelu',
            batch_first=True,
            norm_first=True,          # pre-norm
        )
        self.decoder_blocks = nn.TransformerEncoder(
            decoder_layer, num_layers=depth
        )
        self.decoder_norm = nn.LayerNorm(decoder_dim)

        # Pixel reconstruction head: decoder_dim → patch_size² × channels
        self.decoder_pred = nn.Linear(
            decoder_dim, patch_size * patch_size * in_channels
        )

    def forward(self, x: torch.Tensor,
                ids_restore: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:           (B, num_visible + 1, encoder_dim)  encoder output w/ CLS
            ids_restore: (B, 196)  permutation to unshuffle patches
        Returns:
            pred:        (B, 196, patch_size² * channels)  pixel predictions
        """
        # Embed to decoder dim
        x = self.decoder_embed(x)                              # (B, V+1, D_dec)

        # Append mask tokens
        B, V1, D = x.shape
        num_masked = ids_restore.shape[1] - (V1 - 1)           # total - visible
        mask_tokens = self.mask_token.repeat(B, num_masked, 1) # (B, M, D_dec)
        x_ = torch.cat([x[:, 1:], mask_tokens], dim=1)         # (B, 196, D_dec), no CLS

        # Unshuffle to original patch order
        x_ = torch.gather(
            x_, dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, D)
        )

        # Re-prepend CLS token
        x = torch.cat([x[:, :1], x_], dim=1)                  # (B, 197, D_dec)

        # Add positional embeddings
        x = x + self.decoder_pos_embed

        # Decode
        x = self.decoder_blocks(x)
        x = self.decoder_norm(x)

        # Predict pixels (remove CLS)
        pred = self.decoder_pred(x[:, 1:])                     # (B, 196, P²C)
        return pred


class MAE(nn.Module):
    """
    Masked Autoencoder model for SSS-TT pretraining.

    Usage:
        mae = MAE()
        loss, pred, mask = mae(img)   # img: (B, 3, 224, 224)

        # After pretraining:
        encoder = mae.encoder
        encoder weights → initialize ViT in SSS-TT
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        encoder_embed_dim: int = 768,
        encoder_depth: int = 12,
        encoder_num_heads: int = 12,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mask_ratio: float = 0.75,
        norm_pix_loss: bool = True,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.norm_pix_loss = norm_pix_loss

        num_patches = (img_size // patch_size) ** 2            # 196

        # ----- Encoder -----
        self.encoder = ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            dropout=0.0,
            return_all_tokens=True,
        )

        # ----- Decoder -----
        self.decoder = MAEDecoder(
            encoder_dim=encoder_embed_dim,
            decoder_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            patch_size=patch_size,
            in_channels=in_channels,
        )

    # ------------------------------------------------------------------
    # Masking utilities
    # ------------------------------------------------------------------

    def random_masking(self, x: torch.Tensor, mask_ratio: float):
        """
        Per-sample random masking.

        Args:
            x: (B, N, D)  patch embeddings
        Returns:
            x_masked:    (B, num_visible, D)
            mask:        (B, N)  bool — True = masked
            ids_restore: (B, N)  int  — permutation to unshuffle
        """
        B, N, D = x.shape
        num_keep = int(N * (1 - mask_ratio))                   # 49

        # Random noise for shuffle
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)              # ascending: first = small = kept
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep only first num_keep
        ids_keep = ids_shuffle[:, :num_keep]
        x_masked = torch.gather(
            x, dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )

        # Binary mask: 0=keep, 1=removed
        mask = torch.ones(B, N, device=x.device)
        mask[:, :num_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)    # (B, N) shuffled back
        mask = mask.bool()

        return x_masked, mask, ids_restore

    # ------------------------------------------------------------------
    # Pixel target extraction
    # ------------------------------------------------------------------

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Convert image to patch-level pixel targets.
        Args:
            imgs: (B, 3, H, W)
        Returns:
            patches: (B, N, patch_size² * 3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3]
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], 3, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p * p * 3)
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct image from patch predictions.
        Args:
            x: (B, N, patch_size² * 3)
        Returns:
            imgs: (B, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, p, p, 3)
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], 3, h * p, w * p)
        return imgs

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, imgs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Args:
            imgs: (B, 3, H, W)  normalized RGB frames
        Returns:
            loss: scalar MAE reconstruction loss (Eq. 2)
            pred: (B, N, patch_size² * 3)  reconstructed patches
            mask: (B, N)  bool — which patches were masked
        """
        # 1. Patch embed (call encoder's patch_embed directly)
        B = imgs.shape[0]
        x = self.encoder.patch_embed(imgs)                     # (B, 196, D_enc)

        # 2. Add positional embeddings (skip CLS for now)
        x = x + self.encoder.pos_embed[:, 1:, :]

        # 3. Random masking
        x_vis, mask, ids_restore = self.random_masking(x, self.mask_ratio)

        # 4. Prepend CLS token
        cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x_vis = torch.cat([cls_tokens, x_vis], dim=1)         # (B, 49+1, D_enc)

        # 5. Encode (only visible patches)
        x_vis = self.encoder.pos_dropout(x_vis)
        attn_w = None
        for block in self.encoder.blocks:
            x_vis, attn_w = block(x_vis)
        x_vis = self.encoder.norm(x_vis)                       # (B, 50, D_enc)

        # 6. Decode
        pred = self.decoder(x_vis, ids_restore)                # (B, 196, P²C)

        # 7. Compute loss (Eq. 2): MSE on masked patches only
        target = self.patchify(imgs)                           # (B, 196, P²C)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()

        loss = (pred - target) ** 2                            # (B, 196, P²C)
        loss = loss.mean(dim=-1)                               # (B, 196)
        loss = (loss * mask).sum() / mask.sum()                # mean on masked only

        return loss, pred, mask

    def get_encoder(self) -> ViTEncoder:
        """Return the pretrained encoder for downstream fine-tuning."""
        return self.encoder
