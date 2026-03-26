"""
Vision Transformer (ViT) Encoder for SSS-TT.

Implements ViT-Base/16 with:
- 12 Transformer layers
- 768-dim embeddings
- 12 attention heads
- Pre-norm + residual connections (Xiong et al., 2020)
- Learnable 2D positional embeddings
- Temporal sinusoidal positional embeddings
- [CLS] token aggregation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PatchEmbedding(nn.Module):
    """
    Partition each frame into P=196 non-overlapping 16×16 patches,
    project each to D=768-dim embedding (Eq. 1 in paper).
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2          # 196

        # Linear projection: flatten patch then project
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            patches: (B, num_patches, embed_dim)
        """
        x = self.proj(x)                   # (B, embed_dim, H/P, W/P)
        x = rearrange(x, 'b d h w -> b (h w) d')
        return x


class PositionalEmbedding2D(nn.Module):
    """Learnable 2D spatial positional embeddings for patch grid."""

    def __init__(self, num_patches: int, embed_dim: int):
        super().__init__()
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embed


def sinusoidal_positional_encoding(seq_len: int, dim: int,
                                   device: torch.device) -> torch.Tensor:
    """
    Temporal sinusoidal positional encodings (Vaswani et al., 2017).
    Returns: (seq_len, dim)
    """
    position = torch.arange(seq_len, device=device).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device).float()
        * (-math.log(10000.0) / dim)
    )
    pe = torch.zeros(seq_len, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with 12 heads, 64 dim/head."""

    def __init__(self, embed_dim: int = 768, num_heads: int = 12,
                 attn_dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, N, D)
        Returns:
            out: (B, N, D)
            attn_weights: (B, heads, N, N)
        """
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                     # each (B, heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn_weights = attn
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        return out, attn_weights


class TransformerBlock(nn.Module):
    """
    Transformer block with pre-LayerNorm (Xiong et al., 2020):
      x = x + Attention(LN(x))
      x = x + FFN(LN(x))
    """

    def __init__(self, embed_dim: int = 768, num_heads: int = 12,
                 mlp_ratio: float = 4.0, dropout: float = 0.1,
                 attn_dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, attn_dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(x_norm)
        x = residual + self.dropout(attn_out)

        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x, attn_weights


class ViTEncoder(nn.Module):
    """
    ViT-Base/16 encoder.

    Args:
        img_size:     Input image size (224)
        patch_size:   Patch size (16)
        in_channels:  Input channels (3 for RGB)
        embed_dim:    Token dimension (768)
        depth:        Number of Transformer blocks (12)
        num_heads:    Attention heads (12)
        mlp_ratio:    FFN expansion ratio (4)
        dropout:      Dropout rate (0.1)
        return_all_tokens: If True, return all patch tokens; else return [CLS]
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        return_all_tokens: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.return_all_tokens = return_all_tokens

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size,
                                          in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # 2D spatial positional embedding (for patches) + cls slot
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.pos_dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio,
                             dropout, attn_dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor,
                return_attn: bool = False) -> tuple[torch.Tensor, ...]:
        """
        Args:
            x: (B, C, H, W)  — single frame or batch of frames
            return_attn: whether to return last-layer attention weights
        Returns:
            cls_out: (B, embed_dim)  — [CLS] token representation
            attn_weights: (B, heads, N+1, N+1)  [optional]
        """
        B = x.shape[0]

        # Patch embedding
        patches = self.patch_embed(x)                              # (B, N, D)

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)              # (B, 1, D)
        x = torch.cat([cls_tokens, patches], dim=1)                # (B, N+1, D)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        # Transformer blocks
        attn_weights_last = None
        for block in self.blocks:
            x, attn_w = block(x)
            attn_weights_last = attn_w

        x = self.norm(x)

        if self.return_all_tokens:
            out = x                                                 # (B, N+1, D)
        else:
            out = x[:, 0]                                          # (B, D) — CLS

        if return_attn:
            return out, attn_weights_last
        return out, None


class ViTEncoderWithTemporalPE(nn.Module):
    """
    Wraps ViTEncoder to process a sequence of T frames, adding
    temporal sinusoidal positional encodings to the CLS tokens
    before passing downstream to the TCN.
    """

    def __init__(self, vit: ViTEncoder, T: int = 32):
        super().__init__()
        self.vit = vit
        self.T = T

    def forward(self, video: torch.Tensor,
                return_attn: bool = False) -> torch.Tensor:
        """
        Args:
            video: (B, T, C, H, W)
        Returns:
            spatial_seq: (B, T, D)  — per-frame CLS representations
        """
        B, T, C, H, W = video.shape
        # Flatten batch and time for parallel processing
        frames = video.reshape(B * T, C, H, W)
        cls_out, _ = self.vit(frames, return_attn=False)           # (B*T, D)
        spatial_seq = cls_out.reshape(B, T, -1)                    # (B, T, D)

        # Add temporal sinusoidal PE
        D = spatial_seq.shape[-1]
        temp_pe = sinusoidal_positional_encoding(T, D, spatial_seq.device)
        spatial_seq = spatial_seq + temp_pe.unsqueeze(0)           # (B, T, D)

        return spatial_seq
