"""
Unit tests for SSS-TT model components.

Run with: pytest tests/test_model.py -v
"""

import pytest
import torch
import numpy as np


# ──────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────

@pytest.fixture
def device():
    return torch.device('cpu')


@pytest.fixture
def dummy_video(device):
    """(B=2, T=4, C=3, H=224, W=224) random video clip."""
    return torch.randn(2, 4, 3, 224, 224, device=device)


@pytest.fixture
def dummy_frame(device):
    """(B=2, C=3, H=224, W=224) single frame."""
    return torch.randn(2, 3, 224, 224, device=device)


@pytest.fixture
def dummy_labels(device):
    """(B=2,) ordinal labels."""
    return torch.tensor([0, 2], device=device)


# ──────────────────────────────────────────────────────
# ViT Encoder Tests
# ──────────────────────────────────────────────────────

class TestViTEncoder:
    def test_patch_embedding_shape(self, dummy_frame, device):
        from src.models.vit_encoder import PatchEmbedding
        pe = PatchEmbedding(224, 16, 3, 768)
        out = pe(dummy_frame)
        assert out.shape == (2, 196, 768), f"Expected (2,196,768), got {out.shape}"

    def test_vit_cls_output(self, dummy_frame, device):
        from src.models.vit_encoder import ViTEncoder
        # Use tiny config for speed
        vit = ViTEncoder(img_size=224, patch_size=16, embed_dim=128,
                         depth=2, num_heads=4)
        out, _ = vit(dummy_frame)
        assert out.shape == (2, 128), f"Expected (2,128), got {out.shape}"

    def test_vit_all_tokens(self, dummy_frame, device):
        from src.models.vit_encoder import ViTEncoder
        vit = ViTEncoder(img_size=224, patch_size=16, embed_dim=128,
                         depth=2, num_heads=4, return_all_tokens=True)
        out, _ = vit(dummy_frame)
        assert out.shape == (2, 197, 128)   # 196 patches + CLS

    def test_temporal_pe(self, dummy_video, device):
        from src.models.vit_encoder import ViTEncoder, ViTEncoderWithTemporalPE
        vit = ViTEncoder(img_size=224, patch_size=16, embed_dim=128,
                         depth=2, num_heads=4)
        vit_t = ViTEncoderWithTemporalPE(vit, T=4)
        out = vit_t(dummy_video)
        assert out.shape == (2, 4, 128)


# ──────────────────────────────────────────────────────
# TCN Tests
# ──────────────────────────────────────────────────────

class TestTCN:
    def test_output_shape(self, device):
        from src.models.tcn import TCN
        tcn = TCN(in_channels=128, out_channels=128, num_layers=4)
        x = torch.randn(2, 8, 128, device=device)
        out = tcn(x)
        assert out.shape == (2, 8, 128), f"Expected (2,8,128), got {out.shape}"

    def test_causal_property(self, device):
        """Output at t should not depend on future inputs."""
        from src.models.tcn import TCN
        tcn = TCN(in_channels=16, out_channels=16, num_layers=2)
        tcn.eval()
        T = 8
        x = torch.randn(1, T, 16, device=device)
        out_full = tcn(x)

        # Truncate to first T//2 frames, check those outputs are unchanged
        x_trunc = x[:, :T//2, :]
        out_trunc = tcn(x_trunc)
        # Causal: first T//2 outputs should be identical
        assert torch.allclose(out_full[:, :T//2, :], out_trunc, atol=1e-5), \
            "TCN output at t depends on future frames (causality violated)"

    def test_receptive_field(self):
        from src.models.tcn import TCN
        tcn = TCN(num_layers=4, kernel_size=3)
        assert tcn.receptive_field == 31, \
            f"Expected RF=31, got {tcn.receptive_field}"

    def test_clip_representation(self, device):
        from src.models.tcn import TCN
        tcn = TCN(in_channels=64, out_channels=64)
        x = torch.randn(2, 8, 64, device=device)
        clip = tcn.get_clip_representation(x)
        assert clip.shape == (2, 64)


# ──────────────────────────────────────────────────────
# CORAL Head Tests
# ──────────────────────────────────────────────────────

class TestCORALHead:
    def test_cumprob_shape(self, device):
        from src.models.coral_head import CORALHead
        head = CORALHead(in_features=128, num_classes=4)
        z = torch.randn(3, 128, device=device)
        cumprobs = head(z)
        assert cumprobs.shape == (3, 3), f"Expected (3,3), got {cumprobs.shape}"

    def test_cumprobs_monotone(self, device):
        """Cumulative probabilities must be non-decreasing (rank consistency)."""
        from src.models.coral_head import CORALHead
        head = CORALHead(in_features=64, num_classes=4)
        z = torch.randn(10, 64, device=device)
        cumprobs = head(z).detach()
        for k in range(cumprobs.shape[1] - 1):
            assert (cumprobs[:, k+1] >= cumprobs[:, k] - 1e-4).all(), \
                f"Cumprobs not monotone at k={k}"

    def test_pred_range(self, device):
        from src.models.coral_head import CORALHead
        head = CORALHead(in_features=64, num_classes=4)
        z = torch.randn(8, 64, device=device)
        pred = head.predict(z)
        assert pred.min() >= 0 and pred.max() <= 3

    def test_coral_loss_shape(self, device):
        from src.models.coral_head import coral_loss
        cumprobs = torch.sigmoid(torch.randn(4, 3, device=device))
        targets = torch.tensor([0, 1, 2, 3], device=device)
        loss = coral_loss(cumprobs, targets)
        assert loss.shape == ()  # scalar

    def test_cumprobs_to_class_probs(self, device):
        from src.models.coral_head import cumprobs_to_class_probs
        cumprobs = torch.tensor([[0.9, 0.6, 0.2]], device=device)
        probs = cumprobs_to_class_probs(cumprobs, 4)
        assert probs.shape == (1, 4)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(1, device=device), atol=1e-5)


# ──────────────────────────────────────────────────────
# Cross Attention Fusion Tests
# ──────────────────────────────────────────────────────

class TestCrossAttentionFusion:
    def test_with_all_modalities(self, device):
        from src.models.cross_attention import CrossAttentionFusion
        caf = CrossAttentionFusion(embed_dim=128, num_heads=4,
                                    modalities=['thermal', 'physiology'])
        z = torch.randn(2, 128, device=device)
        signals = {
            'thermal': torch.randn(2, 1, 14, 14, device=device),
            'physiology': torch.randn(2, 128, device=device),
        }
        out, _ = caf(z, signals)
        assert out.shape == (2, 128)

    def test_with_missing_modality(self, device):
        from src.models.cross_attention import CrossAttentionFusion
        caf = CrossAttentionFusion(embed_dim=128, num_heads=4,
                                    modalities=['thermal', 'physiology'])
        z = torch.randn(2, 128, device=device)
        out, _ = caf(z, {'thermal': None, 'physiology': None})
        assert out.shape == (2, 128)

    def test_residual_preserved(self, device):
        """With zero attention weights, output should equal input (residual)."""
        from src.models.cross_attention import CrossAttentionFusion
        caf = CrossAttentionFusion(embed_dim=64, num_heads=4,
                                    modalities=['thermal'])
        # Zero out projection weights to get ~zero attention contribution
        caf.eval()
        z = torch.ones(1, 64, device=device)
        out, _ = caf(z, {'thermal': None})
        # Output must be close to input (residual dominates when attn → 0)
        assert out.shape == (1, 64)


# ──────────────────────────────────────────────────────
# Full SSS-TT Model Tests
# ──────────────────────────────────────────────────────

class TestSSTT:
    @pytest.fixture
    def tiny_model(self, device):
        from src.models.sss_tt import SSSTT
        return SSSTT(
            T=4, img_size=224, patch_size=16,
            embed_dim=128, vit_depth=2, vit_heads=4,
            tcn_layers=2, caf_heads=4,
            modalities=['thermal'],
        ).to(device)

    def test_forward_rgb_only(self, tiny_model, dummy_video, device):
        dummy_vid = dummy_video[:, :4]                        # use 4 frames
        out = tiny_model(dummy_vid)
        assert 'pred' in out
        assert 'cumprobs' in out
        assert 'class_probs' in out
        assert out['pred'].shape == (2,)
        assert out['class_probs'].shape == (2, 4)

    def test_forward_with_modality(self, tiny_model, dummy_video, device):
        dummy_vid = dummy_video[:, :4]
        signals = {'thermal': torch.randn(2, 1, 14, 14, device=device)}
        out = tiny_model(dummy_vid, signals)
        assert out['pred'].shape == (2,)

    def test_forward_missing_modality(self, tiny_model, dummy_video, device):
        dummy_vid = dummy_video[:, :4]
        out = tiny_model(dummy_vid, {'thermal': None})
        assert out['pred'].shape == (2,)

    def test_pred_in_range(self, tiny_model, dummy_video, device):
        dummy_vid = dummy_video[:, :4]
        out = tiny_model(dummy_vid)
        assert out['pred'].min() >= 0
        assert out['pred'].max() <= 3

    def test_class_probs_sum_to_one(self, tiny_model, dummy_video, device):
        dummy_vid = dummy_video[:, :4]
        out = tiny_model(dummy_vid)
        sums = out['class_probs'].sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


# ──────────────────────────────────────────────────────
# MAE Tests
# ──────────────────────────────────────────────────────

class TestMAE:
    @pytest.fixture
    def tiny_mae(self, device):
        from src.models.mae import MAE
        return MAE(
            encoder_embed_dim=128, encoder_depth=2, encoder_num_heads=4,
            decoder_embed_dim=64, decoder_depth=2, decoder_num_heads=4,
            mask_ratio=0.75,
        ).to(device)

    def test_forward_loss_scalar(self, tiny_mae, dummy_frame, device):
        loss, pred, mask = tiny_mae(dummy_frame)
        assert loss.shape == ()           # scalar
        assert loss.item() > 0

    def test_pred_shape(self, tiny_mae, dummy_frame, device):
        _, pred, _ = tiny_mae(dummy_frame)
        assert pred.shape == (2, 196, 16 * 16 * 3)

    def test_mask_ratio(self, tiny_mae, dummy_frame, device):
        _, _, mask = tiny_mae(dummy_frame)
        actual_ratio = mask.float().mean().item()
        assert abs(actual_ratio - 0.75) < 0.05, \
            f"Expected ~0.75 mask ratio, got {actual_ratio:.3f}"

    def test_get_encoder_returns_vit(self, tiny_mae):
        from src.models.vit_encoder import ViTEncoder
        enc = tiny_mae.get_encoder()
        assert isinstance(enc, ViTEncoder)


# ──────────────────────────────────────────────────────
# Loss Tests
# ──────────────────────────────────────────────────────

class TestSSTTLoss:
    def test_total_loss_positive(self, device, dummy_labels):
        from src.training.losses import SSTTLoss
        criterion = SSTTLoss()
        cumprobs = torch.sigmoid(torch.randn(2, 3, device=device))
        losses = criterion(cumprobs, dummy_labels, epoch=5)
        assert losses['total'].item() > 0

    def test_mae_loss_inactive_after_epoch(self, device, dummy_labels):
        from src.training.losses import SSTTLoss
        criterion = SSTTLoss(mae_epochs=10)
        cumprobs = torch.sigmoid(torch.randn(2, 3, device=device))
        mae_loss = torch.tensor(0.5, device=device)
        losses_early = criterion(cumprobs, dummy_labels,
                                  mae_loss=mae_loss, epoch=5)
        losses_late = criterion(cumprobs, dummy_labels,
                                 mae_loss=mae_loss, epoch=15)
        # MAE should contribute more in early epochs
        assert losses_early['mae'].item() > 0
        assert losses_late['mae'].item() == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
