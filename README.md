# SSS-TT: Self-Supervised Sequential Spatio-Temporal Transformers

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.13](https://img.shields.io/badge/PyTorch-1.13-red.svg)](https://pytorch.org/)

> **SSS-TT: Self-Supervised Sequential Spatio-Temporal Transformers with Adaptive Multimodal Fusion for Automated Neonatal Pain Assessment**  
> Oussama El Othmani, Riadh Ouersighni  
> Military Research Center, Tunisia | Polytechnic School of Tunisia, University of Carthage

---

## 🏆 Key Results

| Method | Accuracy | QWK | MAE | AUC |
|--------|----------|-----|-----|-----|
| CNN+LSTM | 74.8% | 0.65 | 0.38 | 0.78 |
| ViViT | 77.9% | 0.68 | 0.34 | 0.80 |
| TimeSformer | 78.5% | 0.70 | 0.32 | 0.82 |
| ViViT+MAE+CORAL† | 80.4% | 0.76 | 0.29 | 0.83 |
| **SSS-TT (Ours)** | **84.6%** | **0.82** | **0.23** | **0.87** |

*†: fair comparison with same NICU MAE pretraining + CORAL loss*

---

## 📐 Architecture

```
Input Video (T×H×W×3)
        │
        ▼
┌──────────────────┐
│  RetinaFace      │  Face detection & alignment (224×224)
│  Preprocessing   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  MAE Pretraining │  75% masking, self-supervised on unlabeled NICU video
│  (ViT-Base)      │  No pain labels required
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  ViT Encoder     │  12 layers, 768-dim, 12 attention heads
│  (Spatial)       │  Per-frame spatial feature extraction
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  TCN Module      │  Dilated convolutions (rates 1,2,4,8)
│  (Temporal)      │  31-frame receptive field, causal padding
└────────┬─────────┘
         │         ┌─────────────────────┐
         │◄────────│ Cross-Attention      │◄── Thermal + Physiology
         │         │ Fusion (CAF)         │    (optional modalities)
         │         └─────────────────────┘
         ▼
┌──────────────────┐
│  CORAL Ordinal   │  Rank-consistent cumulative probabilities
│  Regression Head │  Pain levels 0-3 aligned with NIPS
└────────┬─────────┘
         │
         ▼
  Pain Score (0-3) + Confidence
```

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/oussama123-ai/Self-Supervised-Spatio-Temporal.git
cd Self-Supervised-Spatio-Temporal

# Using Docker (recommended)
docker build -t sss-tt .
docker run --gpus all -v /path/to/data:/data sss-tt

# Or manual install
pip install -r requirements.txt
```

### MAE Pretraining

```bash
python scripts/pretrain_mae.py \
    --data_dir /data/unlabeled_nicu \
    --output_dir checkpoints/mae \
    --epochs 800 \
    --batch_size 16 \
    --mask_ratio 0.75 \
    --gpus 8
```

### Supervised Fine-tuning

```bash
python scripts/train.py \
    --data_dir /data/icope \
    --mae_checkpoint checkpoints/mae/best.pth \
    --output_dir checkpoints/sss-tt \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-4
```

### Evaluation

```bash
python scripts/evaluate.py \
    --data_dir /data/icope \
    --checkpoint checkpoints/sss-tt/best.pth \
    --output_dir results/
```

### Inference on Custom Video

```bash
python scripts/inference.py \
    --video_path /path/to/infant_video.mp4 \
    --checkpoint checkpoints/sss-tt/best.pth \
    --output_path results/predictions.json
```

---

## 📁 Repository Structure

```
SSS-TT/
├── src/
│   ├── models/
│   │   ├── mae.py              # Masked Autoencoder (pretraining)
│   │   ├── vit_encoder.py      # Vision Transformer encoder
│   │   ├── tcn.py              # Temporal Convolutional Network
│   │   ├── cross_attention.py  # Cross-Attention Fusion module
│   │   ├── coral_head.py       # CORAL ordinal regression head
│   │   └── sss_tt.py           # Full SSS-TT model
│   ├── data/
│   │   ├── icope_dataset.py    # iCOPE dataset loader
│   │   ├── preprocessing.py    # RetinaFace preprocessing
│   │   └── augmentations.py    # Clinical degradation augmentations
│   ├── training/
│   │   ├── trainer.py          # MAE pretraining loop
│   │   ├── finetune.py         # Supervised fine-tuning
│   │   └── losses.py           # CORAL + MAE + temporal losses
│   ├── evaluation/
│   │   ├── metrics.py          # QWK, accuracy, AUC, MAE
│   │   ├── robustness.py       # Clinical degradation evaluation
│   │   └── visualization.py    # Attention maps, confusion matrix
│   └── utils/
│       ├── config.py           # Hyperparameter configuration
│       ├── checkpoint.py       # Save/load utilities
│       └── uncertainty.py      # MC Dropout + entropy confidence
├── configs/
│   ├── pretrain_config.yaml
│   └── finetune_config.yaml
├── scripts/
│   ├── pretrain_mae.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── notebooks/
│   └── demo_inference.ipynb
├── tests/
│   └── test_model.py
├── docker/
│   └── Dockerfile
├── requirements.txt
└── README.md
```

---

## 📊 Reproducing Paper Results

### 5-Fold Cross-Validation

```bash
python scripts/train.py --cross_val --n_folds 5 --subject_level_split
```

### Ablation Studies

```bash
# Without MAE pretraining
python scripts/train.py --no_mae_pretrain

# Without TCN (ViT only)
python scripts/train.py --no_tcn

# Without Cross-Attention Fusion
python scripts/train.py --no_caf

# Without CORAL (standard softmax)
python scripts/train.py --loss_fn crossentropy
```

### Robustness Testing

```bash
python scripts/evaluate.py \
    --robustness_test \
    --degradations gaussian_noise jpeg_compression motion_blur occlusion \
    --checkpoint checkpoints/sss-tt/best.pth
```

---

## 🏥 Clinical Deployment

The model supports edge deployment on NVIDIA Jetson AGX Xavier (TensorRT INT8):

```bash
# Export to TensorRT
python scripts/export_tensorrt.py \
    --checkpoint checkpoints/sss-tt/best.pth \
    --output sss_tt_int8.trt

# Run edge inference
python scripts/edge_inference.py \
    --model sss_tt_int8.trt \
    --camera_stream rtsp://nicu-camera:554/stream
```

**Hardware Performance:**

| Hardware | Latency | FPS | Use Case |
|----------|---------|-----|----------|
| NVIDIA A100 | 0.15s | 6.7 | Cloud/datacenter |
| Jetson AGX Xavier (TensorRT INT8) | 0.08s | 12.5 | Edge NICU |
| RTX 3060 | 0.22s | 4.5 | Lab/pilot |
| CPU (Intel Xeon) | 1.8s | 0.56 | Periodic only |

---

## 📄 Citation

```bibtex
@article{elothmani2025ssstt,
  title={SSS-TT: Self-Supervised Sequential Spatio-Temporal Transformers 
         with Adaptive Multimodal Fusion for Automated Neonatal Pain Assessment},
  author={El Othmani, Oussama and Ouersighni, Riadh},
  journal={arXiv preprint},
  year={2025}
}
```

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 🤝 Acknowledgments

Supported by the DARPA GARD Program and the Amazon AI2AI fellowship.  
We thank the neonatal care teams and parents who consented to participate.

## ⚕️ Ethics & Privacy

- On-device processing with AES-256 encryption
- HIPAA/GDPR compliant
- All alerts require nurse acknowledgment (human oversight maintained)
- IRB-approved data collection protocols
