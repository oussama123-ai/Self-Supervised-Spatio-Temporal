"""
SSS-TT Supervised Fine-tuning Script.

Training strategy (Section 3.9):
  - AdamW, cosine annealing, lr_max=1e-4, lr_min=1e-6
  - 10-epoch warmup (linear ramp)
  - Batch size 32 (2 GPUs), gradient clipping norm=1.0
  - Freeze ViT encoder for first 10 epochs, then unfreeze (lr=1e-5)
  - Total loss = CORAL + 0.1*temporal + 0.5*MAE (first 10 epochs)
  - 5-fold cross-validation with strict subject-level splitting

Usage:
    python scripts/train.py \
        --data_dir /data/icope \
        --mae_checkpoint checkpoints/mae/best.pth \
        --output_dir checkpoints/sss-tt \
        --epochs 50 \
        --fold 1
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.sss_tt import build_sss_tt
from src.data.icope_dataset import get_dataloaders
from src.training.losses import SSTTLoss
from src.evaluation.metrics import MetricTracker, print_metrics
from src.utils.checkpoint import CheckpointManager


def parse_args():
    p = argparse.ArgumentParser(description='SSS-TT Fine-tuning')
    p.add_argument('--data_dir', type=str, required=True)
    p.add_argument('--mae_checkpoint', type=str, default=None)
    p.add_argument('--output_dir', type=str, default='checkpoints/sss-tt')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--warmup_epochs', type=int, default=10)
    p.add_argument('--freeze_vit_epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--lr_min', type=float, default=1e-6)
    p.add_argument('--lr_vit_unfreeze', type=float, default=1e-5)
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--label_smoothing', type=float, default=0.1)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--fold', type=int, default=1)
    p.add_argument('--cross_val', action='store_true')
    p.add_argument('--n_folds', type=int, default=5)
    p.add_argument('--T', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--no_mae_pretrain', action='store_true')
    p.add_argument('--no_tcn', action='store_true')
    p.add_argument('--no_caf', action='store_true')
    p.add_argument('--loss_fn', type=str, default='coral',
                   choices=['coral', 'crossentropy', 'mse', 'emd'])
    p.add_argument('--modalities', nargs='+',
                   default=['thermal', 'physiology'])
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_lr_schedule(optimizer, epoch: int, warmup: int,
                    total: int, lr_max: float, lr_min: float) -> float:
    """Linear warmup + cosine annealing."""
    import math
    if epoch < warmup:
        lr = lr_max * (epoch + 1) / warmup
    else:
        progress = (epoch - warmup) / max(1, total - warmup)
        lr = lr_min + (lr_max - lr_min) * 0.5 * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr


def train_one_fold(args, fold: int) -> dict:
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = os.path.join(args.output_dir, f'fold{fold}')
    os.makedirs(out_dir, exist_ok=True)

    # Dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_dir, fold=fold,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        T=args.T,
        modalities=args.modalities,
    )

    # Model
    cfg = {
        'T': args.T,
        'dropout': args.dropout,
        'modalities': args.modalities,
    }
    model = build_sss_tt(cfg).to(device)

    # Load MAE weights
    if not args.no_mae_pretrain and args.mae_checkpoint:
        model.load_mae_weights(args.mae_checkpoint)
        print(f"Loaded MAE weights from {args.mae_checkpoint}")

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Loss
    criterion = SSTTLoss(
        lambda_temp=0.1, lambda_mae=0.5,
        mae_epochs=args.warmup_epochs
    )

    # Optimizer: separate param groups for ViT (lower lr after unfreeze)
    vit_params = list(model.vit_temporal.parameters())
    other_params = [p for p in model.parameters()
                    if not any(p is vp for vp in vit_params)]
    optimizer = AdamW([
        {'params': other_params, 'lr': args.lr},
        {'params': vit_params, 'lr': args.lr_vit_unfreeze},
    ], weight_decay=args.weight_decay)

    ckpt_mgr = CheckpointManager(out_dir, metric='qwk', mode='max')
    best_metrics = {}
    log_rows = []

    for epoch in range(args.epochs):
        # ViT freezing strategy
        raw_model = model.module if hasattr(model, 'module') else model
        raw_model.set_epoch(epoch, freeze_epochs=args.freeze_vit_epochs)

        # LR schedule
        lr = get_lr_schedule(
            optimizer, epoch,
            warmup=args.warmup_epochs,
            total=args.epochs,
            lr_max=args.lr,
            lr_min=args.lr_min,
        )

        # ---- TRAIN ----
        model.train()
        train_tracker = MetricTracker()
        t0 = time.time()

        for batch in train_loader:
            video = batch['video'].to(device)
            labels = batch['label'].to(device)
            mod_signals = {
                k: (v.to(device) if v is not None else None)
                for k, v in batch['modalities'].items()
            }

            optimizer.zero_grad()
            out = model(video, mod_signals)
            losses = criterion(
                out['cumprobs'], labels,
                temporal_features=None,
                epoch=epoch,
            )
            losses['total'].backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            train_tracker.update(
                labels, out['pred'], out['class_probs'],
                loss=losses['total'].item()
            )

        train_metrics = train_tracker.compute()
        elapsed = time.time() - t0

        # ---- VALIDATE ----
        model.eval()
        val_tracker = MetricTracker()
        with torch.no_grad():
            for batch in val_loader:
                video = batch['video'].to(device)
                labels = batch['label'].to(device)
                mod_signals = {
                    k: (v.to(device) if v is not None else None)
                    for k, v in batch['modalities'].items()
                }
                out = model(video, mod_signals)
                losses = criterion(out['cumprobs'], labels, epoch=epoch)
                val_tracker.update(
                    labels, out['pred'], out['class_probs'],
                    loss=losses['total'].item()
                )

        val_metrics = val_tracker.compute()

        print(
            f"[Fold {fold} | Epoch {epoch+1}/{args.epochs}] "
            f"Train Acc={train_metrics['accuracy']:.1f}% "
            f"QWK={train_metrics['qwk']:.4f} | "
            f"Val Acc={val_metrics['accuracy']:.1f}% "
            f"QWK={val_metrics['qwk']:.4f} | "
            f"LR={lr:.2e} | {elapsed:.1f}s"
        )

        log_rows.append({
            'epoch': epoch + 1,
            'train_acc': train_metrics['accuracy'],
            'train_qwk': train_metrics['qwk'],
            'val_acc': val_metrics['accuracy'],
            'val_qwk': val_metrics['qwk'],
            'lr': lr,
        })

        is_best = ckpt_mgr.save(model, optimizer, epoch,
                                  val_metrics['qwk'])
        if is_best:
            best_metrics = val_metrics
            print(f"  ★ New best QWK = {val_metrics['qwk']:.4f}")

    # ---- TEST ----
    ckpt_mgr.load_best(model)
    model.eval()
    test_tracker = MetricTracker()
    with torch.no_grad():
        for batch in test_loader:
            video = batch['video'].to(device)
            labels = batch['label'].to(device)
            mod_signals = {
                k: (v.to(device) if v is not None else None)
                for k, v in batch['modalities'].items()
            }
            out = model(video, mod_signals)
            test_tracker.update(labels, out['pred'], out['class_probs'])

    test_metrics = test_tracker.compute()
    print_metrics(test_metrics, prefix=f'Fold {fold} Test')

    # Save results
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump({
            'fold': fold,
            'val_metrics': {k: v for k, v in best_metrics.items()
                            if not hasattr(v, '__len__') or isinstance(v, float)},
            'test_metrics': {k: float(v) for k, v in test_metrics.items()
                             if isinstance(v, (int, float))},
            'log': log_rows,
        }, f, indent=2)

    return test_metrics


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.cross_val:
        fold_results = []
        for fold in range(1, args.n_folds + 1):
            print(f"\n{'='*60}")
            print(f"  FOLD {fold}/{args.n_folds}")
            print('='*60)
            metrics = train_one_fold(args, fold)
            fold_results.append(metrics)

        # Aggregate
        accs = [m['accuracy'] for m in fold_results]
        qwks = [m['qwk'] for m in fold_results]
        import numpy as np
        print(f"\n{'='*60}")
        print(f"CROSS-VALIDATION SUMMARY ({args.n_folds} folds)")
        print(f"Accuracy: {np.mean(accs):.2f}% ± {np.std(accs):.2f}%")
        print(f"QWK:      {np.mean(qwks):.4f} ± {np.std(qwks):.4f}")
        print('='*60)
    else:
        train_one_fold(args, args.fold)


if __name__ == '__main__':
    main()
