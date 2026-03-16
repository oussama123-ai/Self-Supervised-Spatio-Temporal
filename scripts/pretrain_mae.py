"""
MAE Pretraining Script for SSS-TT.

Usage:
    python scripts/pretrain_mae.py \
        --data_dir /data/unlabeled_nicu \
        --output_dir checkpoints/mae \
        --epochs 800 \
        --batch_size 16 \
        --mask_ratio 0.75 \
        --gpus 8
"""

import os
import sys
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.mae import MAE
from src.data.icope_dataset import UnlabeledNICUDataset


def parse_args():
    p = argparse.ArgumentParser(description='MAE Pretraining')
    p.add_argument('--data_dir', type=str, required=True)
    p.add_argument('--output_dir', type=str, default='checkpoints/mae')
    p.add_argument('--epochs', type=int, default=800)
    p.add_argument('--warmup_epochs', type=int, default=40)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1.5e-4)
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--mask_ratio', type=float, default=0.75)
    p.add_argument('--gpus', type=int, default=1)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--T', type=int, default=32,
                   help='Frames per clip (pretraining uses single frames)')
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--save_every', type=int, default=50)
    return p.parse_args()


def get_lr(optimizer) -> float:
    return optimizer.param_groups[0]['lr']


def cosine_warmup_lr(optimizer, epoch: int, warmup_epochs: int,
                      base_lr: float, min_lr: float = 1e-6,
                      total_epochs: int = 800):
    """Linear warmup + cosine decay (AdamW paper schedule)."""
    if epoch < warmup_epochs:
        lr = base_lr * epoch / max(1, warmup_epochs)
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + torch.cos(
            torch.tensor(torch.pi * progress)
        ).item())
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr


def pretrain(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Dataset (unlabeled NICU frames from Hospital Sites B+C)
    dataset = UnlabeledNICUDataset(args.data_dir, T=args.T)
    loader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True
    )
    print(f"Training samples: {len(dataset)}")

    # Model
    model = MAE(mask_ratio=args.mask_ratio).to(device)
    if args.gpus > 1 and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer (AdamW, Section 3.5.3)
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    start_epoch = 0
    best_loss = float('inf')

    # Resume
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt.get('loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}")

    log_path = os.path.join(args.output_dir, 'pretrain_log.csv')
    with open(log_path, 'w') as f:
        f.write('epoch,loss,lr,time_s\n')

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        lr = cosine_warmup_lr(
            optimizer, epoch, args.warmup_epochs,
            args.lr, total_epochs=args.epochs
        )

        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, video in enumerate(loader):
            # Sample random frame from each clip for MAE pretraining
            # (MAE operates on single frames, not full sequences)
            B, T, C, H, W = video.shape
            t_idx = torch.randint(0, T, (B,))
            frames = video[torch.arange(B), t_idx].to(device)  # (B, 3, 224, 224)

            optimizer.zero_grad()
            loss, _, _ = model(frames)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] "
                      f"Step [{batch_idx}/{len(loader)}] "
                      f"Loss: {loss.item():.4f}  LR: {lr:.2e}")

        avg_loss = epoch_loss / len(loader)
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1} done | Loss: {avg_loss:.4f} | "
              f"LR: {lr:.2e} | Time: {elapsed:.1f}s")

        with open(log_path, 'a') as f:
            f.write(f'{epoch+1},{avg_loss:.6f},{lr:.8f},{elapsed:.1f}\n')

        # Save checkpoint
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss

        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': avg_loss,
        }
        torch.save(state, os.path.join(args.output_dir, 'last.pth'))
        if is_best:
            torch.save(state, os.path.join(args.output_dir, 'best.pth'))
        if (epoch + 1) % args.save_every == 0:
            torch.save(state, os.path.join(
                args.output_dir, f'epoch_{epoch+1}.pth'
            ))

    print(f"\nPretraining complete! Best loss: {best_loss:.4f}")
    print(f"Checkpoint saved to: {args.output_dir}/best.pth")


if __name__ == '__main__':
    args = parse_args()
    pretrain(args)
