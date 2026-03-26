"""Checkpoint saving and loading utilities for SSS-TT."""

import os
import torch
import torch.nn as nn


class CheckpointManager:
    """
    Manages saving the best and latest checkpoints during training.

    Args:
        output_dir:  Directory to save checkpoints
        metric:      Metric name to optimise (e.g., 'qwk', 'accuracy')
        mode:        'max' (higher is better) or 'min'
    """

    def __init__(self, output_dir: str, metric: str = 'qwk',
                 mode: str = 'max'):
        self.output_dir = output_dir
        self.metric = metric
        self.mode = mode
        os.makedirs(output_dir, exist_ok=True)
        self.best_value = float('-inf') if mode == 'max' else float('inf')
        self.best_path = os.path.join(output_dir, 'best.pth')
        self.last_path = os.path.join(output_dir, 'last.pth')

    def _is_better(self, value: float) -> bool:
        if self.mode == 'max':
            return value > self.best_value
        return value < self.best_value

    def save(self, model, optimizer, epoch: int,
             metric_value: float, config: dict | None = None) -> bool:
        """Save checkpoint. Returns True if best was updated."""
        raw_model = model.module if hasattr(model, 'module') else model
        state = {
            'epoch': epoch,
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            self.metric: metric_value,
            'config': config or {},
        }
        torch.save(state, self.last_path)

        if self._is_better(metric_value):
            self.best_value = metric_value
            torch.save(state, self.best_path)
            return True
        return False

    def load_best(self, model) -> dict:
        """Load best checkpoint into model."""
        ckpt = torch.load(self.best_path, map_location='cpu')
        raw_model = model.module if hasattr(model, 'module') else model
        raw_model.load_state_dict(ckpt['model'], strict=False)
        print(f"Loaded best checkpoint "
              f"(epoch {ckpt['epoch']}, "
              f"{self.metric}={ckpt.get(self.metric, '?'):.4f})")
        return ckpt
