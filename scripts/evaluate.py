"""
SSS-TT Evaluation Script.

Evaluates a trained SSS-TT checkpoint on the test set and runs
the full robustness protocol from Section 4.6 of the paper.

Usage:
    python scripts/evaluate.py \
        --data_dir /data/icope \
        --checkpoint checkpoints/sss-tt/fold1/best.pth \
        --output_dir results/ \
        --robustness_test
"""

import os
import sys
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.sss_tt import build_sss_tt
from src.data.icope_dataset import ICOPEDataset
from src.data.augmentations import RobustnessEvaluator
from src.evaluation.metrics import MetricTracker, print_metrics, compute_all_metrics
from src.evaluation.visualization import (
    plot_confusion_matrix, plot_robustness_curves, plot_crossval_results
)


def parse_args():
    p = argparse.ArgumentParser(description='SSS-TT Evaluation')
    p.add_argument('--data_dir', type=str, required=True)
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--output_dir', type=str, default='results/')
    p.add_argument('--fold', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--T', type=int, default=32)
    p.add_argument('--robustness_test', action='store_true')
    p.add_argument('--degradations', nargs='+',
                   default=['gaussian_noise', 'jpeg_compression',
                            'motion_blur', 'occlusion'])
    p.add_argument('--mc_passes', type=int, default=10,
                   help='MC Dropout passes for uncertainty')
    p.add_argument('--modalities', nargs='+',
                   default=['thermal', 'physiology'])
    return p.parse_args()


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    cfg = ckpt.get('config', {})
    model = build_sss_tt(cfg).to(device)
    state = ckpt.get('model', ckpt)
    if hasattr(model, 'module'):
        model.module.load_state_dict(state)
    else:
        model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Loaded checkpoint from: {checkpoint_path}")
    return model


@torch.no_grad()
def evaluate_clean(model, loader, device) -> dict:
    """Standard evaluation on clean (unaugmented) test set."""
    tracker = MetricTracker()
    for batch in loader:
        video = batch['video'].to(device)
        labels = batch['label'].to(device)
        mod_signals = {
            k: (v.to(device) if v is not None else None)
            for k, v in batch['modalities'].items()
        }
        out = model(video, mod_signals)
        tracker.update(labels, out['pred'], out['class_probs'])
    return tracker.compute()


@torch.no_grad()
def evaluate_with_degradation(
    model, dataset, device,
    degradation: str, levels: list,
    batch_size: int = 16,
) -> dict:
    """
    Evaluate model under each severity level of a degradation.
    Returns dict with 'levels', 'accuracy', 'qwk' lists.
    """
    rob_eval = RobustnessEvaluator()
    results = {'levels': levels, 'accuracy': [], 'qwk': []}

    for level in levels:
        all_true, all_pred = [], []
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=2)
        for batch in loader:
            video = batch['video'].to(device)
            labels = batch['label'].numpy().tolist()

            # Apply degradation frame by frame
            degraded = []
            for b in range(video.shape[0]):
                clip = rob_eval.apply_degradation(
                    video[b].cpu(), degradation, level
                )
                degraded.append(clip)
            video_deg = torch.stack(degraded, dim=0).to(device)

            out = model(video_deg)
            all_true.extend(labels)
            all_pred.extend(out['pred'].cpu().numpy().tolist())

        metrics = compute_all_metrics(
            np.array(all_true), np.array(all_pred)
        )
        results['accuracy'].append(metrics['accuracy'])
        results['qwk'].append(metrics['qwk'])
        print(f"  {degradation} @ {level}: "
              f"Acc={metrics['accuracy']:.1f}%, QWK={metrics['qwk']:.3f}")

    return results


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(args.checkpoint, device)

    # Build test dataset
    splits_dir = os.path.join(args.data_dir, 'splits')
    test_ds = ICOPEDataset(
        args.data_dir,
        f'{splits_dir}/fold{args.fold}_test.json',
        T=args.T, augment=False,
        modalities=args.modalities,
        mode='test',
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers
    )

    # ---- Clean evaluation ----
    print("\nEvaluating on clean test set...")
    clean_metrics = evaluate_clean(model, test_loader, device)
    print_metrics(clean_metrics, prefix='Clean Test')

    # Confusion matrix
    cm = clean_metrics.get('confusion_matrix')
    if cm is not None:
        fig = plot_confusion_matrix(cm, normalize=True)
        fig.savefig(
            os.path.join(args.output_dir, 'confusion_matrix.png'),
            dpi=150, bbox_inches='tight'
        )
        print("Saved: confusion_matrix.png")

    # Save clean metrics
    safe_metrics = {
        k: (v.tolist() if hasattr(v, 'tolist') else v)
        for k, v in clean_metrics.items()
        if k not in ('per_class', 'confusion_matrix')
    }
    with open(os.path.join(args.output_dir, 'clean_metrics.json'), 'w') as f:
        json.dump(safe_metrics, f, indent=2)

    # ---- Robustness evaluation ----
    if args.robustness_test:
        print(f"\nRunning robustness evaluation on {len(args.degradations)} degradations...")
        rob_evaluator = RobustnessEvaluator()
        robustness_results = {}

        for deg in args.degradations:
            if deg not in rob_evaluator.DEGRADATION_CONFIGS:
                print(f"  WARNING: Unknown degradation '{deg}', skipping.")
                continue
            cfg = rob_evaluator.DEGRADATION_CONFIGS[deg]
            levels = cfg['levels']
            print(f"\n  {deg}: levels={levels}")
            deg_results = evaluate_with_degradation(
                model, test_ds, device, deg, levels,
                batch_size=args.batch_size,
            )
            deg_results['x_label'] = cfg['param']
            robustness_results[deg] = deg_results

        # Plot robustness curves
        if robustness_results:
            fig = plot_robustness_curves(
                robustness_results,
                clinical_threshold=80.0,
                save_path=os.path.join(args.output_dir, 'robustness_curves.png'),
            )
            print("Saved: robustness_curves.png")

        # Save robustness results
        with open(os.path.join(args.output_dir, 'robustness_results.json'), 'w') as f:
            json.dump(robustness_results, f, indent=2)

    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
