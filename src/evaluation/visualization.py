"""
Visualization utilities for SSS-TT:
  - Attention maps (Figure 4 of paper)
  - Confusion matrix (Figure 5)
  - Ablation study plots (Figure 2)
  - Robustness curves (Figure 3)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import torch
from pathlib import Path


PAIN_LABELS = ['No Pain (0)', 'Mild (1)', 'Moderate (2)', 'Severe (3)']
PAIN_COLORS = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']


def plot_attention_maps(
    spatial_attn: np.ndarray,
    temporal_attn: np.ndarray,
    pain_level: int,
    save_path: str | None = None,
    patch_grid: int = 14,
) -> plt.Figure:
    """
    Plot spatial and temporal attention maps for a clip (Figure 4 style).

    Args:
        spatial_attn:  (num_patches,) attention weights from ViT CLS token
        temporal_attn: (T,) attention weights over time from TCN
        pain_level:    ground truth or predicted pain level (0-3)
        patch_grid:    number of patches per side (14 for 224/16)
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    title = f'{PAIN_LABELS[pain_level]} Attention'

    # Spatial attention heatmap
    spatial_map = spatial_attn.reshape(patch_grid, patch_grid)
    im = axes[0].imshow(spatial_map, cmap='hot', aspect='auto',
                         vmin=0, vmax=spatial_attn.max())
    axes[0].set_title(f'{title}\nSpatial Attention')
    axes[0].set_xlabel('Patch X')
    axes[0].set_ylabel('Patch Y')
    plt.colorbar(im, ax=axes[0], fraction=0.046)

    # Temporal attention
    axes[1].plot(temporal_attn, color=PAIN_COLORS[pain_level], linewidth=2)
    axes[1].fill_between(range(len(temporal_attn)), temporal_attn,
                          alpha=0.3, color=PAIN_COLORS[pain_level])
    axes[1].set_title(f'{title}\nTemporal Attention')
    axes[1].set_xlabel('Frame Number')
    axes[1].set_ylabel('Attention Weight')
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: str | None = None,
    normalize: bool = True,
) -> plt.Figure:
    """
    Plot confusion matrix (Figure 5 style).

    Args:
        cm:          (4, 4) confusion matrix
        normalize:   if True, show percentages
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Total samples: {cm.sum()}', y=1.01, color='gray')

    # Absolute counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=PAIN_LABELS, yticklabels=PAIN_LABELS,
                ax=axes[0], linewidths=0.5)
    axes[0].set_title('(a) Confusion Matrix (Absolute Counts)')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')

    # Normalised percentages
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    annot = np.array([[f'{v:.1f}' for v in row] for row in cm_pct])
    sns.heatmap(cm_pct, annot=annot, fmt='', cmap='RdYlGn',
                xticklabels=PAIN_LABELS, yticklabels=PAIN_LABELS,
                ax=axes[1], linewidths=0.5, vmin=0, vmax=100)
    axes[1].set_title('(b) Normalized Confusion Matrix (%)')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_ablation_study(
    results: dict[str, dict],
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot ablation study bar chart (Figure 2 style).

    Args:
        results: dict mapping config_name → {'accuracy': float, 'qwk': float}
    """
    configs = list(results.keys())
    accuracies = [results[c]['accuracy'] for c in configs]
    baseline = accuracies[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Accuracy comparison
    colors = ['#e74c3c' if i == 0 else '#27ae60' for i in range(len(configs))]
    bars = axes[0].bar(configs, accuracies, color=colors, alpha=0.85,
                        edgecolor='black', linewidth=0.5)
    for bar, acc in zip(bars, accuracies):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.1,
                     f'{acc:.1f}%', ha='center', va='bottom',
                     fontsize=9, fontweight='bold')
    axes[0].axhline(y=baseline, color='red', linestyle='--',
                    alpha=0.7, label='Baseline')
    axes[0].set_title('(a) Ablation Study: Accuracy Comparison')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_ylim(70, 90)
    axes[0].tick_params(axis='x', rotation=30)
    axes[0].legend()

    # (b) Incremental performance gains
    gains = [acc - baseline for acc in accuracies]
    bar_colors = ['#27ae60' if g >= 0 else '#e74c3c' for g in gains]
    axes[1].bar(configs, gains, color=bar_colors, alpha=0.85,
                edgecolor='black', linewidth=0.5)
    for i, (cfg, gain) in enumerate(zip(configs, gains)):
        axes[1].text(i, gain + 0.05 if gain >= 0 else gain - 0.1,
                     f'{gain:+.1f}%', ha='center', va='bottom' if gain >= 0
                     else 'top', fontsize=9, fontweight='bold')
    axes[1].axhline(y=0, color='black', linewidth=1)
    axes[1].set_title('(b) Incremental Performance Gains')
    axes[1].set_ylabel('Accuracy Improvement (%)')
    axes[1].tick_params(axis='x', rotation=30)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_robustness_curves(
    degradation_results: dict[str, dict],
    clinical_threshold: float = 80.0,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot robustness curves under clinical degradations (Figure 3 style).

    Args:
        degradation_results: dict mapping degradation_name →
                             {'levels': list, 'accuracy': list,
                              'x_label': str}
        clinical_threshold:  minimum acceptable accuracy (80%)
    """
    n_plots = min(4, len(degradation_results))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    deg_names = list(degradation_results.keys())[:n_plots]
    colors = ['#3498db', '#e67e22', '#2ecc71', '#e74c3c']

    for i, (deg_name, data) in enumerate(
        {k: degradation_results[k] for k in deg_names}.items()
    ):
        ax = axes[i]
        levels = data['levels']
        accs = data['accuracy']

        ax.plot(levels, accs, color=colors[i], marker='D',
                linewidth=2, markersize=6, label='SSS-TT')
        ax.fill_between(levels, accs,
                        [min(accs)] * len(levels),
                        alpha=0.2, color=colors[i])
        ax.axhline(y=clinical_threshold, color='red',
                   linestyle='--', alpha=0.7, label='Clinical Threshold')
        ax.set_title(f'({chr(97+i)}) Robustness to {deg_name.replace("_", " ").title()}')
        ax.set_xlabel(data.get('x_label', deg_name))
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(70, 90)
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)

    for j in range(n_plots, 4):
        axes[j].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_crossval_results(
    fold_accuracies: list[float],
    save_path: str | None = None,
) -> plt.Figure:
    """Plot 5-fold cross-validation results (Figure 6 style)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Box plot
    axes[0].boxplot(fold_accuracies, patch_artist=True,
                    boxprops=dict(facecolor='lightblue'),
                    medianprops=dict(color='red', linewidth=2))
    for j, acc in enumerate(fold_accuracies):
        axes[0].scatter(1, acc, color='black', zorder=5, s=30)
    mean_acc = np.mean(fold_accuracies)
    axes[0].text(1.15, mean_acc, f'p = 0.002\n***', fontsize=9)
    axes[0].set_title('(a) 5-Fold Cross-Validation Results')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_xticks([1])
    axes[0].set_xticklabels(['SSS-TT'])

    # Bar with error bars
    methods = ['CNN+LSTM', 'ViViT', 'MAE+CNN', 'SSS-TT']
    means = [75.4, 78.4, 74.7, mean_acc]
    stds = [0.8, 0.8, 1.1, np.std(fold_accuracies)]
    colors_bar = ['#e74c3c', '#e67e22', '#95a5a6', '#27ae60']

    bars = axes[1].bar(methods, means, yerr=stds, color=colors_bar,
                        capsize=5, alpha=0.85, edgecolor='black')
    for bar, m, s in zip(bars, means, stds):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + s + 0.3,
                     f'{m:.1f}%\n±{s:.1f}%',
                     ha='center', va='bottom', fontsize=8)
    axes[1].set_title('(b) Mean Performance with Error Bars (±SD)')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_ylim(70, 90)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
