#!/usr/bin/env python3
"""
Grid Search Results Visualization
==================================

Generates graphs and charts for grid search analysis.

Usage:
    python scripts/visualize_grid_search.py
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Output directory for plots
output_dir = project_root / 'docs' / 'images'
output_dir.mkdir(parents=True, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12


def load_phase1_results():
    """Load Phase 1 grid search results."""
    csv_path = project_root / 'grid_search_results' / 'grid_search_phase1' / 'grid_search_results.csv'
    if not csv_path.exists():
        # Try alternate location
        csv_path = project_root / 'grid_search_results' / 'phase1' / 'grid_search_results.csv'
    if not csv_path.exists():
        print(f"Warning: grid_search_results.csv not found")
        return None
    return pd.read_csv(csv_path)


def load_batch64_results():
    """Load quick batch64 test results."""
    import json
    json_path = project_root / 'grid_search_results' / 'quick_batch64_test' / 'quick_test_results.json'
    if not json_path.exists():
        print(f"Warning: quick_test_results.json not found")
        return None
    with open(json_path) as f:
        return json.load(f)


def plot_phase1_comparison(df):
    """Bar chart comparing all Phase 1 configurations."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    configs = df['config_name'].tolist()
    accuracies = df['stage3_val_acc'].tolist()
    
    # Color bars - highlight the best one
    colors = ['#3498db'] * len(configs)
    best_idx = accuracies.index(max(accuracies))
    colors[best_idx] = '#e74c3c'  # Red for best
    
    bars = ax.bar(range(len(configs)), accuracies, color=colors, edgecolor='white', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Formatting
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Stage 3 Validation Accuracy (%)', fontsize=12)
    ax.set_title('Phase 1 Grid Search: All 8 Configurations Compared', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels([f'#{i+1}' for i in range(len(configs))], fontsize=10)
    ax.set_ylim(55, 63)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#e74c3c', label='Best Config (#8)'),
                       Patch(facecolor='#3498db', label='Other Configs')]
    ax.legend(handles=legend_elements, loc='upper left')
    
    # Add horizontal line for baseline
    ax.axhline(y=60, color='gray', linestyle='--', alpha=0.7, label='60% baseline')
    
    plt.tight_layout()
    save_path = output_dir / 'phase1_config_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_batch_size_comparison(df, batch64_results):
    """Compare batch_size=32 vs batch_size=64."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Data
    stages = ['Stage 1', 'Stage 2', 'Stage 3']
    batch32 = [34.90, 51.93, 61.02]  # Config #8 with batch=32
    batch64 = [
        batch64_results['results']['stage1']['acc'],
        batch64_results['results']['stage2']['acc'],
        batch64_results['results']['stage3']['acc']
    ]
    
    # Left plot: Accuracy comparison
    ax1 = axes[0]
    x = np.arange(len(stages))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, batch32, width, label='batch_size=32', color='#3498db', edgecolor='white')
    bars2 = ax1.bar(x + width/2, batch64, width, label='batch_size=64', color='#2ecc71', edgecolor='white')
    
    # Add value labels
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=10)
    
    ax1.set_xlabel('Training Stage', fontsize=12)
    ax1.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax1.set_title('Accuracy: batch_size=32 vs batch_size=64', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages)
    ax1.legend(loc='upper left')
    ax1.set_ylim(30, 68)
    
    # Right plot: Training time comparison
    ax2 = axes[1]
    times = ['batch_size=32', 'batch_size=64']
    time_values = [13.35, batch64_results['total_time_minutes']]
    colors = ['#3498db', '#2ecc71']
    
    bars = ax2.bar(times, time_values, color=colors, edgecolor='white', linewidth=2)
    
    # Add value labels and improvement
    for bar, val in zip(bars, time_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                f'{val:.1f} min', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add improvement arrow
    improvement = (13.35 - time_values[1]) / 13.35 * 100
    ax2.annotate(f'{improvement:.0f}% faster!', 
                xy=(1, time_values[1]), xytext=(0.5, 8),
                fontsize=12, color='#27ae60', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2))
    
    ax2.set_ylabel('Training Time (minutes)', fontsize=12)
    ax2.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 16)
    
    plt.tight_layout()
    save_path = output_dir / 'batch_size_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_stage_progression(batch64_results):
    """Show accuracy progression through training stages."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    stages = ['Baseline\n(Random)', 'Stage 1\n(Warmup)', 'Stage 2\n(Progressive)', 'Stage 3\n(Deep)']
    accuracies = [
        14.29,  # Random guess for 7 classes
        batch64_results['results']['stage1']['acc'],
        batch64_results['results']['stage2']['acc'],
        batch64_results['results']['stage3']['acc']
    ]
    
    colors = ['#95a5a6', '#e74c3c', '#f39c12', '#2ecc71']
    
    # Plot line with markers
    ax.plot(range(len(stages)), accuracies, 'o-', markersize=15, linewidth=3, 
            color='#34495e', markerfacecolor='white', markeredgewidth=3)
    
    # Add colored circles at each point
    for i, (acc, color) in enumerate(zip(accuracies, colors)):
        ax.scatter(i, acc, s=200, c=color, zorder=5, edgecolors='white', linewidth=2)
        
        # Add labels
        offset = 3 if i > 0 else 2
        ax.text(i, acc + offset, f'{acc:.1f}%', ha='center', va='bottom', 
                fontsize=12, fontweight='bold', color=color)
    
    # Add improvement annotations
    improvements = ['+20.9%', '+17.0%', '+9.8%']
    for i in range(1, len(stages)):
        mid_y = (accuracies[i-1] + accuracies[i]) / 2
        ax.annotate(improvements[i-1], xy=(i-0.5, mid_y), 
                   fontsize=10, color='#7f8c8d', ha='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#bdc3c7'))
    
    ax.set_xlabel('Training Phase', fontsize=12)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax.set_title('3-Stage Progressive Training: Accuracy Progression', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, fontsize=11)
    ax.set_ylim(10, 70)
    ax.set_xlim(-0.5, 3.5)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / 'stage_progression.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_hyperparameter_heatmap(df):
    """Heatmap showing hyperparameter effects."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pivot data for dropout vs stage3_epochs
    pivot1 = df.pivot_table(values='stage3_val_acc', 
                            index='dropout', 
                            columns='stage3_epochs', 
                            aggfunc='mean')
    
    # Left: Dropout vs Stage3 epochs (averaged over stage1)
    ax1 = axes[0]
    im1 = ax1.imshow(pivot1.values, cmap='RdYlGn', aspect='auto', vmin=57, vmax=62)
    
    ax1.set_xticks(range(len(pivot1.columns)))
    ax1.set_xticklabels(pivot1.columns)
    ax1.set_yticks(range(len(pivot1.index)))
    ax1.set_yticklabels(pivot1.index)
    ax1.set_xlabel('Stage 3 Epochs', fontsize=12)
    ax1.set_ylabel('Dropout Rate', fontsize=12)
    ax1.set_title('Accuracy: Dropout × Stage3 Epochs', fontsize=12, fontweight='bold')
    
    # Add text annotations
    for i in range(len(pivot1.index)):
        for j in range(len(pivot1.columns)):
            val = pivot1.values[i, j]
            ax1.text(j, i, f'{val:.1f}%', ha='center', va='center', 
                    fontsize=11, fontweight='bold', color='white' if val < 59 else 'black')
    
    plt.colorbar(im1, ax=ax1, label='Val Accuracy (%)')
    
    # Right: Dropout vs Stage1 epochs (averaged over stage3)
    pivot2 = df.pivot_table(values='stage3_val_acc', 
                            index='dropout', 
                            columns='stage1_epochs', 
                            aggfunc='mean')
    
    ax2 = axes[1]
    im2 = ax2.imshow(pivot2.values, cmap='RdYlGn', aspect='auto', vmin=57, vmax=62)
    
    ax2.set_xticks(range(len(pivot2.columns)))
    ax2.set_xticklabels(pivot2.columns)
    ax2.set_yticks(range(len(pivot2.index)))
    ax2.set_yticklabels(pivot2.index)
    ax2.set_xlabel('Stage 1 Epochs', fontsize=12)
    ax2.set_ylabel('Dropout Rate', fontsize=12)
    ax2.set_title('Accuracy: Dropout × Stage1 Epochs', fontsize=12, fontweight='bold')
    
    # Add text annotations
    for i in range(len(pivot2.index)):
        for j in range(len(pivot2.columns)):
            val = pivot2.values[i, j]
            ax2.text(j, i, f'{val:.1f}%', ha='center', va='center', 
                    fontsize=11, fontweight='bold', color='white' if val < 59 else 'black')
    
    plt.colorbar(im2, ax=ax2, label='Val Accuracy (%)')
    
    plt.tight_layout()
    save_path = output_dir / 'hyperparameter_heatmap.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_training_curve(batch64_results):
    """Plot Stage 3 training curve from batch64 test."""
    csv_path = project_root / 'grid_search_results' / 'quick_batch64_test' / 'emotion_stage3_training.csv'
    if not csv_path.exists():
        print(f"Warning: emotion_stage3_training.csv not found")
        return
    
    df = pd.read_csv(csv_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Accuracy
    ax1 = axes[0]
    ax1.plot(df['epoch'], df['train_acc'], 'o-', label='Train Accuracy', color='#3498db', linewidth=2, markersize=6)
    ax1.plot(df['epoch'], df['val_acc'], 's-', label='Val Accuracy', color='#e74c3c', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Stage 3 Training: Accuracy', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 21)
    
    # Right: Loss
    ax2 = axes[1]
    ax2.plot(df['epoch'], df['train_loss'], 'o-', label='Train Loss', color='#3498db', linewidth=2, markersize=6)
    ax2.plot(df['epoch'], df['val_loss'], 's-', label='Val Loss', color='#e74c3c', linewidth=2, markersize=6)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Stage 3 Training: Loss', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 21)
    
    plt.tight_layout()
    save_path = output_dir / 'training_curves.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_key_findings_summary():
    """Create a visual summary of key findings."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.96, 'Grid Search Key Findings', fontsize=22, fontweight='bold',
            ha='center', va='top', transform=ax.transAxes)
    
    # Findings boxes - 2x2 grid layout
    findings = [
        ('BATCH SIZE', 'batch_size=64 > 32', '+0.86% accuracy\n23% faster training', '#2ecc71', 0.25, 0.68),
        ('DROPOUT', 'dropout=0.5 optimal', 'Better regularization\nfor deep fine-tuning', '#3498db', 0.75, 0.68),
        ('STAGE 3', '20 epochs needed', 'Critical for final\naccuracy gains', '#e74c3c', 0.25, 0.35),
        ('WARMUP', '30 epochs Stage 1', 'Better classifier\ninitialization', '#f39c12', 0.75, 0.35),
    ]
    
    for title, main, detail, color, x, y in findings:
        # Box
        rect = plt.Rectangle((x-0.18, y-0.14), 0.36, 0.22, 
                             facecolor=color, alpha=0.12, edgecolor=color, linewidth=2.5,
                             transform=ax.transAxes)
        ax.add_patch(rect)
        
        # Title
        ax.text(x, y+0.08, title, fontsize=15, fontweight='bold', 
                ha='center', va='center', transform=ax.transAxes, color=color)
        # Main text
        ax.text(x, y-0.01, main, fontsize=13, fontweight='bold',
                ha='center', va='center', transform=ax.transAxes, color='#2c3e50')
        # Detail text
        ax.text(x, y-0.09, detail, fontsize=10, 
                ha='center', va='center', transform=ax.transAxes, color='#555', style='italic')
    
    # Bottom summary box
    ax.text(0.5, 0.16, 'BEST RESULT: 61.88% Validation Accuracy', 
            fontsize=17, fontweight='bold', ha='center', va='center', 
            transform=ax.transAxes, color='#27ae60',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#e8f8f5', edgecolor='#27ae60', linewidth=2.5))
    
    ax.text(0.5, 0.05, 'Config: dropout=0.5 | stages=30/15/20 | batch_size=64 | weight_decay=1e-5',
            fontsize=11, ha='center', va='center', transform=ax.transAxes, color='#34495e', style='italic')
    
    plt.tight_layout()
    save_path = output_dir / 'key_findings_summary.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {save_path}")


def main():
    print("=" * 60)
    print("GRID SEARCH VISUALIZATION")
    print("=" * 60)
    print(f"Output directory: {output_dir}\n")
    
    # Load data
    df = load_phase1_results()
    batch64 = load_batch64_results()
    
    if df is None or batch64 is None:
        print("Error: Could not load results data")
        return
    
    print(f"Loaded Phase 1 results: {len(df)} configs")
    print(f"Loaded batch64 test results: {batch64['results']['stage3']['acc']:.2f}% accuracy\n")
    
    # Generate plots
    print("Generating visualizations...")
    
    plot_phase1_comparison(df)
    plot_batch_size_comparison(df, batch64)
    plot_stage_progression(batch64)
    plot_hyperparameter_heatmap(df)
    plot_training_curve(batch64)
    plot_key_findings_summary()
    
    print("\n" + "=" * 60)
    print("✓ All visualizations generated!")
    print(f"  Location: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
