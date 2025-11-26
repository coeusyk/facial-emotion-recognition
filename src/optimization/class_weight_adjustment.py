"""
Component 1: Targeted Class Weight Rebalancing
===============================================

Purpose:
    Fix the Sad over-prediction and Fear under-detection through surgical 
    weight adjustments based on Phase 1 diagnostic findings.

Phase 1 Diagnosed Issues:
    - Fear: Recall 22.3%, F1 27.5% → Aggressive boost needed
    - Angry: Recall 31.3%, F1 37.1% → Moderate boost needed
    - Sad: Over-predicted across 4 confusion pairs → Reduce weight
    - Disgust: Precision 18.8%, F1 28.1% → Reduce weight (over-predicting)
    - Happy: F1 77.8% → Slight reduction (over-learning)
    - Surprise: F1 68.6% → Keep baseline
    - Neutral: F1 49.7% → Keep baseline

Expected Gain: +3-4% accuracy, +0.05 macro F1

Author: FER-2013 Optimization Pipeline
"""

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple
from collections import Counter
from torchvision import datasets


class WeightAdjustmentStrategy:
    """
    Defines adjustment strategies for class weights based on Phase 1 diagnostics.
    """
    
    # Conservative strategy: Factor range [0.85-1.5]
    CONSERVATIVE = {
        'angry': 1.3,
        'disgust': 0.85,
        'fear': 1.5,
        'happy': 0.95,
        'neutral': 1.0,
        'sad': 0.9,
        'surprise': 1.0
    }
    
    # Moderate strategy: Factor range [0.7-2.5] (RECOMMENDED)
    MODERATE = {
        'angry': 1.8,
        'disgust': 0.7,
        'fear': 2.5,
        'happy': 0.9,
        'neutral': 1.0,
        'sad': 0.8,
        'surprise': 1.0
    }
    
    # Aggressive strategy: Factor range [0.5-3.0]
    AGGRESSIVE = {
        'angry': 2.2,
        'disgust': 0.5,
        'fear': 3.0,
        'happy': 0.85,
        'neutral': 1.0,
        'sad': 0.6,
        'surprise': 1.0
    }
    
    @classmethod
    def get_strategy(cls, name: str) -> Dict[str, float]:
        """Get adjustment factors for a strategy by name."""
        strategies = {
            'conservative': cls.CONSERVATIVE,
            'moderate': cls.MODERATE,
            'aggressive': cls.AGGRESSIVE
        }
        
        if name.lower() not in strategies:
            raise ValueError(f"Unknown strategy: {name}. Choose from {list(strategies.keys())}")
        
        return strategies[name.lower()]


def calculate_effective_number_weights(data_dir: Path, beta: float = 0.9999) -> Tuple[torch.Tensor, list]:
    """
    Calculate base Effective Number weights from training data.
    
    Args:
        data_dir: Path to training data directory
        beta: Smoothing parameter (0.9999 recommended)
    
    Returns:
        Tuple of (weights tensor, class names)
    """
    dataset = datasets.ImageFolder(root=data_dir)
    labels = [label for _, label in dataset.samples]
    class_counts = Counter(labels)
    
    samples_per_class = np.array([class_counts[i] for i in range(len(class_counts))])
    
    # Effective number formula
    effective_num = 1.0 - np.power(beta, samples_per_class)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * len(class_counts)  # Normalize
    
    weights_tensor = torch.FloatTensor(weights)
    
    print("\nEffective Number Base Weights (beta={:.4f}):".format(beta))
    for i, (class_name, weight) in enumerate(zip(dataset.classes, weights_tensor)):
        count = class_counts[i]
        print(f"  {class_name:12s}: {weight:5.2f} (samples: {count:5d})")
    
    return weights_tensor, dataset.classes


def calculate_diagnostic_adjusted_weights(
    base_weights: torch.Tensor,
    class_names: list,
    strategy: str = 'moderate',
    phase1_metrics: Dict = None
) -> torch.Tensor:
    """
    Adjust class weights based on Phase 1 confusion matrix and precision-recall analysis.
    
    Args:
        base_weights: Effective Number base weights (beta=0.9999)
        class_names: List of emotion class names
        strategy: 'conservative', 'moderate', or 'aggressive'
        phase1_metrics: Optional dict containing Phase 1 diagnostic results
                       (not used in this version, reserved for future auto-adjustment)
    
    Returns:
        Adjusted weights tensor
    
    Example:
        >>> base_weights, class_names = calculate_effective_number_weights('data/raw/train')
        >>> adjusted = calculate_diagnostic_adjusted_weights(base_weights, class_names, 'moderate')
    """
    print(f"""\n{'='*80}
CALCULATING DIAGNOSTIC-ADJUSTED CLASS WEIGHTS
{'='*80}
Strategy: {strategy.upper()}""")
    
    # Get adjustment factors for the chosen strategy
    adjustment_factors = WeightAdjustmentStrategy.get_strategy(strategy)
    
    # Apply adjustments
    adjusted = base_weights.clone()
    
    print(f"""\nApplying {strategy} adjustment factors:
{'Emotion':<12} | {'Base':>6} | {'Factor':>6} | {'Adjusted':>8} | {'Reason':<30}""")
    print("-" * 80)
    
    # Reasoning for each adjustment
    reasoning = {
        'angry': 'Low recall (31.3%)',
        'disgust': 'Low precision (18.8%), over-predicted',
        'fear': 'Very low recall (22.3%), critical',
        'happy': 'High F1 (77.8%), slight over-learning',
        'neutral': 'Moderate performance, baseline',
        'sad': 'Over-predicted across 4 classes',
        'surprise': 'Good balance (68.6% F1)'
    }
    
    for i, emotion_name in enumerate(class_names):
        factor = adjustment_factors[emotion_name]
        old_weight = adjusted[i].item()
        adjusted[i] *= factor
        new_weight = adjusted[i].item()
        
        change_symbol = "↑" if factor > 1.0 else "↓" if factor < 1.0 else "="
        print(f"{emotion_name:<12} | {old_weight:6.2f} | {factor:6.2f} | {new_weight:8.2f} {change_symbol} | {reasoning[emotion_name]}")
    
    print("=" * 80)
    
    return adjusted


def test_weight_strategies(
    data_dir: Path,
    strategies: list = ['conservative', 'moderate', 'aggressive'],
    output_dir: Path = None
) -> pd.DataFrame:
    """
    Compare all weight adjustment strategies.
    
    Args:
        data_dir: Path to training data
        strategies: List of strategy names to test
        output_dir: Where to save comparison outputs
    
    Returns:
        DataFrame with comparison results
    """
    print(f"""\n{'='*80}
TESTING WEIGHT ADJUSTMENT STRATEGIES""")
    print(f"{'='*80}")
    
    # Calculate base weights
    base_weights, class_names = calculate_effective_number_weights(data_dir)
    
    results = []
    all_weights = {'base': base_weights}
    
    # Test each strategy
    for strategy in strategies:
        print(f"""\n\n{'='*80}
STRATEGY: {strategy.upper()}""")
        print(f"{'='*80}")
        
        adjusted = calculate_diagnostic_adjusted_weights(base_weights, class_names, strategy)
        all_weights[strategy] = adjusted
        
        # Calculate statistics
        for i, emotion in enumerate(class_names):
            results.append({
                'strategy': strategy,
                'emotion': emotion,
                'base_weight': base_weights[i].item(),
                'adjusted_weight': adjusted[i].item(),
                'factor': adjusted[i].item() / base_weights[i].item(),
                'change': adjusted[i].item() - base_weights[i].item()
            })
    
    df = pd.DataFrame(results)
    
    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path = output_dir / 'weight_strategy_comparison.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Results saved to: {csv_path}")
        
        # Create visualization
        _visualize_weight_comparison(all_weights, class_names, output_dir)
    
    return df


def _visualize_weight_comparison(
    weights_dict: Dict[str, torch.Tensor],
    class_names: list,
    output_dir: Path
):
    """Create visualization comparing different weight strategies."""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Prepare data
    strategies = list(weights_dict.keys())
    weights_array = np.array([[w[i].item() for i in range(len(class_names))] 
                              for w in weights_dict.values()])
    
    # Plot 1: Grouped bar chart
    x = np.arange(len(class_names))
    width = 0.2
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    for i, (strategy, color) in enumerate(zip(strategies, colors)):
        offset = (i - len(strategies)/2 + 0.5) * width
        axes[0].bar(x + offset, weights_array[i], width, 
                   label=strategy.capitalize(), color=color, alpha=0.8)
    
    axes[0].set_xlabel('Emotion Class', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Class Weight', fontsize=12, fontweight='bold')
    axes[0].set_title('Class Weight Comparison Across Strategies', 
                     fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0].legend(loc='upper right')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Heatmap of weights
    df_heatmap = pd.DataFrame(weights_array, 
                              index=[s.capitalize() for s in strategies],
                              columns=class_names)
    
    sns.heatmap(df_heatmap, annot=True, fmt='.2f', cmap='RdYlGn', 
                ax=axes[1], cbar_kws={'label': 'Weight Value'},
                vmin=0, vmax=weights_array.max())
    
    axes[1].set_title('Class Weights Heatmap', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Emotion Class', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Strategy', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    save_path = output_dir / 'weight_strategy_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualization saved to: {save_path}")


def save_weight_comparison_report(
    comparison_df: pd.DataFrame,
    output_dir: Path
):
    """
    Generate a detailed comparison report.
    
    Args:
        comparison_df: DataFrame from test_weight_strategies()
        output_dir: Where to save the report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / 'weight_adjustment_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("CLASS WEIGHT ADJUSTMENT COMPARISON REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Phase 1 Diagnostic Summary:\n")
        f.write("-" * 80 + "\n")
        f.write("Critical Issues:\n")
        f.write("  - Fear: Recall 22.3%, F1 27.5% (severe under-detection)\n")
        f.write("  - Angry: Recall 31.3%, F1 37.1% (moderate under-detection)\n")
        f.write("  - Sad: Over-predicted across 4 confusion pairs\n")
        f.write("  - Disgust: Precision 18.8%, F1 28.1% (over-predicting)\n\n")
        
        # Strategy summaries
        for strategy in comparison_df['strategy'].unique():
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"STRATEGY: {strategy.upper()}\n")
            f.write("=" * 80 + "\n\n")
            
            strategy_df = comparison_df[comparison_df['strategy'] == strategy]
            
            f.write(f"{'Emotion':<12} | {'Base':>8} | {'Adjusted':>8} | {'Factor':>6} | {'Change':>8}\n")
            f.write("-" * 80 + "\n")
            
            for _, row in strategy_df.iterrows():
                f.write(f"{row['emotion']:<12} | {row['base_weight']:8.2f} | "
                       f"{row['adjusted_weight']:8.2f} | {row['factor']:6.2f} | "
                       f"{row['change']:+8.2f}\n")
            
            # Summary statistics
            f.write("\nSummary:\n")
            f.write(f"  Mean adjustment factor: {strategy_df['factor'].mean():.2f}\n")
            f.write(f"  Max boost (Fear): {strategy_df.loc[strategy_df['emotion']=='fear', 'factor'].values[0]:.2f}x\n")
            f.write(f"  Max reduction (Disgust): {strategy_df.loc[strategy_df['emotion']=='disgust', 'factor'].values[0]:.2f}x\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("=" * 80 + "\n\n")
        f.write("1. RECOMMENDED STRATEGY: Moderate\n")
        f.write("   - Balanced adjustments (0.7-2.5x range)\n")
        f.write("   - Expected gain: +3-4% accuracy, +0.05 macro F1\n")
        f.write("   - Targets critical issues without over-correction\n\n")
        
        f.write("2. Next Steps:\n")
        f.write("   a) Train Stage 3 with moderate adjusted weights\n")
        f.write("   b) Compare validation metrics to baseline\n")
        f.write("   c) Check confusion matrix for improvements:\n")
        f.write("      - Fear -> Sad should drop from 24.3% to <18%\n")
        f.write("      - Angry -> Sad should drop from 22.3% to <16%\n")
        f.write("      - Neutral -> Sad should drop from 19.7% to <15%\n\n")
        
        f.write("3. If moderate strategy underperforms:\n")
        f.write("   - Try aggressive strategy for Fear/Angry\n")
        f.write("   - Monitor for over-correction (Happy/Surprise degradation)\n\n")
    
    print(f"✓ Detailed report saved to: {report_path}")


def save_adjusted_weights(
    weights: torch.Tensor,
    class_names: list,
    strategy: str,
    output_dir: Path = None
):
    """
    Save adjusted weights to a .pth file for use in training.
    
    Args:
        weights: Adjusted weights tensor
        class_names: List of class names
        strategy: Strategy name (for filename)
        output_dir: Where to save (default: configs/)
    """
    if output_dir is None:
        output_dir = Path('configs')
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as .pth file
    save_dict = {
        'weights': weights,
        'class_names': class_names,
        'strategy': strategy,
        'description': f'Diagnostic-adjusted class weights ({strategy} strategy)'
    }
    
    filename = f'class_weights_{strategy}.pth'
    save_path = output_dir / filename
    
    torch.save(save_dict, save_path)
    print(f"""\n✓ Adjusted weights saved to: {save_path}
  Load with: torch.load('{save_path}')""")


def main():
    """Example usage of class weight adjustment."""
    
    # Configuration
    DATA_DIR = Path('data/raw/train')
    OUTPUT_DIR = Path('results/optimization/class_weights')
    
    if not DATA_DIR.exists():
        print(f"""Error: Data directory not found: {DATA_DIR}
Please update DATA_DIR in this script""")
        return
    
    print(f"""{'='*80}
COMPONENT 1: CLASS WEIGHT ADJUSTMENT""")
    print("=" * 80)
    
    # Test all strategies
    comparison_df = test_weight_strategies(
        data_dir=DATA_DIR,
        strategies=['conservative', 'moderate', 'aggressive'],
        output_dir=OUTPUT_DIR
    )
    
    # Generate detailed report
    save_weight_comparison_report(comparison_df, OUTPUT_DIR)
    
    # Save recommended (moderate) weights
    base_weights, class_names = calculate_effective_number_weights(DATA_DIR)
    moderate_weights = calculate_diagnostic_adjusted_weights(
        base_weights, class_names, strategy='moderate'
    )
    
    save_adjusted_weights(
        weights=moderate_weights,
        class_names=class_names,
        strategy='moderate',
        output_dir=Path('configs')
    )
    
    print(f"""\n{'='*80}
CLASS WEIGHT ADJUSTMENT COMPLETE
{'='*80}

Next steps:
1. Review comparison plots in: {OUTPUT_DIR}
2. Use moderate weights in training:
   checkpoint = torch.load('configs/class_weights_moderate.pth')
   weights = checkpoint['weights']
   criterion = nn.CrossEntropyLoss(weight=weights)
{'='*80}""")


if __name__ == '__main__':
    main()
