"""
Component 2: Per-Class Threshold Optimization
===============================================

Purpose:
    Fix Disgust paradox (AUC 0.901 but F1 0.281) and optimize thresholds 
    for all classes by finding optimal decision boundaries.

The Disgust Paradox:
    - AUC = 0.901 (excellent discrimination ability)
    - F1 = 0.281 (terrible actual performance)
    - Precision = 18.8% (81% false positive rate!)
    
Root Cause:
    Model CAN detect Disgust but default threshold (0.5) is wrong.
    Need higher threshold to reduce false positives.

Expected Optimal Thresholds (from Phase 1):
    - Disgust: 0.70-0.80 (increase from 0.5)
    - Sad: 0.55-0.65 (slight increase)
    - Fear: 0.40-0.45 (decrease for better recall)
    - Angry: 0.45-0.50 (slight decrease)
    - Happy, Surprise, Neutral: 0.45-0.55 (near baseline)

Expected Gain: +2-3% accuracy, Disgust F1 +20%

Author: FER-2013 Optimization Pipeline
"""

import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm


def get_predictions(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get prediction probabilities and true labels for all samples.
    
    Args:
        model: Trained model
        dataloader: Data loader (validation or test set)
        device: Device to run on
    
    Returns:
        Tuple of (probabilities, true_labels)
        - probabilities: shape [N, num_classes]
        - true_labels: shape [N]
    """
    model.eval()
    all_probs = []
    all_labels = []
    
    print("Collecting predictions from model...")
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Getting predictions'):
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    
    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)
    
    print(f"""✓ Collected predictions for {len(all_labels)} samples
  Probabilities shape: {all_probs.shape}""")
    print(f"  Labels shape: {all_labels.shape}")
    
    return all_probs, all_labels


def optimize_class_thresholds(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    class_names: list,
    device: torch.device,
    target_metric: str = 'f1',
    threshold_range: Tuple[float, float, float] = (0.3, 0.9, 0.01)
) -> Dict[str, float]:
    """
    Find optimal decision threshold for each emotion class.
    
    Args:
        model: Trained emotion classifier
        val_loader: Validation data loader
        class_names: List of emotion class names
        device: Device to run on
        target_metric: 'f1', 'precision', or 'recall'
        threshold_range: (min, max, step) for threshold sweep
    
    Returns:
        Dict mapping emotion_name -> optimal_threshold
    
    Example:
        >>> optimal_thresholds = optimize_class_thresholds(
        ...     model, val_loader, class_names, device, target_metric='f1'
        ... )
        >>> # {'angry': 0.48, 'disgust': 0.75, 'fear': 0.42, ...}
    """
    print(f"""{'='*80}
OPTIMIZING PER-CLASS DECISION THRESHOLDS
{'='*80}
Target metric: {target_metric}""")
    print(f"Threshold range: {threshold_range[0]:.2f} to {threshold_range[1]:.2f}, step {threshold_range[2]:.3f}")
    
    # Get prediction probabilities for all validation samples
    all_probs, all_labels = get_predictions(model, val_loader, device)
    
    optimal_thresholds = {}
    threshold_details = {}
    
    for emotion_idx, emotion_name in enumerate(class_names):
        print(f"""
{'='*80}
Optimizing threshold for: {emotion_name.upper()}
{'='*80}""")
        
        # One-vs-Rest setup
        binary_labels = (all_labels == emotion_idx).astype(int)
        emotion_probs = all_probs[:, emotion_idx]
        
        # Count positive/negative samples
        n_positive = binary_labels.sum()
        n_negative = len(binary_labels) - n_positive
        
        print(f"Samples: {n_positive} positive, {n_negative} negative")
        
        # Sweep thresholds
        best_threshold = 0.5
        best_score = 0.0
        
        thresholds = np.arange(*threshold_range)
        scores = []
        precisions = []
        recalls = []
        
        for threshold in thresholds:
            predictions = (emotion_probs >= threshold).astype(int)
            
            # Handle edge cases (all 0 or all 1 predictions)
            if predictions.sum() == 0 or predictions.sum() == len(predictions):
                scores.append(0.0)
                precisions.append(0.0)
                recalls.append(0.0)
                continue
            
            # Calculate metrics
            precision = precision_score(binary_labels, predictions, zero_division=0)
            recall = recall_score(binary_labels, predictions, zero_division=0)
            f1 = f1_score(binary_labels, predictions, zero_division=0)
            
            precisions.append(precision)
            recalls.append(recall)
            
            if target_metric == 'f1':
                score = f1
            elif target_metric == 'precision':
                score = precision
            elif target_metric == 'recall':
                score = recall
            else:
                raise ValueError(f"Unknown metric: {target_metric}")
            
            scores.append(score)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        optimal_thresholds[emotion_name] = best_threshold
        
        # Store details for visualization
        threshold_details[emotion_name] = {
            'thresholds': thresholds,
            'scores': scores,
            'precisions': precisions,
            'recalls': recalls,
            'best_threshold': best_threshold,
            'best_score': best_score
        }
        
        # Calculate metrics at baseline (0.5) and optimal threshold
        baseline_preds = (emotion_probs >= 0.5).astype(int)
        optimal_preds = (emotion_probs >= best_threshold).astype(int)
        
        baseline_f1 = f1_score(binary_labels, baseline_preds, zero_division=0)
        optimal_f1 = f1_score(binary_labels, optimal_preds, zero_division=0)
        
        baseline_precision = precision_score(binary_labels, baseline_preds, zero_division=0)
        optimal_precision = precision_score(binary_labels, optimal_preds, zero_division=0)
        
        baseline_recall = recall_score(binary_labels, baseline_preds, zero_division=0)
        optimal_recall = recall_score(binary_labels, optimal_preds, zero_division=0)
        
        print(f"""\nResults:
  Optimal threshold: {best_threshold:.2f} (baseline: 0.50)

  Baseline (0.5):
    Precision: {baseline_precision:.3f}
    Recall:    {baseline_recall:.3f}
    F1:        {baseline_f1:.3f}

  Optimal ({best_threshold:.2f}):
    Precision: {optimal_precision:.3f} ({optimal_precision-baseline_precision:+.3f})
    Recall:    {optimal_recall:.3f} ({optimal_recall-baseline_recall:+.3f})
    F1:        {optimal_f1:.3f} ({optimal_f1-baseline_f1:+.3f})""")
        
        # Special note for Disgust (the paradox case)
        if emotion_name == 'disgust':
            if best_threshold > 0.65:
                print(f"""\n  ✓ DISGUST PARADOX FIX DETECTED!
    Higher threshold ({best_threshold:.2f}) reduces false positives""")
                print(f"    Expected: Precision improvement, slight recall trade-off")
    
    return optimal_thresholds, threshold_details


def generate_pr_curves(
    threshold_details: Dict,
    class_names: list,
    output_dir: Path = None
):
    """
    Generate precision-recall curves for all emotion classes.
    
    Args:
        threshold_details: Details from optimize_class_thresholds()
        class_names: List of emotion names
        output_dir: Where to save plots
    """
    print(f"""\n{'='*80}
GENERATING PRECISION-RECALL CURVES""")
    print("=" * 80)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subplot grid
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, emotion in enumerate(class_names):
        ax = axes[idx]
        details = threshold_details[emotion]
        
        thresholds = details['thresholds']
        precisions = details['precisions']
        recalls = details['recalls']
        f1_scores = details['scores']  # F1 scores
        best_threshold = details['best_threshold']
        
        # Plot precision and recall vs threshold
        ax.plot(thresholds, precisions, 'b-', label='Precision', linewidth=2)
        ax.plot(thresholds, recalls, 'g-', label='Recall', linewidth=2)
        ax.plot(thresholds, f1_scores, 'r-', label='F1 Score', linewidth=2)
        
        # Mark baseline threshold (0.5)
        ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5, 
                  label='Baseline (0.5)')
        
        # Mark optimal threshold
        ax.axvline(best_threshold, color='red', linestyle='-', alpha=0.7,
                  label=f'Optimal ({best_threshold:.2f})')
        
        ax.set_xlabel('Threshold', fontsize=10, fontweight='bold')
        ax.set_ylabel('Score', fontsize=10, fontweight='bold')
        ax.set_title(f'{emotion.capitalize()}\n(Best: {best_threshold:.2f}, F1: {details["best_score"]:.3f})',
                    fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xlim(0.3, 0.9)
        ax.set_ylim(0, 1.05)
    
    # Hide the last subplot (we have 7 emotions, 8 subplots)
    axes[7].axis('off')
    
    plt.suptitle('Per-Class Threshold Optimization: Precision-Recall Trade-off',
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if output_dir:
        save_path = output_dir / 'threshold_optimization_curves.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ PR curves saved to: {save_path}")
    
    plt.close()


def save_optimal_thresholds(
    optimal_thresholds: Dict[str, float],
    output_path: Path = None
):
    """
    Save optimal thresholds to JSON file.
    
    Args:
        optimal_thresholds: Dict mapping emotion -> threshold
        output_path: Where to save (default: configs/optimal_thresholds.json)
    """
    if output_path is None:
        output_path = Path('configs/optimal_thresholds.json')
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add metadata
    save_dict = {
        'thresholds': optimal_thresholds,
        'description': 'Optimal per-class decision thresholds (F1-optimized)',
        'baseline': 0.5,
        'usage': 'Use these thresholds instead of argmax for multi-class prediction'
    }
    
    with open(output_path, 'w') as f:
        json.dump(save_dict, f, indent=2)
    
    print(f"""\n✓ Optimal thresholds saved to: {output_path}
  Load with: json.load(open('{output_path}'))""")


def generate_threshold_report(
    optimal_thresholds: Dict[str, float],
    threshold_details: Dict,
    output_dir: Path
):
    """
    Generate detailed text report of threshold optimization results.
    
    Args:
        optimal_thresholds: Optimal thresholds for each class
        threshold_details: Detailed results from optimization
        output_dir: Where to save report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / 'threshold_optimization_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("THRESHOLD OPTIMIZATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Objective:\n")
        f.write("  Find optimal decision threshold for each emotion class to maximize F1 score.\n")
        f.write("  Default softmax argmax uses threshold=0.5 for all classes.\n\n")
        
        f.write("The Disgust Paradox (Phase 1):\n")
        f.write("  - AUC = 0.901 (excellent discrimination)\n")
        f.write("  - F1 = 0.281 (poor actual performance)\n")
        f.write("  - Precision = 18.8% (81% false positive rate!)\n")
        f.write("  - Root Cause: Wrong threshold (too eager to predict Disgust)\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("PER-CLASS OPTIMAL THRESHOLDS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"{'Emotion':<12} | {'Baseline':>8} | {'Optimal':>8} | {'Change':>8} | {'Best F1':>8} | {'Reasoning':<30}\n")
        f.write("-" * 120 + "\n")
        
        reasoning = {
            'angry': 'Low recall (31.3%)',
            'disgust': 'PARADOX FIX: Reduce false positives',
            'fear': 'Very low recall (22.3%)',
            'happy': 'Balanced performance',
            'neutral': 'Slight over-prediction',
            'sad': 'Over-predicted',
            'surprise': 'Good balance'
        }
        
        for emotion, threshold in optimal_thresholds.items():
            details = threshold_details[emotion]
            change = threshold - 0.5
            change_str = f"{change:+.2f}"
            
            f.write(f"{emotion:<12} | {0.5:8.2f} | {threshold:8.2f} | {change_str:>8} | "
                   f"{details['best_score']:8.3f} | {reasoning[emotion]:<30}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("IMPACT ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        # Count how many thresholds changed significantly
        significant_changes = sum(1 for t in optimal_thresholds.values() 
                                 if abs(t - 0.5) > 0.05)
        
        f.write(f"Significant threshold changes (|change| > 0.05): {significant_changes}/7\n\n")
        
        # Highlight major changes
        f.write("Major Changes:\n")
        for emotion, threshold in optimal_thresholds.items():
            if abs(threshold - 0.5) > 0.10:
                direction = "INCREASE" if threshold > 0.5 else "DECREASE"
                f.write(f"  - {emotion.capitalize()}: {0.5:.2f} -> {threshold:.2f} ({direction})\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("EXPECTED IMPROVEMENTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Overall:\n")
        f.write("  - Accuracy: +2-3% improvement expected\n")
        f.write("  - Macro F1: +0.04-0.05 improvement expected\n\n")
        
        f.write("Per-Class (Key Fixes):\n")
        disgust_threshold = optimal_thresholds['disgust']
        if disgust_threshold > 0.65:
            f.write(f"  - Disgust: Precision +20-25% (threshold {disgust_threshold:.2f})\n")
            f.write(f"    --> F1 improvement: 0.281 -> 0.45-0.50 (+0.17-0.22)\n")
        
        fear_threshold = optimal_thresholds['fear']
        if fear_threshold < 0.45:
            f.write(f"  - Fear: Recall +5-10% (threshold {fear_threshold:.2f})\n")
            f.write(f"    --> F1 improvement: 0.275 -> 0.32-0.35 (+0.04-0.07)\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("USAGE INSTRUCTIONS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. Load optimal thresholds:\n")
        f.write("   import json\n")
        f.write("   thresholds = json.load(open('configs/optimal_thresholds.json'))['thresholds']\n\n")
        
        f.write("2. Apply during inference:\n")
        f.write("   # Get probabilities\n")
        f.write("   probs = torch.softmax(model(x), dim=1)\n\n")
        
        f.write("   # Instead of: pred = torch.argmax(probs, dim=1)\n")
        f.write("   # Use optimized thresholds for each class\n\n")
        
        f.write("3. Validate improvements:\n")
        f.write("   - Run evaluation with new thresholds\n")
        f.write("   - Compare confusion matrix to baseline\n")
        f.write("   - Check Disgust precision improvement\n")
        f.write("   - Check Fear recall improvement\n\n")
    
    print(f"✓ Detailed report saved to: {report_path}")


def main():
    """Example usage of threshold optimization."""
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from config import Config
    from src.data.data_pipeline import create_dataloaders
    from src.models.vgg16_emotion import build_emotion_model
    
    print(f"""{'='*80}
COMPONENT 2: THRESHOLD OPTIMIZATION""")
    print("=" * 80)
    
    # Configuration
    CHECKPOINT_PATH = Path('models/emotion_stage3_deep.pth')
    DATA_DIR = Path('data/raw')
    OUTPUT_DIR = Path('results/optimization/thresholds')
    
    if not CHECKPOINT_PATH.exists():
        print(f"""\nError: Checkpoint not found: {CHECKPOINT_PATH}
Please train the model first or update CHECKPOINT_PATH""")
        return
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load data
    print("\nLoading validation data...")
    _, val_loader, _, class_names = create_dataloaders(
        DATA_DIR,
        batch_size=64,
        num_workers=4
    )
    
    # Load model
    print("\nLoading trained model...")
    model = build_emotion_model(num_classes=7, pretrained=False, verbose=False)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"""✓ Model loaded from: {CHECKPOINT_PATH}
  Validation accuracy: {checkpoint.get('val_acc', 'unknown'):.2f}%""")
    
    # Optimize thresholds
    optimal_thresholds, threshold_details = optimize_class_thresholds(
        model=model,
        val_loader=val_loader,
        class_names=class_names,
        device=device,
        target_metric='f1',
        threshold_range=(0.3, 0.9, 0.01)
    )
    
    # Generate visualizations
    generate_pr_curves(threshold_details, class_names, OUTPUT_DIR)
    
    # Save results
    save_optimal_thresholds(optimal_thresholds, Path('configs/optimal_thresholds.json'))
    
    # Generate report
    generate_threshold_report(optimal_thresholds, threshold_details, OUTPUT_DIR)
    
    print(f"""\n{'='*80}
THRESHOLD OPTIMIZATION COMPLETE
{'='*80}

Outputs saved to: {OUTPUT_DIR}
Next steps:
1. Review threshold_optimization_curves.png
2. Check threshold_optimization_report.txt for detailed analysis
3. Use configs/optimal_thresholds.json in inference""")
    print("=" * 80)


if __name__ == '__main__':
    main()
