"""
ROC Curve and AUC Analysis
===========================

Generates One-vs-Rest ROC curves for each emotion class and identifies
classes with poor discrimination ability.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import torch.nn.functional as F


def generate_roc_curves(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: List[str],
    output_dir: Path = Path("results/evaluation")
) -> Dict:
    """
    Generate ROC curves for each emotion class (One-vs-Rest).
    
    Args:
        model: Trained emotion recognition model
        dataloader: Test/validation dataloader
        device: Device to run on
        class_names: List of emotion class names
        output_dir: Directory to save outputs
    
    Returns:
        Dict with AUC scores and poor discriminators
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    all_probs = []
    all_labels = []
    
    print(f"\n{'='*80}")
    print("GENERATING ROC CURVES")
    print(f"{'='*80}")
    print("Computing prediction probabilities...")
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Binarize labels for One-vs-Rest
    n_classes = len(class_names)
    y_bin = label_binarize(all_labels, classes=list(range(n_classes)))
    
    # Compute ROC curve and AUC for each class
    roc_data = {}
    auc_scores = {}
    
    for class_idx in range(n_classes):
        fpr, tpr, thresholds = roc_curve(y_bin[:, class_idx], all_probs[:, class_idx])
        roc_auc = auc(fpr, tpr)
        
        roc_data[class_idx] = {
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc
        }
        auc_scores[class_names[class_idx]] = roc_auc
    
    # Print AUC scores
    print(f"\n{'='*80}")
    print("AUC SCORES (Area Under ROC Curve)")
    print(f"{'='*80}")
    print(f"{'Emotion':<15} {'AUC':<10} {'Status':<20}")
    print(f"{'-'*80}")
    
    poor_discriminators = []
    for class_name, auc_score in sorted(auc_scores.items(), key=lambda x: x[1]):
        if auc_score < 0.70:
            status = "⚠ POOR DISCRIMINATION"
            poor_discriminators.append((class_name, auc_score))
        elif auc_score < 0.80:
            status = "⚠ Fair"
        elif auc_score < 0.90:
            status = "✓ Good"
        else:
            status = "✓ Excellent"
        
        print(f"{class_name:<15} {auc_score:>8.4f}   {status:<20}")
    
    # Compute macro and micro AUC
    micro_auc = roc_auc_score(y_bin, all_probs, average='micro')
    macro_auc = np.mean(list(auc_scores.values()))
    
    print(f"{'-'*80}")
    print(f"{'Macro Average':<15} {macro_auc:>8.4f}")
    print(f"{'Micro Average':<15} {micro_auc:>8.4f}")
    
    if poor_discriminators:
        print(f"\n{'='*80}")
        print("⚠ POOR DISCRIMINATORS (AUC < 0.70)")
        print(f"{'='*80}")
        for class_name, auc_score in poor_discriminators:
            print(f"  {class_name}: AUC = {auc_score:.4f}")
            print(f"    → Model struggles to distinguish this class from others")
            print(f"    → Consider: More training data, higher class weight, or better features")
    
    # Generate ROC curve plot
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
    
    for class_idx in range(n_classes):
        data = roc_data[class_idx]
        plt.plot(
            data['fpr'], data['tpr'],
            color=colors[class_idx],
            lw=2.5,
            label=f"{class_names[class_idx]} (AUC = {data['auc']:.3f})"
        )
    
    # Plot random classifier diagonal
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier', alpha=0.5)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Emotion Recognition (One-vs-Rest)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    roc_png_path = output_dir / "roc_curves.png"
    plt.savefig(roc_png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ ROC curves saved to: {roc_png_path}")
    
    print(f"{'='*80}\n")
    
    return {
        'auc_scores': auc_scores,
        'macro_auc': macro_auc,
        'micro_auc': micro_auc,
        'poor_discriminators': poor_discriminators,
        'roc_data': roc_data,
        'predictions': all_probs,
        'labels': all_labels
    }


def suggest_threshold_adjustments(
    roc_result: Dict,
    class_names: List[str],
    target_recall: float = 0.80
) -> Dict:
    """
    Suggest optimal decision thresholds for each class based on target recall.
    
    Args:
        roc_result: Output from generate_roc_curves()
        class_names: List of emotion class names
        target_recall: Target recall level (default 0.80)
    
    Returns:
        Dict mapping class name to optimal threshold
    """
    suggestions = {}
    
    for class_idx, class_name in enumerate(class_names):
        data = roc_result['roc_data'][class_idx]
        tpr = data['tpr']
        thresholds = data['thresholds']
        
        # Find threshold closest to target recall
        idx = np.argmin(np.abs(tpr - target_recall))
        optimal_threshold = thresholds[idx]
        
        suggestions[class_name] = {
            'threshold': float(optimal_threshold),
            'target_recall': target_recall,
            'expected_tpr': float(tpr[idx])
        }
    
    return suggestions
