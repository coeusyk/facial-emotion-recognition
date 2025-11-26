"""
Confusion Matrix Generator for Emotion Recognition
===================================================

Analyzes misclassification patterns to identify which emotion pairs
are commonly confused by the model.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import torch.nn.functional as F


def generate_confusion_matrix(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: List[str],
    output_dir: Path = Path("results/evaluation")
) -> Dict:
    """
    Generate confusion matrix and analyze misclassification patterns.
    
    Args:
        model: Trained emotion recognition model
        dataloader: Test/validation dataloader
        device: Device to run on (cuda/cpu)
        class_names: List of emotion class names
        output_dir: Directory to save outputs
    
    Returns:
        Dict containing:
            - confusion_matrix: np array [7, 7]
            - accuracy: overall accuracy %
            - per_class_metrics: Dict of precision, recall, f1 per class
            - top_misclassifications: List of top confused pairs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    print(f"\n{'='*80}")
    print("GENERATING CONFUSION MATRIX")
    print(f"{'='*80}")
    print("Processing test set predictions...")
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
    
    # Calculate overall accuracy
    overall_acc = np.trace(cm) / cm.sum() * 100
    
    # Calculate per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(len(class_names))), average=None
    )
    
    per_class_metrics = {}
    for idx, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            'precision': precision[idx],
            'recall': recall[idx],
            'f1': f1[idx],
            'support': int(support[idx])
        }
    
    # Find top misclassifications (off-diagonal)
    misclassifications = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j:
                count = cm[i, j]
                if count > 0:
                    percentage = count / cm[i].sum() * 100
                    misclassifications.append({
                        'true': class_names[i],
                        'predicted': class_names[j],
                        'count': count,
                        'percentage': percentage
                    })
    
    # Sort by count descending
    misclassifications.sort(key=lambda x: x['count'], reverse=True)
    top_misclassifications = misclassifications[:10]  # Top 10
    
    # Print results
    print(f"\n{'='*80}")
    print("CONFUSION MATRIX ANALYSIS")
    print(f"{'='*80}")
    print(f"Total samples: {cm.sum()}")
    print(f"Overall accuracy: {overall_acc:.2f}%")
    
    print(f"\n{'='*80}")
    print("TOP MISCLASSIFICATIONS")
    print(f"{'='*80}")
    for idx, mis in enumerate(top_misclassifications, 1):
        print(f"{idx}. {mis['true']:12s} → {mis['predicted']:12s}: "
              f"{mis['count']:4d} ({mis['percentage']:5.2f}%)")
    
    print(f"\n{'='*80}")
    print("PER-CLASS METRICS (Precision / Recall / F1)")
    print(f"{'='*80}")
    print(f"{'Emotion':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print(f"{'-'*80}")
    
    for class_name in class_names:
        metrics = per_class_metrics[class_name]
        print(f"{class_name:<12} {metrics['precision']:>10.4f}   "
              f"{metrics['recall']:>10.4f}   {metrics['f1']:>10.4f}   "
              f"{metrics['support']:>8}")
    
    # Macro and weighted averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    weighted_precision = np.average(precision, weights=support)
    weighted_recall = np.average(recall, weights=support)
    weighted_f1 = np.average(f1, weights=support)
    
    print(f"{'-'*80}")
    print(f"{'Macro Average':<12} {macro_precision:>10.4f}   "
          f"{macro_recall:>10.4f}   {macro_f1:>10.4f}")
    print(f"{'Weighted Avg':<12} {weighted_precision:>10.4f}   "
          f"{weighted_recall:>10.4f}   {weighted_f1:>10.4f}")
    
    # Save confusion matrix as CSV
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_csv_path = output_dir / "confusion_matrix.csv"
    cm_df.to_csv(cm_csv_path)
    print(f"\n✓ Confusion matrix saved to: {cm_csv_path}")
    
    # Generate heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'},
        annot_kws={'size': 10}
    )
    plt.title('Confusion Matrix - Emotion Recognition', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_png_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix heatmap saved to: {cm_png_path}")
    
    # Generate percentage heatmap
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_percent,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage (%)'},
        annot_kws={'size': 9},
        vmin=0,
        vmax=100
    )
    plt.title('Confusion Matrix (Percentage) - Emotion Recognition', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_percent_png_path = output_dir / "confusion_matrix_percentage.png"
    plt.savefig(cm_percent_png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix percentage heatmap saved to: {cm_percent_png_path}")
    
    # Save classification report
    report_path = output_dir / "classification_metrics.txt"
    with open(report_path, 'w') as f:
        f.write("EMOTION RECOGNITION - CLASSIFICATION METRICS\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Overall Accuracy: {overall_acc:.2f}%\n")
        f.write(f"Total Test Samples: {cm.sum()}\n\n")
        
        f.write(f"{'Emotion':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
        f.write(f"{'-'*80}\n")
        for class_name in class_names:
            metrics = per_class_metrics[class_name]
            f.write(f"{class_name:<12} {metrics['precision']:>10.4f}   "
                   f"{metrics['recall']:>10.4f}   {metrics['f1']:>10.4f}   "
                   f"{metrics['support']:>8}\n")
        
        f.write(f"{'-'*80}\n")
        f.write(f"{'Macro Average':<12} {macro_precision:>10.4f}   "
               f"{macro_recall:>10.4f}   {macro_f1:>10.4f}\n")
        f.write(f"{'Weighted Avg':<12} {weighted_precision:>10.4f}   "
               f"{weighted_recall:>10.4f}   {weighted_f1:>10.4f}\n")
        
        f.write(f"\n{'='*80}\n")
        f.write("TOP MISCLASSIFICATIONS\n")
        f.write(f"{'='*80}\n")
        for idx, mis in enumerate(top_misclassifications, 1):
            f.write(f"{idx}. {mis['true']:12s} predicted as {mis['predicted']:12s}: "
                   f"{mis['count']:4d} ({mis['percentage']:5.2f}%)\n")
    
    print(f"✓ Classification report saved to: {report_path}")
    print(f"{'='*80}\n")
    
    return {
        'confusion_matrix': cm,
        'accuracy': overall_acc,
        'per_class_metrics': per_class_metrics,
        'top_misclassifications': top_misclassifications,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'class_names': class_names
    }


def identify_confused_pairs(confusion_matrix_result: Dict) -> List[Tuple[str, str, float]]:
    """
    Identify emotion pairs that are most commonly confused.
    
    Args:
        confusion_matrix_result: Output from generate_confusion_matrix()
    
    Returns:
        List of (emotion_A, emotion_B, confusion_rate) tuples
    """
    pairs = []
    for mis in confusion_matrix_result['top_misclassifications']:
        pairs.append((
            mis['true'],
            mis['predicted'],
            mis['percentage']
        ))
    
    return pairs
