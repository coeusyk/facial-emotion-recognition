"""
Comprehensive Classification Metrics Reporter
==============================================

Computes per-class and aggregate metrics (precision, recall, F1, MCC)
with special handling for class imbalance.
"""

import torch
import numpy as np
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List
from sklearn.metrics import (
    classification_report,
    matthews_corrcoef,
    cohen_kappa_score,
    balanced_accuracy_score
)


def compute_classification_metrics(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: List[str],
    output_dir: Path = Path("results/evaluation")
) -> Dict:
    """
    Compute comprehensive classification metrics.
    
    Args:
        model: Trained emotion recognition model
        dataloader: Test/validation dataloader
        device: Device to run on
        class_names: List of emotion class names
        output_dir: Directory to save outputs
    
    Returns:
        Dict with metrics and recommendations
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    print(f"\n{'='*80}")
    print("COMPUTING CLASSIFICATION METRICS")
    print(f"{'='*80}")
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Generate classification report
    report_dict = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    # Compute additional metrics
    mcc = matthews_corrcoef(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    
    # Print results
    print("\nPer-Class Metrics:")
    print(f"{'='*80}")
    print(f"{'Emotion':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    print(f"{'-'*80}")
    
    # Identify underperforming classes
    underperforming = []
    for class_name in class_names:
        metrics = report_dict[class_name]
        f1 = metrics['f1-score']
        
        print(f"{class_name:<12} {metrics['precision']:>10.4f}   "
              f"{metrics['recall']:>10.4f}   {f1:>10.4f}   "
              f"{int(metrics['support']):>8}")
        
        if f1 < 0.30:
            underperforming.append((class_name, f1))
    
    # Print averages
    print(f"{'-'*80}")
    print(f"{'Macro Avg':<12} {report_dict['macro avg']['precision']:>10.4f}   "
          f"{report_dict['macro avg']['recall']:>10.4f}   "
          f"{report_dict['macro avg']['f1-score']:>10.4f}")
    print(f"{'Weighted Avg':<12} {report_dict['weighted avg']['precision']:>10.4f}   "
          f"{report_dict['weighted avg']['recall']:>10.4f}   "
          f"{report_dict['weighted avg']['f1-score']:>10.4f}")
    
    print(f"\n{'='*80}")
    print("AGGREGATE METRICS")
    print(f"{'='*80}")
    print(f"Accuracy: {report_dict['accuracy']:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Matthews Correlation Coeff: {mcc:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    
    # Analyze class balance issues
    if underperforming:
        print(f"\n{'='*80}")
        print("⚠ CRITICALLY UNDERPERFORMING CLASSES (F1 < 0.30)")
        print(f"{'='*80}")
        for class_name, f1_score in sorted(underperforming, key=lambda x: x[1]):
            print(f"  {class_name}: F1 = {f1_score:.4f}")
            
            metrics = report_dict[class_name]
            precision = metrics['precision']
            recall = metrics['recall']
            
            if precision < recall:
                print(f"    → False positives issue: model predicts {class_name} too often")
                print(f"      Consider increasing class weight or adjusting threshold")
            elif recall < precision:
                print(f"    → False negatives issue: model misses {class_name} instances")
                print(f"      Consider increasing class weight to boost recall")
    
    # Save results as JSON
    json_path = output_dir / "classification_metrics.json"
    with open(json_path, 'w') as f:
        json.dump({
            'report': report_dict,
            'mcc': float(mcc),
            'kappa': float(kappa),
            'balanced_accuracy': float(balanced_acc),
            'underperforming_classes': underperforming
        }, f, indent=2)
    print(f"\n✓ Metrics saved to: {json_path}")
    
    # Save as CSV for easier comparison
    metrics_list = []
    for class_name in class_names:
        m = report_dict[class_name]
        metrics_list.append({
            'class': class_name,
            'precision': m['precision'],
            'recall': m['recall'],
            'f1_score': m['f1-score'],
            'support': int(m['support'])
        })
    
    # Add aggregate rows
    metrics_list.append({
        'class': 'macro_avg',
        'precision': report_dict['macro avg']['precision'],
        'recall': report_dict['macro avg']['recall'],
        'f1_score': report_dict['macro avg']['f1-score'],
        'support': int(sum(report_dict[cn]['support'] for cn in class_names))
    })
    
    metrics_df = pd.DataFrame(metrics_list)
    csv_path = output_dir / "classification_metrics.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"✓ CSV metrics saved to: {csv_path}")
    
    print(f"{'='*80}\n")
    
    return {
        'report': report_dict,
        'mcc': mcc,
        'kappa': kappa,
        'balanced_accuracy': balanced_acc,
        'underperforming_classes': underperforming,
        'all_preds': all_preds,
        'all_labels': all_labels
    }


def analyze_class_balance_issues(metrics_result: Dict, class_names: List[str]) -> List[Dict]:
    """
    Analyze precision vs recall imbalances in underperforming classes.
    
    Returns:
        List of recommendations
    """
    recommendations = []
    report = metrics_result['report']
    
    for class_name in class_names:
        m = report[class_name]
        precision = m['precision']
        recall = m['recall']
        f1 = m['f1-score']
        
        if f1 < 0.35:  # Struggling class
            gap = abs(precision - recall)
            
            if gap > 0.15:  # Significant imbalance
                if precision < recall:
                    recommendations.append({
                        'class': class_name,
                        'issue': 'high_false_positives',
                        'precision': precision,
                        'recall': recall,
                        'action': 'Increase threshold or reduce class weight'
                    })
                else:
                    recommendations.append({
                        'class': class_name,
                        'issue': 'high_false_negatives',
                        'precision': precision,
                        'recall': recall,
                        'action': 'Decrease threshold or increase class weight'
                    })
    
    return recommendations
