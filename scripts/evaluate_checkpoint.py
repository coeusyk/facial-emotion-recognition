#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Pipeline
========================================

Runs all diagnostic and evaluation tools on a trained checkpoint.
Generates confusion matrices, classification metrics, and ROC curves.
"""

import sys
from pathlib import Path
import argparse
import torch
import torch.nn as nn

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import Config
from src.data.data_pipeline import create_dataloaders
from src.models.vgg16_emotion import build_emotion_model
from src.evaluation.confusion_matrix import generate_confusion_matrix
from src.evaluation.classification_metrics import compute_classification_metrics, analyze_class_balance_issues
from src.evaluation.roc_curves import generate_roc_curves, suggest_threshold_adjustments


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Model Evaluation')
    parser.add_argument('--checkpoint', type=Path, default=Config.STAGE3_CHECKPOINT,
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=Path, default=Config.DATA_DIR,
                        help='Path to dataset directory')
    parser.add_argument('--output-dir', type=Path, default=Config.RESULTS_DIR / "evaluation",
                        help='Directory to save evaluation outputs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for evaluation')
    args = parser.parse_args()
    
    print("=" * 80)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 80)
    
    # Device setup
    if not torch.cuda.is_available():
        print("\n⚠ WARNING: CUDA not available, using CPU (slow)")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    
    print(f"\n✓ Device: {device}")
    
    # Load checkpoint
    if not args.checkpoint.exists():
        print(f"\n✗ ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    print(f"✓ Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Build model
    model = build_emotion_model(num_classes=7, pretrained=True, verbose=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Create dataloaders
    print("\n✓ Loading test data...")
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=4
    )
    
    emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("EVALUATION DIAGNOSTICS")
    print(f"{'='*80}")
    
    # Component 1: Confusion Matrix
    print("\n[1/3] Confusion Matrix Analysis")
    print("-" * 80)
    cm_result = generate_confusion_matrix(
        model, test_loader, device, emotion_classes, output_dir
    )
    
    # Component 2: Classification Metrics
    print("\n[2/3] Classification Metrics")
    print("-" * 80)
    metrics_result = compute_classification_metrics(
        model, test_loader, device, emotion_classes, output_dir
    )
    
    # Component 3: ROC Curves
    print("\n[3/3] ROC Curves and AUC Analysis")
    print("-" * 80)
    roc_result = generate_roc_curves(
        model, test_loader, device, emotion_classes, output_dir
    )
    
    # Generate recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS SUMMARY")
    print(f"{'='*80}")
    
    print("\n1. CLASS BALANCE ISSUES:")
    recommendations = analyze_class_balance_issues(metrics_result, emotion_classes)
    if recommendations:
        for rec in recommendations:
            print(f"   • {rec['class']}: {rec['action']}")
    else:
        print("   ✓ No critical class balance issues detected")
    
    print("\n2. POOR DISCRIMINATORS (AUC < 0.70):")
    if roc_result['poor_discriminators']:
        for class_name, auc in roc_result['poor_discriminators']:
            print(f"   ⚠ {class_name}: AUC = {auc:.4f}")
    else:
        print("   ✓ All classes have AUC >= 0.70")
    
    print("\n3. NEXT STEPS:")
    if cm_result['accuracy'] < 50:
        print("   • Model accuracy < 50%: Consider architectural changes or data quality review")
    elif cm_result['accuracy'] < 65:
        print("   • Apply more aggressive data augmentation")
        print("   • Use stratified cross-validation to validate improvements")
        print("   • Consider ensemble methods")
    else:
        print("   • Model shows acceptable performance")
        print("   • Focus on per-class improvements for underperforming emotions")
    
    print(f"\n{'='*80}")
    print(f"✓ Evaluation complete! Results saved to: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
