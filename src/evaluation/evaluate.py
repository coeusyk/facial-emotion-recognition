"""
Part 5.1: Model Evaluation & Analysis
Comprehensive evaluation with confusion matrix, classification report, and metrics.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    accuracy_score, precision_recall_fscore_support
)

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.vgg16_emotion import build_emotion_model
from src.data.data_pipeline import create_dataloaders


def evaluate_model(model_path, test_loader, class_names, device):
    """
    Comprehensive model evaluation.
    
    Args:
        model_path (str): Path to model checkpoint
        test_loader (DataLoader): Test data loader
        class_names (list): List of class names
        device: Device to evaluate on
        
    Returns:
        dict: Evaluation metrics
    """
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # Build model architecture
    num_classes = len(class_names)
    model = build_emotion_model(num_classes=num_classes, pretrained=False, verbose=False)
    
    # Load model weights
    print(f"\nLoading model from: {model_path}")
    
    if model_path.endswith('.pth'):
        # Check if it's a checkpoint or just weights
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            if 'val_acc' in checkpoint:
                print(f"  Validation accuracy: {checkpoint['val_acc']:.2f}%")
        else:
            model.load_state_dict(checkpoint)
            print(f"✓ Loaded model weights")
    
    model = model.to(device)
    model.eval()
    
    # Initialize lists for predictions and labels
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    print(f"\n{'='*60}")
    print("GENERATING PREDICTIONS")
    print(f"{'='*60}")
    
    # Evaluate on test set
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Get predictions
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            # Store results
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    print(f"✓ Generated predictions for {len(all_labels)} samples")
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_predictions),
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities
    }
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names, save_path='results/confusion_matrix.png'):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save plot
    """
    print(f"\n{'='*60}")
    print("GENERATING CONFUSION MATRIX")
    print(f"{'='*60}")
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Counts
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0],
        cbar_kws={'label': 'Count'},
        linewidths=0.5,
        linecolor='gray'
    )
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold', pad=20)
    axes[0].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12, fontweight='bold')
    axes[0].tick_params(labelsize=10)
    
    # Plot 2: Percentages
    sns.heatmap(
        cm_percent, 
        annot=True, 
        fmt='.1f', 
        cmap='Greens',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1],
        cbar_kws={'label': 'Percentage (%)'},
        linewidths=0.5,
        linecolor='gray'
    )
    axes[1].set_title('Confusion Matrix (Percentages)', fontsize=14, fontweight='bold', pad=20)
    axes[1].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=12, fontweight='bold')
    axes[1].tick_params(labelsize=10)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Confusion matrix saved to: {save_path}")
    
    return cm, cm_percent


def print_classification_report(y_true, y_pred, class_names, save_path='results/classification_report.txt'):
    """
    Print and save classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save report
    """
    print(f"\n{'='*60}")
    print("CLASSIFICATION REPORT")
    print(f"{'='*60}")
    
    # Generate report
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=class_names,
        digits=4
    )
    
    print(report)
    
    # Save report
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write("EMOTION RECOGNITION - CLASSIFICATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
    
    print(f"\n✓ Classification report saved to: {save_path}")


def plot_per_class_metrics(y_true, y_pred, class_names, save_path='results/per_class_metrics.png'):
    """
    Plot per-class precision, recall, and F1-score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save plot
    """
    print(f"\n{'='*60}")
    print("PER-CLASS METRICS")
    print(f"{'='*60}")
    
    # Calculate metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    
    # Create DataFrame for easy plotting
    import pandas as pd
    df = pd.DataFrame({
        'Class': class_names,
        'Precision': precision * 100,
        'Recall': recall * 100,
        'F1-Score': f1 * 100,
        'Support': support
    })
    
    print(df.to_string(index=False))
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Precision
    axes[0, 0].barh(class_names, precision * 100, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Precision (%)', fontweight='bold')
    axes[0, 0].set_title('Precision by Class', fontweight='bold')
    axes[0, 0].set_xlim([0, 100])
    axes[0, 0].grid(axis='x', alpha=0.3)
    for i, v in enumerate(precision * 100):
        axes[0, 0].text(v + 1, i, f'{v:.1f}%', va='center')
    
    # Recall
    axes[0, 1].barh(class_names, recall * 100, color='lightgreen', edgecolor='black')
    axes[0, 1].set_xlabel('Recall (%)', fontweight='bold')
    axes[0, 1].set_title('Recall by Class', fontweight='bold')
    axes[0, 1].set_xlim([0, 100])
    axes[0, 1].grid(axis='x', alpha=0.3)
    for i, v in enumerate(recall * 100):
        axes[0, 1].text(v + 1, i, f'{v:.1f}%', va='center')
    
    # F1-Score
    axes[1, 0].barh(class_names, f1 * 100, color='lightcoral', edgecolor='black')
    axes[1, 0].set_xlabel('F1-Score (%)', fontweight='bold')
    axes[1, 0].set_title('F1-Score by Class', fontweight='bold')
    axes[1, 0].set_xlim([0, 100])
    axes[1, 0].grid(axis='x', alpha=0.3)
    for i, v in enumerate(f1 * 100):
        axes[1, 0].text(v + 1, i, f'{v:.1f}%', va='center')
    
    # Support
    axes[1, 1].barh(class_names, support, color='lightyellow', edgecolor='black')
    axes[1, 1].set_xlabel('Number of Samples', fontweight='bold')
    axes[1, 1].set_title('Support by Class', fontweight='bold')
    axes[1, 1].grid(axis='x', alpha=0.3)
    for i, v in enumerate(support):
        axes[1, 1].text(v + 50, i, f'{v}', va='center')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Per-class metrics plot saved to: {save_path}")


def main():
    """Main evaluation function."""
    print("=" * 60)
    print("FACIAL EMOTION RECOGNITION - MODEL EVALUATION")
    print("=" * 60)
    
    # Configuration
    DATA_DIR = "data/raw"
    BATCH_SIZE = 64
    IMG_SIZE = 48
    NUM_WORKERS = 4
    
    # Model path - use Stage 2 best model if available, otherwise Stage 1
    MODELS_DIR = "models"
    MODEL_PATH = os.path.join(MODELS_DIR, "emotion_model_stage2_best.pth")
    
    if not os.path.exists(MODEL_PATH):
        MODEL_PATH = os.path.join(MODELS_DIR, "emotion_model_best.pth")
        print(f"\nStage 2 model not found, using Stage 1 model")
    
    if not os.path.exists(MODEL_PATH):
        print(f"\n✗ Error: No trained model found at {MODEL_PATH}")
        print("Please train the model first:")
        print("  Stage 1: python train_emotion_model.py")
        print("  Stage 2: python finetune_emotion_model.py")
        return
    
    print(f"\nUsing model: {MODEL_PATH}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    print(f"\n{'='*60}")
    print("LOADING TEST DATA")
    print(f"{'='*60}")
    
    _, _, test_loader, class_names = create_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        num_workers=NUM_WORKERS,
        val_split=0.0  # No validation split needed for evaluation
    )
    
    if test_loader is None:
        print("\n✗ Error: Test dataset not found")
        return
    
    # Evaluate model
    metrics = evaluate_model(MODEL_PATH, test_loader, class_names, device)
    
    # Print overall accuracy
    print(f"\n{'='*60}")
    print("OVERALL METRICS")
    print(f"{'='*60}")
    print(f"Test Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Total samples: {len(metrics['labels'])}")
    
    # Plot confusion matrix
    cm, cm_percent = plot_confusion_matrix(
        metrics['labels'], 
        metrics['predictions'], 
        class_names,
        save_path='results/confusion_matrix.png'
    )
    
    # Print classification report
    print_classification_report(
        metrics['labels'], 
        metrics['predictions'], 
        class_names,
        save_path='results/classification_report.txt'
    )
    
    # Plot per-class metrics
    plot_per_class_metrics(
        metrics['labels'], 
        metrics['predictions'], 
        class_names,
        save_path='results/per_class_metrics.png'
    )
    
    # Summary
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print("\nGenerated files:")
    print("  - results/confusion_matrix.png")
    print("  - results/classification_report.txt")
    print("  - results/per_class_metrics.png")
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
