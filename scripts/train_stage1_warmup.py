#!/usr/bin/env python3
"""
Stage 1: Warmup Training - Classification Head Stabilization
=====================================================

Objective:
    Train classification head on frozen ImageNet features to establish 
    stable task-specific representations.

Configuration:
    - Freeze: 100% backbone (all features)
    - Trainable: Classification head only (~9k params)
    - Optimizer: Adam(lr=1e-4, weight_decay=1e-5)
    - LR Scheduler: LinearLR warmup (1e-6 → 1e-4 over 3 epochs)
    - Epochs: 20
    - Early Stopping: Disabled (let head stabilize fully)

Success Criteria:
    - Validation accuracy: 40-42%
    - Training loss: Smooth decrease to ~1.5
    - Initial loss: ~1.946 (expected for 7 classes)

Output:
    - models/emotion_stage1_warmup.pth
    - logs/emotion_stage1_training.csv
"""

import sys
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import Config
from src.data.data_pipeline import create_dataloaders, calculate_class_weights
from src.models.vgg16_emotion import build_emotion_model
from src.training.trainer import train_one_epoch, validate
from src.training.utils import MetricTracker


def main():
    parser = argparse.ArgumentParser(description='Stage 1: Warmup Training')
    parser.add_argument('--data-dir', type=Path, default=Config.DATA_DIR,
                        help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=Config.STAGE1_EPOCHS,
                        help=f'Number of epochs (default: {Config.STAGE1_EPOCHS})')
    parser.add_argument('--lr', type=float, default=Config.STAGE1_LR,
                        help=f'Learning rate (default: {Config.STAGE1_LR})')
    parser.add_argument('--batch-size', type=int, default=Config.DATA_BATCH_SIZE,
                        help=f'Batch size (default: {Config.DATA_BATCH_SIZE})')
    parser.add_argument('--weight-decay', type=float, default=Config.STAGE1_WEIGHT_DECAY,
                        help=f'Weight decay (default: {Config.STAGE1_WEIGHT_DECAY})')
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                        help='Label smoothing factor (0.0-0.2, default: 0.0)')
    args = parser.parse_args()
    
    print("=" * 80)
    print("STAGE 1: WARMUP TRAINING - CLASSIFICATION HEAD STABILIZATION")
    print("=" * 80)
    
    # Device setup - REQUIRE CUDA
    if not torch.cuda.is_available():
        print("\n✗ ERROR: CUDA is not available!")
        print("  This training script requires a GPU.")
        print("  Please run on a machine with CUDA support.")
        sys.exit(1)
    
    device = torch.device('cuda')
    print(f"\n✓ Using device: {device}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load data
    print(f"\n{'='*80}")
    print("LOADING DATA")
    print("="*80)
    
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=4
    )
    
    # Calculate base class weights (Effective Number method)
    # These will be used as reference for adaptive weighting in later stages
    base_weights = calculate_class_weights(args.data_dir / 'train')
    current_weights = base_weights.clone()  # Stage 1 uses base weights directly
    
    print(f"\n✓ Data loaded successfully")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    
    # Build model (features frozen by default)
    print(f"\n{'='*80}")
    print("BUILDING MODEL")
    print("="*80)
    
    model = build_emotion_model(num_classes=7, pretrained=True, verbose=True)
    model = model.to(device)
    
    # Verify all features are frozen
    frozen_count = sum(1 for p in model.features.parameters() if not p.requires_grad)
    total_feature_params = len(list(model.features.parameters()))
    
    print(f"\n✓ Stage 1 freeze verification:")
    print(f"  Frozen feature layers: {frozen_count}/{total_feature_params}")
    
    if frozen_count != total_feature_params:
        print("  ⚠ WARNING: Not all features are frozen!")
        sys.exit(1)
    
    # Loss function with class weights
    print(f"\n{'='*80}")
    print("SETTING UP TRAINING")
    print("="*80)
    
    criterion = nn.CrossEntropyLoss(
        weight=current_weights.to(device),
        label_smoothing=args.label_smoothing
    )
    
    # Optimizer (only classifier parameters)
    optimizer = optim.Adam(
        model.classifier.parameters(),  # Only train classifier
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # LR Warmup Scheduler
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=Config.STAGE1_WARMUP_START_FACTOR,  # Start at 1% of base LR
        total_iters=Config.STAGE1_WARMUP_EPOCHS          # Reach base LR at epoch 3
    )
    
    print(f"\n✓ Optimizer: Adam")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Parameters: Classifier only ({sum(p.numel() for p in model.classifier.parameters()):,})")
    
    print(f"\n✓ LR Warmup Scheduler: LinearLR")
    print(f"  Warmup epochs: {Config.STAGE1_WARMUP_EPOCHS}")
    print(f"  Start LR: {args.lr * Config.STAGE1_WARMUP_START_FACTOR:.2e} ({Config.STAGE1_WARMUP_START_FACTOR*100:.0f}% of base)")
    print(f"  End LR: {args.lr:.2e} (base)")
    
    print(f"\n✓ Loss: CrossEntropyLoss with Effective Number class weights")
    if args.label_smoothing > 0:
        print(f"  Label smoothing: {args.label_smoothing} (reduces overconfidence)")
    else:
        print(f"  Label smoothing: disabled")
    
    # Metric tracking
    tracker = MetricTracker()
    
    # Training loop
    print(f"\n{'='*80}")
    print("TRAINING")
    print("="*80)
    
    best_val_acc = 0.0
    best_epoch = 0
    
    emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    
    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{args.epochs} | LR: {current_lr:.2e}")
        print("="*80)
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc, per_class_acc = validate(
            model, val_loader, criterion, device, desc='Validation'
        )
        
        # Update LR (warmup for first epochs)
        if epoch <= Config.STAGE1_WARMUP_EPOCHS:
            warmup_scheduler.step()
        
        # Track metrics
        tracker.update(epoch, train_loss, train_acc, val_loss, val_acc, current_lr)
        
        # Print results
        print(f"\n{'='*80}")
        print(f"Epoch {epoch} Results:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"\nPer-Class Accuracy:")
        for class_idx, acc in sorted(per_class_acc.items()):
            print(f"  {emotion_classes[class_idx]:8s}: {acc:5.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'per_class_accuracy': per_class_acc,  # For adaptive weighting
                'base_weights': base_weights.cpu(),   # Original Effective Number weights
                'current_weights': current_weights.cpu(),  # Weights used (same as base in Stage 1)
                'stage': 'warmup'
            }
            
            save_path = Config.STAGE1_CHECKPOINT
            torch.save(checkpoint, save_path)
            print(f"\n✓ Best model saved (Val Acc: {val_acc:.2f}%)")
    
    # Save training history
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print("="*80)
    
    log_path = Config.STAGE1_LOG
    tracker.save_to_csv(log_path)
    
    print(f"\n✓ Training completed successfully!")
    print(f"  Best validation accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"  Model saved to: {Config.STAGE1_CHECKPOINT}")
    print(f"  Training log saved to: {Config.STAGE1_LOG}")
    
    print(f"\n{'='*80}")
    print("STAGE 1 SUCCESS CRITERIA CHECK")
    print("="*80)
    
    target_min = Config.STAGE1_TARGET_VAL_ACC_MIN
    target_max = Config.STAGE1_TARGET_VAL_ACC_MAX
    
    if target_min <= best_val_acc <= target_max:
        print(f"✓ PASSED: Val acc {best_val_acc:.2f}% within target range [{target_min}%, {target_max}%]")
    elif best_val_acc < target_min:
        print(f"⚠ BELOW TARGET: Val acc {best_val_acc:.2f}% < {target_min}%")
        print(f"  Consider training longer or adjusting hyperparameters")
    else:
        print(f"✓ EXCEEDED: Val acc {best_val_acc:.2f}% > {target_max}%")
    
    print(f"\nNext step: python scripts/train_stage2_progressive.py")
    print("="*80)


if __name__ == '__main__':
    main()
