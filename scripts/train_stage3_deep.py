#!/usr/bin/env python3
"""
Stage 3: Deep Fine-tuning - Full Backbone Refinement
======================================================

Objective:
    Unfreeze 90% of backbone (blocks 2-5, features[5-30]) for maximum task 
    adaptation. Use only if Stage 2 plateaus below 64%.

Configuration:
    - Load: Stage 2 checkpoint (models/emotion_stage2_progressive.pth)
    - Freeze: Block 1 (features[0-4]) - preserve low-level edges
    - Unfreeze: Blocks 2-5 (features[5-30])
    - Trainable: ~90% backbone + classifier (~120M params)
    - Optimizer: Adam(lr=5e-6, weight_decay=1e-4)
    - LR Scheduler: ReduceLROnPlateau(patience=3, factor=0.3)
    - Epochs: 10 (short to avoid overfitting)
    - Early Stopping: Patience=8

Success Criteria:
    - Validation accuracy: 64-67% (+2-3% over Stage 2)
    - Monitor overfitting: Stop if train/val gap > 0.20
    - Diminishing returns: If gain < 1%, stop and use Stage 2 model

Output:
    - models/emotion_stage3_deep.pth
    - logs/emotion_stage3_training.csv
"""

import sys
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import Config
from src.data.data_pipeline import create_dataloaders
from src.models.vgg16_emotion import build_emotion_model, unfreeze_vgg16_blocks
from src.training.trainer import train_one_epoch, validate
from src.training.utils import MetricTracker, EarlyStopping
from src.training.adaptive_weights import calculate_adaptive_weights, save_weight_history, print_weight_comparison


def main():
    parser = argparse.ArgumentParser(description='Stage 3: Deep Fine-tuning')
    parser.add_argument('--data-dir', type=Path, default=Config.DATA_DIR,
                        help='Path to dataset directory')
    parser.add_argument('--stage2-checkpoint', type=Path, 
                        default=Config.STAGE2_CHECKPOINT,
                        help='Path to Stage 2 checkpoint')
    parser.add_argument('--epochs', type=int, default=Config.STAGE3_EPOCHS,
                        help=f'Number of epochs (default: {Config.STAGE3_EPOCHS})')
    parser.add_argument('--lr', type=float, default=Config.STAGE3_LR,
                        help=f'Learning rate (default: {Config.STAGE3_LR})')
    parser.add_argument('--batch-size', type=int, default=Config.DATA_BATCH_SIZE,
                        help=f'Batch size (default: {Config.DATA_BATCH_SIZE})')
    parser.add_argument('--weight-decay', type=float, default=Config.STAGE3_WEIGHT_DECAY,
                        help=f'Weight decay (default: {Config.STAGE3_WEIGHT_DECAY})')
    parser.add_argument('--early-stop-patience', type=int, default=Config.STAGE3_EARLY_STOP_PATIENCE,
                        help=f'Early stopping patience (default: {Config.STAGE3_EARLY_STOP_PATIENCE})')
    parser.add_argument('--overfitting-threshold', type=float, default=Config.STAGE3_MAX_OVERFITTING_GAP,
                        help=f'Stop if train/val loss gap exceeds this (default: {Config.STAGE3_MAX_OVERFITTING_GAP})')
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                        help='Label smoothing factor (0.0-0.2, default: 0.0)')
    args = parser.parse_args()
    
    print("=" * 80)
    print("STAGE 3: DEEP FINE-TUNING - FULL BACKBONE REFINEMENT")
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
    
    print(f"\n✓ Data loaded successfully")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    
    # Build model and load Stage 2 checkpoint
    print(f"\n{'='*80}")
    print("LOADING STAGE 2 CHECKPOINT")
    print("="*80)
    
    stage2_path = args.stage2_checkpoint
    
    if not stage2_path.exists():
        print(f"\n✗ ERROR: Stage 2 checkpoint not found: {stage2_path}")
        print(f"  Please run: python scripts/train_stage2_progressive.py")
        sys.exit(1)
    
    checkpoint = torch.load(stage2_path, map_location=device)
    
    print(f"\n✓ Checkpoint loaded from: {stage2_path}")
    print(f"  Stage: {checkpoint.get('stage', 'unknown')}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Val Acc: {checkpoint.get('val_acc', 0.0):.2f}%")
    
    stage2_val_acc = checkpoint.get('val_acc', 0.0)
    
    # Extract adaptive weight data from Stage 2
    stage2_per_class_acc = checkpoint.get('per_class_accuracy', {})
    base_weights = checkpoint.get('base_weights', None)
    
    if base_weights is None:
        print(f"\n✗ ERROR: Stage 2 checkpoint missing 'base_weights'")
        print(f"  Please re-train Stage 2 with updated script")
        sys.exit(1)
    
    if not stage2_per_class_acc:
        print(f"\n✗ ERROR: Stage 2 checkpoint missing 'per_class_accuracy'")
        print(f"  Please re-train Stage 2 with updated script")
        sys.exit(1)
    
    # Calculate adaptive weights based on Stage 2 performance
    emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    current_weights, weight_metadata = calculate_adaptive_weights(
        per_class_accuracy=stage2_per_class_acc,
        base_weights=base_weights,
        class_names=emotion_classes,
        stage=2  # Stage 2→3 transition
    )
    
    # Save weight history
    save_weight_history(weight_metadata, stage=2)
    
    # Print comparison
    print_weight_comparison(base_weights, current_weights, emotion_classes)
    
    # Check if Stage 3 is needed
    if stage2_val_acc >= 64.0:
        print(f"\n⚠ Stage 2 already achieved {stage2_val_acc:.2f}% (≥64%)")
        print(f"  Stage 3 may provide diminishing returns (<1% improvement)")
        response = input("\nContinue with Stage 3? (y/n): ")
        if response.lower() != 'y':
            print("Exiting. Use Stage 2 model for best results.")
            sys.exit(0)
    
    # Build model with frozen features
    model = build_emotion_model(num_classes=7, pretrained=True, verbose=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"\n✓ Model weights loaded from Stage 2")
    
    # Unfreeze blocks 2-5 (keep block 1 frozen)
    print(f"\n{'='*80}")
    print("UNFREEZING BLOCKS FOR DEEP FINE-TUNING")
    print("="*80)
    
    model = unfreeze_vgg16_blocks(model, blocks_to_unfreeze=Config.STAGE3_UNFROZEN_BLOCKS, verbose=True)
    model = model.to(device)
    
    # Verify unfreezing
    unfrozen_count = sum(1 for p in model.features.parameters() if p.requires_grad)
    total_feature_params = len(list(model.features.parameters()))
    
    print(f"\n✓ Stage 3 unfreeze verification:")
    print(f"  Unfrozen feature layers: {unfrozen_count}/{total_feature_params}")
    print(f"  Frozen: Block 1 only (preserve low-level edges)")
    
    # Loss function with class weights
    print(f"\n{'='*80}")
    print("SETTING UP TRAINING")
    print("="*80)
    
    criterion = nn.CrossEntropyLoss(
        weight=current_weights.to(device),
        label_smoothing=args.label_smoothing
    )
    
    # Optimizer (all unfrozen parameters)
    optimizer = optim.Adam(
        model.parameters(),  # All trainable params (blocks 2-5 + classifier)
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # LR Scheduler: ReduceLROnPlateau (more aggressive)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=Config.STAGE3_SCHEDULER_MODE,
        factor=Config.STAGE3_SCHEDULER_FACTOR,  # More aggressive reduction
        patience=Config.STAGE3_SCHEDULER_PATIENCE  # Shorter patience
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=args.early_stop_patience,
        mode='max',
        verbose=True
    )
    
    print(f"\n✓ Optimizer: Adam")
    print(f"  Learning rate: {args.lr} (very low to prevent catastrophic forgetting)")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Parameters: Blocks 2-5 + Classifier ({sum(p.numel() for p in model.parameters() if p.requires_grad):,})")
    
    print(f"\n✓ LR Scheduler: ReduceLROnPlateau (aggressive)")
    print(f"  Mode: {Config.STAGE3_SCHEDULER_MODE} (reduce on val_loss)")
    print(f"  Patience: {Config.STAGE3_SCHEDULER_PATIENCE} epochs")
    print(f"  Factor: {Config.STAGE3_SCHEDULER_FACTOR} (reduce to {Config.STAGE3_SCHEDULER_FACTOR*100:.0f}%)")
    
    print(f"\n✓ Early Stopping: Enabled")
    print(f"  Patience: {args.early_stop_patience} epochs")
    print(f"  Mode: max (monitor val_acc)")
    
    print(f"\n✓ Overfitting Protection: Enabled")
    print(f"  Threshold: Train/val loss gap > {args.overfitting_threshold}")
    
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
        
        # Update LR scheduler (based on val_loss)
        scheduler.step(val_loss)
        
        # Track metrics
        tracker.update(epoch, train_loss, train_acc, val_loss, val_acc, current_lr)
        
        # Calculate loss gap (overfitting indicator)
        loss_gap = train_loss - val_loss
        
        # Print results
        print(f"\n{'='*80}")
        print(f"Epoch {epoch} Results:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  Loss Gap:   {loss_gap:.4f} {'⚠ OVERFITTING!' if loss_gap > args.overfitting_threshold else '(OK)'}")
        print(f"\nPer-Class Accuracy:")
        for class_idx, acc in sorted(per_class_acc.items()):
            print(f"  {emotion_classes[class_idx]:8s}: {acc:5.2f}%")
        
        # Overfitting check
        if loss_gap > args.overfitting_threshold:
            print(f"\n✗ OVERFITTING DETECTED!")
            print(f"  Train/val loss gap ({loss_gap:.4f}) exceeds threshold ({args.overfitting_threshold})")
            print(f"  Stopping training to prevent performance degradation")
            break
        
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
                'per_class_accuracy': per_class_acc,  # Final per-class metrics
                'base_weights': base_weights.cpu(),   # Original Effective Number weights
                'current_weights': current_weights.cpu(),  # Final adapted weights
                'stage': 'deep'
            }
            
            save_path = Config.STAGE3_CHECKPOINT
            torch.save(checkpoint, save_path)
            print(f"\n✓ Best model saved (Val Acc: {val_acc:.2f}%)")
        
        # Early stopping check
        if early_stopping(val_acc):
            print(f"\n✗ Early stopping triggered at epoch {epoch}")
            print(f"  No improvement for {args.early_stop_patience} epochs")
            break
    
    # Save training history
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print("="*80)
    
    log_path = Config.STAGE3_LOG
    tracker.save_to_csv(log_path)
    
    print(f"\n✓ Training completed successfully!")
    print(f"  Best validation accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"  Model saved to: {Config.STAGE3_CHECKPOINT}")
    print(f"  Training log saved to: {Config.STAGE3_LOG}")
    
    print(f"\n{'='*80}")
    print("STAGE 3 SUCCESS CRITERIA CHECK")
    print("="*80)
    
    target_min = Config.STAGE3_TARGET_VAL_ACC_MIN
    target_max = Config.STAGE3_TARGET_VAL_ACC_MAX
    improvement = best_val_acc - stage2_val_acc
    
    print(f"  Stage 2 Val Acc: {stage2_val_acc:.2f}%")
    print(f"  Stage 3 Val Acc: {best_val_acc:.2f}%")
    print(f"  Improvement: +{improvement:.2f}%")
    
    if target_min <= best_val_acc <= target_max:
        print(f"\n✓ PASSED: Val acc {best_val_acc:.2f}% within target range [{target_min}%, {target_max}%]")
    elif best_val_acc < target_min:
        print(f"\n⚠ BELOW TARGET: Val acc {best_val_acc:.2f}% < {target_min}%")
    else:
        print(f"\n✓ EXCEEDED: Val acc {best_val_acc:.2f}% > {target_max}%")
    
    if improvement >= Config.STAGE3_TARGET_IMPROVEMENT:
        print(f"✓ PASSED: Improvement {improvement:.2f}% >= {Config.STAGE3_TARGET_IMPROVEMENT}% target")
    elif improvement >= Config.STAGE3_MIN_IMPROVEMENT_THRESHOLD:
        print(f"⚠ Marginal: Improvement {improvement:.2f}% between {Config.STAGE3_MIN_IMPROVEMENT_THRESHOLD}-{Config.STAGE3_TARGET_IMPROVEMENT}%")
    else:
        print(f"✗ DIMINISHING RETURNS: Improvement {improvement:.2f}% < {Config.STAGE3_MIN_IMPROVEMENT_THRESHOLD}%")
        print(f"  Recommendation: Use Stage 2 model instead")
    
    print(f"\n{'='*80}")
    print("3-STAGE PROGRESSIVE TRAINING COMPLETE!")
    print("="*80)
    print("\nFinal Model Selection:")
    if improvement >= Config.STAGE3_MIN_IMPROVEMENT_THRESHOLD and best_val_acc >= stage2_val_acc:
        print(f"  ✓ Use Stage 3 model: {Config.STAGE3_CHECKPOINT}")
        print(f"    Val Acc: {best_val_acc:.2f}%")
    else:
        print(f"  ✓ Use Stage 2 model: {Config.STAGE2_CHECKPOINT}")
        print(f"    Val Acc: {stage2_val_acc:.2f}%")
        print(f"    (Stage 3 showed diminishing returns)")
    
    print("="*80)


if __name__ == '__main__':
    main()
