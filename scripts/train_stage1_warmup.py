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

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import Config
from src.data.data_pipeline import create_dataloaders, calculate_class_weights
from src.models.vgg16_emotion import build_emotion_model
from src.training.trainer import train_one_epoch, validate
from src.training.utils import MetricTracker
from src.training.optimizer import create_optimizer, get_warmup_scheduler, print_optimizer_info, apply_gradient_clipping


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
                        help='Label smoothing factor (0.0-0.2, default: 0.0) from Phase 2 optimization')
    parser.add_argument('--use-optimized-weights', action='store_true',
                        help='Use Phase 2 optimized class weights as base instead of Effective Number weights')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Custom output directory for model checkpoint and logs (for grid search)')
    parser.add_argument('--dropout', type=float, default=Config.CLASSIFIER_DROPOUT,
                        help=f'Dropout rate for classifier (default: {Config.CLASSIFIER_DROPOUT})')
    parser.add_argument('--preprocess', action='store_true',
                        help='Enable preprocessing (Unsharp Mask + CLAHE) for +4-5%% expected gain')
    parser.add_argument('--no-preprocess', action='store_true',
                        help='Explicitly disable preprocessing (overrides config)')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['adam', 'sgd'],
                        help='Optimizer: adam (baseline) or sgd+nesterov (recommended, +2-3%% gain)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9, ignored for Adam)')
    args = parser.parse_args()
    
    # Determine output paths (custom or default)
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = args.output_dir / 'emotion_stage1_warmup.pth'
        log_path = args.output_dir / 'emotion_stage1_training.csv'
    else:
        checkpoint_path = Config.STAGE1_CHECKPOINT
        log_path = Config.STAGE1_LOG
    
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
    
    # Determine preprocessing settings
    apply_preprocessing = False
    if args.preprocess:
        apply_preprocessing = True
        print("\n✓ Preprocessing ENABLED via --preprocess flag")
    elif args.no_preprocess:
        apply_preprocessing = False
        print("\n✓ Preprocessing DISABLED via --no-preprocess flag")
    elif Config.PREPROCESSING_ENABLED:
        apply_preprocessing = True
        print("\n✓ Preprocessing ENABLED via config")
    else:
        print("\n✓ Preprocessing DISABLED (default)")
    
    if apply_preprocessing:
        preprocess_config = Config.get_preprocessing_config()
        print(f"  Expected gain: +4-5% accuracy (+3-4% in Stage 1)")
    else:
        preprocess_config = None
    
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=4,
        apply_preprocessing=apply_preprocessing,
        preprocess_config=preprocess_config
    )
    
    # Calculate base class weights
    # Option 1: Effective Number method (default)
    # Option 2: Phase 2 optimized weights (if --use-optimized-weights flag set)
    base_weights_en = calculate_class_weights(args.data_dir / 'train')
    
    # Check if Phase 2 optimized weights should be used as base
    optimized_weights_path = Path('configs/class_weights_moderate.pth')
    
    if args.use_optimized_weights:
        print(f"\n{'='*80}")
        print("LOADING PHASE 2 OPTIMIZED WEIGHTS AS BASE")
        print("="*80)
        
        if not optimized_weights_path.exists():
            print(f"\n✗ ERROR: Phase 2 optimized weights not found: {optimized_weights_path}")
            print(f"  Run Phase 2 optimization first: python scripts/run_phase2_optimization.py --components 1")
            print(f"  Falling back to Effective Number weights")
            base_weights = base_weights_en
        else:
            optimized_checkpoint = torch.load(optimized_weights_path, map_location=device)
            base_weights = optimized_checkpoint['weights'].clone()
            
            print(f"\n✓ Loaded Phase 2 optimized class weights: {optimized_weights_path}")
            print(f"  Strategy: {optimized_checkpoint.get('strategy', 'moderate')}")
            print(f"  These weights will cascade through Stage 2 and Stage 3")
            
            # Print comparison
            emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
            print(f"\nWeight Comparison (EN Baseline vs Phase 2 Optimized):")
            print(f"{'Emotion':<12} {'EN Weight':<12} {'Phase 2 Weight':<15} {'Change':<10}")
            print(f"{'-'*55}")
            for idx, class_name in enumerate(emotion_classes):
                en = base_weights_en[idx].item()
                opt = base_weights[idx].item()
                change = ((opt - en) / en) * 100 if en > 0 else 0
                change_str = f"+{change:.1f}%" if change >= 0 else f"{change:.1f}%"
                print(f"{class_name:<12} {en:>8.4f}    {opt:>8.4f}         {change_str:>8}")
            print("="*80)
    else:
        base_weights = base_weights_en
        print(f"\n✓ Using Effective Number base weights")
    
    print(f"\n✓ Data loaded successfully")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    
    # Build model (features frozen by default)
    print(f"\n{'='*80}")
    print("BUILDING MODEL")
    print("="*80)
    
    model = build_emotion_model(num_classes=7, pretrained=True, dropout=args.dropout, verbose=True)
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
        weight=base_weights.to(device),
        label_smoothing=args.label_smoothing
    )
    
    # Optimizer setup (SGD+Nesterov recommended, Adam for baseline)
    if args.optimizer.lower() == 'sgd':
        # SGD with Nesterov momentum (recommended for vision tasks)
        # Stage 1: Higher LR (0.01) because training from scratch
        optimizer = create_optimizer(
            model.classifier,
            optimizer_type='sgd',
            stage=1,
            lr=args.lr if args.lr != Config.STAGE1_LR else None,  # Use custom LR if provided
            momentum=args.momentum,
            nesterov=True
        )
        
        # Create warmup scheduler (linear warmup from 0.001 to 0.01 over 3 epochs)
        warmup_scheduler = get_warmup_scheduler(optimizer, warmup_epochs=Config.STAGE1_WARMUP_EPOCHS)
        scheduler = warmup_scheduler
        scheduler_type = "LinearLR (warmup)"
        
    else:
        # Adam optimizer (baseline comparison)
        optimizer = create_optimizer(
            model.classifier,
            optimizer_type='adam',
            stage=1,
            lr=args.lr if args.lr != Config.STAGE1_LR else None
        )
        scheduler = None
        scheduler_type = "None (Adam typically doesn't need warmup)"
    
    print_optimizer_info(optimizer, stage=1)
    print(f"\n✓ Scheduler: {scheduler_type}")
    
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
        
        # Update LR scheduler if available (only for SGD warmup or other schedulers)
        if scheduler is not None:
            scheduler.step()
        
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
            
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'optimizer_type': args.optimizer.lower(),  # 'sgd' or 'adam' for stage continuity
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'per_class_accuracy': per_class_acc,  # For adaptive weighting in Stage 2
                'base_weights': base_weights.cpu(),   # Original Effective Number weights
                'dropout': args.dropout,  # Track dropout for stage continuity
                'stage': 'warmup'
            }
            
            torch.save(checkpoint_data, checkpoint_path)
            print(f"\n✓ Best model saved (Val Acc: {val_acc:.2f}%)")
    
    # Save training history
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print("="*80)
    
    tracker.save_to_csv(log_path)
    
    print(f"\n✓ Training completed successfully!")
    print(f"  Best validation accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"  Model saved to: {checkpoint_path}")
    print(f"  Training log saved to: {log_path}")
    
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
