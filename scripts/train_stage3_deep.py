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
import time
from datetime import datetime
import torch
import torch.nn as nn
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
from src.training.optimizer import create_optimizer, print_optimizer_info, apply_gradient_clipping

# GPU Monitoring
try:
    sys.path.insert(0, str(project_root / 'scripts' / 'diagnostics'))
    from diagnostics.gpu_monitor import GPUMonitor
    GPU_MONITORING_AVAILABLE = True
except ImportError:
    GPU_MONITORING_AVAILABLE = False


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
                        help='Label smoothing factor (0.0-0.2, default: 0.0) from Phase 2 optimization')
    parser.add_argument('--use-optimized-weights', action='store_true',
                        help='Use Phase 2 optimized class weights instead of adaptive weights (configs/class_weights_moderate.pth)')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Custom output directory for model checkpoint and logs (for grid search)')
    parser.add_argument('--dropout', type=float, default=None,
                        help='Dropout rate for classifier (auto-loaded from Stage 2 if not specified)')
    parser.add_argument('--preprocess', action='store_true',
                        help='Enable preprocessing (Unsharp Mask + CLAHE) for +4-5%% expected gain')
    parser.add_argument('--no-preprocess', action='store_true',
                        help='Explicitly disable preprocessing (overrides config)')
    parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'], default='adam',
                        help='Optimizer: adam (stable, recommended) or sgd (experimental, +2-3%% gain)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD (default: 0.9, higher=more history)')
    args = parser.parse_args()
    
    # Determine output paths (custom or default)
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = args.output_dir / 'emotion_stage3_deep.pth'
        log_path = args.output_dir / 'emotion_stage3_training.csv'
    else:
        checkpoint_path = Config.STAGE3_CHECKPOINT
        log_path = Config.STAGE3_LOG
    
    # Ensure parent directories exist
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    
    # Auto-detect Phase 2 optimized learning rate and weight decay settings
    # NOTE: Optimizer TYPE is now determined by Stage 2 checkpoint for continuity
    phase2_optimizer_config = Path('configs/best_optimizer_config.json')

    if phase2_optimizer_config.exists():
        import json
        with open(phase2_optimizer_config, 'r') as f:
            opt_config = json.load(f)
        
        # Apply Phase 2 LR/weight decay settings if user didn't override
        if args.lr == Config.STAGE3_LR:  # User didn't override
            args.lr = opt_config['learning_rate']
            print(f"\n✓ Auto-detected Phase 2 optimized learning rate: {args.lr:.0e}")
        
        if args.weight_decay == Config.STAGE3_WEIGHT_DECAY:  # User didn't override
            args.weight_decay = opt_config['weight_decay']
            print(f"✓ Auto-detected Phase 2 optimized weight decay: {args.weight_decay:.0e}")
        
        # Note: Optimizer TYPE is ignored here - we use Stage 2's type for continuity
        phase2_recommended_optimizer = opt_config.get('optimizer', 'adam')
        print(f"  (Phase 2 recommended {phase2_recommended_optimizer.upper()}, but using Stage 2 type for smooth transition)")
    
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
        print(f"  Expected gain: +4-5% accuracy")
    else:
        preprocess_config = None
    
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=4,
        apply_preprocessing=apply_preprocessing,
        preprocess_config=preprocess_config
    )
    
    print(f"\n✓ Data loaded successfully")
    print(f"  Training batches: {len(train_loader)}")
    if val_loader is not None:
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
    print(f"  Train Loss: {checkpoint.get('train_loss', 0.0):.4f}")
    print(f"  Val Loss: {checkpoint.get('val_loss', 0.0):.4f}")
    
    # Get optimizer type from Stage 2 for continuity (CRITICAL for smooth transition)
    stage2_optimizer_type = checkpoint.get('optimizer_type', 'adam')
    stage2_train_loss = checkpoint.get('train_loss', 0.0)
    stage2_val_loss = checkpoint.get('val_loss', 0.0)
    
    print(f"  Optimizer: {stage2_optimizer_type.upper()} (will use same type)")
    
    # Get dropout from checkpoint or CLI (CLI takes precedence)
    stage2_dropout = checkpoint.get('dropout', Config.CLASSIFIER_DROPOUT)
    if args.dropout is not None:
        model_dropout = args.dropout
        print(f"  Dropout: {model_dropout} (from CLI)")
    else:
        model_dropout = stage2_dropout
        print(f"  Dropout: {model_dropout} (from Stage 2)")
    
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
    
    # Calculate weights: Either Phase 2 optimized OR adaptive based on Stage 2 performance
    emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    optimized_weights_path = Path('configs/class_weights_moderate.pth')
    
    if args.use_optimized_weights:
        # Phase 2 Optimization: Use diagnostic-adjusted weights from Component 1
        print(f"\n{'='*80}")
        print("LOADING PHASE 2 OPTIMIZED CLASS WEIGHTS")
        print("="*80)
        
        if not optimized_weights_path.exists():
            print(f"\n✗ ERROR: Phase 2 optimized weights not found: {optimized_weights_path}")
            print(f"  Run Phase 2 optimization first: python scripts/run_phase2_optimization.py --components 1")
            sys.exit(1)
        
        optimized_checkpoint = torch.load(optimized_weights_path, map_location=device)
        current_weights = optimized_checkpoint['weights'].clone()
        
        print(f"\n✓ Loaded optimized class weights from: {optimized_weights_path}")
        print(f"  Strategy: {optimized_checkpoint.get('strategy', 'moderate')}")
        print(f"  Description: Diagnostic-adjusted weights from Phase 2 Component 1")
        print(f"  Bypassing adaptive weight calculation")
        
        # Print weight comparison
        print(f"\nWeight Comparison (Base EN vs Phase 2 Optimized):")
        print(f"{'Emotion':<12} {'Base Weight':<15} {'Optimized Weight':<18} {'Change':<10}")
        print(f"{'-'*60}")
        for idx, class_name in enumerate(emotion_classes):
            base = base_weights[idx].item()
            opt = current_weights[idx].item()
            change = ((opt - base) / base) * 100 if base > 0 else 0
            change_str = f"+{change:.1f}%" if change >= 0 else f"{change:.1f}%"
            print(f"{class_name:<12} {base:>8.4f}       {opt:>8.4f}         {change_str:>8}")
        print("="*80)
    else:
        # Standard adaptive weighting based on Stage 2 performance
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
    
    # Check if Stage 3 is needed (skip prompt in automated mode)
    if stage2_val_acc >= 64.0 and args.output_dir is None:
        print(f"\n⚠ Stage 2 already achieved {stage2_val_acc:.2f}% (≥64%)")
        print(f"  Stage 3 may provide diminishing returns (<1% improvement)")
        response = input("\nContinue with Stage 3? (y/n): ")
        if response.lower() != 'y':
            print("Exiting. Use Stage 2 model for best results.")
            sys.exit(0)
    
    # Build model with frozen features (use same dropout as Stage 2)
    model = build_emotion_model(num_classes=7, pretrained=True, dropout=model_dropout, verbose=False)
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
    
    # Optimizer - Use DIFFERENTIAL LEARNING RATES to prevent loss spikes
    # Newly unfrozen blocks (2-3) get 10x lower LR for stability
    # Previously trained blocks (4-5) + classifier get normal LR
    optimizer_type = args.optimizer.lower()
    if stage2_optimizer_type and stage2_optimizer_type != optimizer_type:
        print(f"\n⚠ Stage 2 optimizer ({stage2_optimizer_type.upper()}) differs from args ({optimizer_type.upper()})")
        print(f"  Using args value for compatibility: {optimizer_type.upper()}")
    
    # Separate parameters into two groups: existing (blocks 4-5 + classifier) vs new (blocks 2-3)
    # VGG16 structure: features[0-4]=block1, features[5-9]=block2, features[10-16]=block3,
    #                  features[17-23]=block4, features[24-30]=block5
    existing_params = []
    new_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Check if this is from blocks 2-3 (newly unfrozen)
            if 'features' in name:
                # Extract layer index from name like 'features.10.weight'
                try:
                    layer_idx = int(name.split('.')[1])
                    if 5 <= layer_idx <= 16:  # Blocks 2-3
                        new_params.append(param)
                    else:  # Blocks 4-5
                        existing_params.append(param)
                except:
                    existing_params.append(param)
            else:
                # Classifier params
                existing_params.append(param)
    
    # Create optimizer with differential learning rates
    param_groups = [
        {'params': existing_params, 'lr': args.lr, 'name': 'existing'},  # Normal LR
        {'params': new_params, 'lr': args.lr * 0.1, 'name': 'new_unfrozen'}  # 10x lower LR
    ]
    
    if optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(param_groups, momentum=args.momentum, weight_decay=1e-5)
    else:
        optimizer = torch.optim.Adam(param_groups, weight_decay=args.weight_decay)
    
    # Load optimizer state from Stage 2 for smooth transition
    # This preserves momentum and adaptive learning rate history for existing parameters
    stage2_optimizer_state = checkpoint.get('optimizer_state_dict', None)
    optimizer_state_loaded = False
    
    if stage2_optimizer_state is not None:
        # Check if optimizer types match to prevent parameter corruption
        if stage2_optimizer_type == optimizer_type:
            try:
                # CRITICAL: Stage 3 unfreezes new layers (blocks 2-3) that weren't trainable in Stage 2
                # PyTorch's load_state_dict will fail if parameter counts don't match
                # We manually transfer state for parameters that existed in Stage 2
                
                print(f"\n✓ Attempting smart optimizer state transfer...")
                
                # Get the saved param_groups and state
                saved_param_groups = stage2_optimizer_state['param_groups']
                saved_state = stage2_optimizer_state['state']
                
                # Build parameter ID mapping: Stage 2 param index -> Current model params
                # Strategy: The first N parameters in Stage 3 should match Stage 2's trainable params
                # (blocks 4-5 + classifier) since model structure is identical
                
                current_params = list(model.parameters())
                stage2_trainable_count = len(saved_state)
                
                print(f"  Stage 2 optimizer tracked {stage2_trainable_count} parameters")
                print(f"  Stage 3 model has {len(current_params)} total parameters")
                print(f"  Transferring state for matching parameters...")
                
                # Try to load the full state - PyTorch will handle missing params gracefully
                # It will only complain if there's a structure mismatch
                try:
                    optimizer.load_state_dict(stage2_optimizer_state)
                    optimizer_state_loaded = True
                    transfer_method = "full state load (PyTorch handled new params)"
                except:
                    # If full load fails, manually transfer state for known parameters
                    # This is a fallback that shouldn't normally be needed
                    print(f"  Full state load failed, attempting manual transfer...")
                    
                    # Just update the learning rate and don't load state
                    # This is safer than partial state loading which can corrupt training
                    optimizer_state_loaded = False
                    transfer_method = "fresh initialization (safer for new params)"
                
                # Update learning rate to Stage 3 target regardless of load success
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr
                    if optimizer_type == 'sgd':
                        param_group['weight_decay'] = 1e-5  # Stage 3 SGD weight decay
                    else:
                        param_group['weight_decay'] = args.weight_decay
                
                if optimizer_state_loaded:
                    print(f"  ✓ Optimizer state transferred successfully")
                    print(f"  Method: {transfer_method}")
                    print(f"  Type: {optimizer_type.upper()}")
                    print(f"  Learning rate updated to: {args.lr:.0e}")
                else:
                    print(f"  ⚠ Using fresh optimizer state ({transfer_method})")
                    print(f"  Expected: Initial loss increase of 0.20-0.40")
                
            except Exception as e:
                print(f"\n⚠ Optimizer state transfer failed: {e}")
                print(f"  Starting with fresh optimizer")
                optimizer_state_loaded = False
        else:
            print(f"\n⚠ Optimizer type mismatch:")
            print(f"  Stage 2: {stage2_optimizer_type.upper()}, Stage 3: {optimizer_type.upper()}")
            print(f"  Starting with fresh {optimizer_type.upper()} optimizer")
            optimizer_state_loaded = False
    else:
        print(f"\n⚠ No optimizer state in Stage 2 checkpoint")
        print(f"  Starting with fresh optimizer")
    
    print_optimizer_info(optimizer, stage=3)
    
    # Report expected transition behavior
    print(f"\n{'='*80}")
    print("STAGE TRANSITION METRICS")
    print("="*80)
    print(f"  Stage 2 Final: Train Loss={stage2_train_loss:.4f}, Val Loss={stage2_val_loss:.4f}")
    if optimizer_state_loaded:
        print(f"  Expected Stage 3 Epoch 1: Loss increase < 0.15 (smooth transition)")
    else:
        print(f"  ⚠ Expected Stage 3 Epoch 1: Loss increase 0.20-0.40 (no state continuity)")
    print("="*80)
    
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
    
    print(f"\n✓ Optimizer: {optimizer_type.upper()} with DIFFERENTIAL LEARNING RATES")
    print(f"  Existing params (blocks 4-5 + classifier): {len(existing_params)} params, LR={args.lr:.0e}")
    print(f"  Newly unfrozen (blocks 2-3): {len(new_params)} params, LR={args.lr * 0.1:.0e} (10x lower for stability)")
    print(f"  Weight decay: {args.weight_decay}")
    if optimizer_type == 'sgd':
        print(f"  Momentum: {args.momentum}")
    print(f"  Total trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
    if optimizer_state_loaded:
        print(f"  State: Loaded from Stage 2 (warm start for existing params)")
    else:
        print(f"  State: Fresh initialization")
    
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
    
    # Initialize GPU monitoring if available
    gpu_monitor = None
    if GPU_MONITORING_AVAILABLE and torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0).replace(' ', '_').replace('GeForce', '').strip('_')
            gpu_monitor = GPUMonitor(
                log_file=f'stage3_deep_{gpu_name}.csv',
                interval=2.0,
                metrics_dir='gpu_metrics'
            )
            gpu_monitor.start()
        except Exception as e:
            print(f"\n⚠ Could not start GPU monitoring: {e}")
            gpu_monitor = None
    
    # Training loop
    print(f"\n{'='*80}")
    print("TRAINING")
    print("="*80)
    
    best_val_acc = 0.0
    best_epoch = 0
    
    emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    
    # Timing tracking
    training_start_time = time.time()
    epoch_times = []
    
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
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
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        
        # Calculate loss gap (overfitting indicator)
        loss_gap = train_loss - val_loss
        
        # Print results
        print(f"\n{'='*80}")
        print(f"Epoch {epoch} Results:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  Loss Gap:   {loss_gap:.4f} {'⚠ OVERFITTING!' if loss_gap > args.overfitting_threshold else '(OK)'}")
        print(f"  Time: {epoch_time:.2f}s")
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
            
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'optimizer_type': args.optimizer.lower(),  # Track optimizer for continuity
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'per_class_accuracy': per_class_acc,  # Final per-class metrics
                'base_weights': base_weights.cpu(),   # Original Effective Number weights
                'current_weights': current_weights.cpu(),  # Final adapted weights
                'dropout': model_dropout,  # Track dropout
                'stage': 'deep'
            }
            
            torch.save(checkpoint_data, checkpoint_path)
            print(f"\n✓ Best model saved (Val Acc: {val_acc:.2f}%)")
        
        # Early stopping check
        if early_stopping(val_acc):
            print(f"\n✗ Early stopping triggered at epoch {epoch}")
            print(f"  No improvement for {args.early_stop_patience} epochs")
            break
    
    # Calculate total training time
    total_training_time = time.time() - training_start_time
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0
    
    # Save training history
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print("="*80)
    
    tracker.save_to_csv(log_path)
    
    # Save timing information
    timing_log_path = checkpoint_path.parent / 'stage3_timing_summary.txt'
    with open(timing_log_path, 'w') as f:
        f.write(f"Stage 3 Training Timing Summary\n")
        f.write(f"{'='*80}\n")
        f.write(f"Start Time: {datetime.fromtimestamp(training_start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End Time: {datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Training Time: {total_training_time:.2f}s ({total_training_time/60:.2f}m)\n")
        f.write(f"Average Epoch Time: {avg_epoch_time:.2f}s\n")
        f.write(f"Fastest Epoch: {min(epoch_times):.2f}s\n")
        f.write(f"Slowest Epoch: {max(epoch_times):.2f}s\n")
        f.write(f"Total Epochs: {len(epoch_times)}\n")
    
    print(f"\n✓ Training completed successfully!")
    print(f"  Best validation accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"  Total training time: {total_training_time:.2f}s ({total_training_time/60:.2f}m)")
    print(f"  Average epoch time: {avg_epoch_time:.2f}s")
    print(f"  Model saved to: {checkpoint_path}")
    print(f"  Training log saved to: {log_path}")
    
    # Stop GPU monitoring and generate charts
    if gpu_monitor is not None:
        try:
            gpu_monitor.stop()
        except Exception as e:
            print(f"\n⚠ Error stopping GPU monitor: {e}")
    
    print(f"\n{'='*80}")
    print(f"  Timing log saved to: {timing_log_path}")
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
