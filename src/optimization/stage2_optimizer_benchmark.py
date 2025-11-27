"""
Component 5B: Stage 2 Optimizer & Learning Rate Tuning
========================================================

Purpose:
    Compare Adam vs AdamW and find optimal Stage 2 learning rate.
    
Current Setup (from config.py):
    - Optimizer: Adam
    - Stage 2 LR: 1e-5
    - Weight decay: 1e-4

Hypotheses:
    - Current LR (1e-5) might not be optimal for Stage 2 dynamics
    - Stage 2 unfreezes blocks 4-5 (~40% trainable) vs Stage 3's ~90%
    - AdamW might handle weight decay better
    - Stage 2 might benefit from different LR than Stage 3

Test Matrix:
    - Adam with [5e-6, 1e-5, 2e-5, 5e-5]
    - AdamW with [5e-6, 1e-5, 2e-5, 5e-5]
    Total: 8 configurations

Expected Gain: +1-3% accuracy over default Stage 2 LR

Author: FER-2013 Optimization Pipeline
"""

from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import matplotlib.pyplot as plt
import json
from typing import Dict, Tuple
from copy import deepcopy


def run_stage2_optimizer_benchmark(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_epochs: int = 10,
    output_dir: Path = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Benchmark different optimizers and learning rates for Stage 2 training.
    
    Args:
        model: VGG16 emotion model (Stage 1 checkpoint loaded, blocks 4-5 unfrozen)
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function (with class weights)
        device: Device to train on
        num_epochs: Number of epochs per config (default: 10)
        output_dir: Where to save results
    
    Returns:
        Tuple of (DataFrame with benchmark results, best config dict)
    """
    print(f"""{'='*80}
COMPONENT 5B: STAGE 2 OPTIMIZER BENCHMARK
{'='*80}
Number of epochs per config: {num_epochs}""")
    print(f"Total configurations: 8")
    
    # Define optimizer configurations for Stage 2
    configs = [
        # Lower LR variants
        {'name': 'Adam_5e-6', 'optimizer': 'adam', 'lr': 5e-6, 'weight_decay': 1e-4},
        
        # Current baseline
        {'name': 'Adam_1e-5', 'optimizer': 'adam', 'lr': 1e-5, 'weight_decay': 1e-4},
        
        # Higher LR variants for Adam
        {'name': 'Adam_2e-5', 'optimizer': 'adam', 'lr': 2e-5, 'weight_decay': 1e-4},
        {'name': 'Adam_5e-5', 'optimizer': 'adam', 'lr': 5e-5, 'weight_decay': 1e-4},
        
        # AdamW variants (better weight decay handling)
        {'name': 'AdamW_5e-6', 'optimizer': 'adamw', 'lr': 5e-6, 'weight_decay': 5e-5},
        {'name': 'AdamW_1e-5', 'optimizer': 'adamw', 'lr': 1e-5, 'weight_decay': 5e-5},
        {'name': 'AdamW_2e-5', 'optimizer': 'adamw', 'lr': 2e-5, 'weight_decay': 5e-5},
        {'name': 'AdamW_5e-5', 'optimizer': 'adamw', 'lr': 5e-5, 'weight_decay': 5e-5},
    ]
    
    results = []
    best_config = None
    best_overall_acc = 0.0
    
    # Save initial model state
    initial_state = deepcopy(model.state_dict())
    
    for config_idx, config in enumerate(configs, 1):
        print(f"""\n{'='*80}
TESTING CONFIG {config_idx}/8: {config['name']}
{'='*80}
  Optimizer: {config['optimizer'].upper()}
  Learning rate: {config['lr']:.0e}
  Weight decay: {config['weight_decay']:.0e}""")
        
        # Reset model to initial state
        model.load_state_dict(deepcopy(initial_state))
        
        # Create optimizer
        if config['optimizer'] == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=config['lr'],
                weight_decay=config['weight_decay']
            )
        elif config['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config['lr'],
                weight_decay=config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {config['optimizer']}")
        
        # LR Scheduler
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Training loop
        epoch_results = []
        best_val_acc = 0.0
        best_epoch = 0
        
        for epoch in range(1, num_epochs + 1):
            # Train
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)
            
            train_loss /= len(train_loader)
            train_acc = 100.0 * train_correct / train_total
            
            # Validate
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)
            
            val_loss /= len(val_loader)
            val_acc = 100.0 * val_correct / val_total
            
            # Update scheduler
            scheduler.step(val_loss)
            
            epoch_results.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            })
            
            # Track best validation accuracy (like early stopping would)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
            
            best_marker = " *" if val_acc == best_val_acc else ""
            print(f"Epoch {epoch:2d}/{num_epochs} | "
                  f"Train: {train_loss:.4f}/{train_acc:5.2f}% | "
                  f"Val: {val_loss:.4f}/{val_acc:5.2f}%{best_marker}")
        
        # Get final metrics
        final_epoch = epoch_results[-1]
        
        # Check convergence (did it improve from epoch 1 to last epoch?)
        first_val_loss = epoch_results[0]['val_loss']
        final_val_loss = final_epoch['val_loss']
        converged = final_val_loss < first_val_loss - 0.05
        
        results.append({
            'config': config['name'],
            'optimizer': config['optimizer'],
            'learning_rate': config['lr'],
            'weight_decay': config['weight_decay'],
            'best_val_acc': best_val_acc,
            'best_epoch': best_epoch,
            'final_val_acc': final_epoch['val_acc'],
            'final_val_loss': final_val_loss,
            'final_train_loss': final_epoch['train_loss'],
            'final_train_acc': final_epoch['train_acc'],
            'converged': converged,
            'loss_improvement': first_val_loss - final_val_loss
        })
        
        # Track best config (use best val acc, not final)
        if best_val_acc > best_overall_acc:
            best_overall_acc = best_val_acc
            best_config = config
            best_config['best_val_acc'] = best_val_acc
            best_config['best_epoch'] = best_epoch
        
        print(f"""\nConfig {config['name']} Results:
  Best Val Acc: {best_val_acc:.2f}% (Epoch {best_epoch})
  Final Val Acc: {final_epoch['val_acc']:.2f}% (Epoch {num_epochs})
  Converged: {'Yes' if converged else 'No'}""")
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    
    # Display summary
    print(f"""\n{'='*80}
STAGE 2 OPTIMIZER BENCHMARK SUMMARY
{'='*80}

{df.to_string(index=False)}""")
    
    print(f"""\n{'='*80}
BEST CONFIGURATION FOR STAGE 2
{'='*80}
Config: {best_config['name']}
  Optimizer: {best_config['optimizer'].upper()}
  Learning rate: {best_config['lr']:.0e}
  Weight decay: {best_config['weight_decay']:.0e}
  Best Val Acc: {best_overall_acc:.2f}% (Epoch {best_config.get('best_epoch', 'N/A')})
  
Note: Rankings based on BEST validation accuracy, not final epoch.
This reflects real training with early stopping and best checkpoint saving.""")
    
    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save CSV
        csv_path = output_dir / 'stage2_optimizer_benchmark_results.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Results saved to: {csv_path}")
        
        # Save best config JSON
        best_config_path = Path('configs/best_stage2_optimizer_config.json')
        best_config_path.parent.mkdir(parents=True, exist_ok=True)
        
        best_config_data = {
            'optimizer': best_config['optimizer'],
            'learning_rate': best_config['lr'],
            'weight_decay': best_config['weight_decay'],
            'best_val_acc': best_overall_acc,
            'best_epoch': best_config.get('best_epoch', 0),
            'description': f'Best Stage 2 optimizer config from benchmark (Best Val Acc: {best_overall_acc:.2f}%)',
            'usage': 'Automatically applied to Stage 2 training unless overridden with --lr/--weight-decay'
        }
        
        with open(best_config_path, 'w') as f:
            json.dump(best_config_data, f, indent=2)
        
        print(f"✓ Best config saved to: {best_config_path}")
        
        # Generate report
        save_stage2_benchmark_report(df, best_config, best_overall_acc, output_dir)
    
    return df, best_config


def save_stage2_benchmark_report(df: pd.DataFrame, best_config: Dict, best_val_acc: float, output_dir: Path):
    """Generate detailed benchmark report for Stage 2."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / 'stage2_optimizer_benchmark_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("STAGE 2 OPTIMIZER & LEARNING RATE BENCHMARK REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Objective:\n")
        f.write("  Compare Adam vs AdamW with different learning rates for Stage 2 training.\n")
        f.write("  Current baseline: Adam with LR=1e-5\n\n")
        
        f.write("Hypothesis:\n")
        f.write("  - Stage 2 has different dynamics than Stage 3 (~40% vs ~90% trainable)\n")
        f.write("  - Current LR (1e-5) might not be optimal\n")
        f.write("  - AdamW might handle weight decay better\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("RESULTS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        # Sort by best_val_acc
        df_sorted = df.sort_values('best_val_acc', ascending=False)
        
        f.write(f"{'Rank':<6} | {'Config':<12} | {'Best Val Acc':>13} | {'Best Epoch':>10} | {'Converged':>10}\n")
        f.write("-" * 80 + "\n")
        
        for rank, (_, row) in enumerate(df_sorted.iterrows(), 1):
            marker = "***" if rank == 1 else ""
            f.write(f"{rank:<6} | {row['config']:<12} | {row['best_val_acc']:>11.2f}% | "
                   f"{row['best_epoch']:>10} | {str(row['converged']):>10} {marker}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("BEST CONFIGURATION FOR STAGE 2\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Config: {best_config['name']}\n")
        f.write(f"  Optimizer: {best_config['optimizer'].upper()}\n")
        f.write(f"  Learning Rate: {best_config['lr']:.0e}\n")
        f.write(f"  Weight Decay: {best_config['weight_decay']:.0e}\n\n")
        
        best_row = df[df['config'] == best_config['name']].iloc[0]
        f.write(f"Performance:\n")
        f.write(f"  Best Validation Accuracy: {best_row['best_val_acc']:.2f}%\n")
        f.write(f"  Best Epoch: {best_row['best_epoch']}\n")
        f.write(f"  Converged: {'Yes' if best_row['converged'] else 'No'}\n\n")
        
        # Comparison to baseline
        baseline_row = df[df['config'] == 'Adam_1e-5'].iloc[0]
        f.write("Improvement over baseline (Adam_1e-5):\n")
        f.write(f"  Best Val Acc: {baseline_row['best_val_acc']:.2f}% -> {best_row['best_val_acc']:.2f}% "
               f"({best_row['best_val_acc'] - baseline_row['best_val_acc']:+.2f}%)\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. Stage 2 training will automatically use these settings:\n")
        f.write(f"   python scripts/train_stage2_progressive.py\n")
        f.write(f"   (Auto-loads lr={best_config['lr']:.0e}, wd={best_config['weight_decay']:.0e})\n\n")
        
        f.write("2. To override auto-detection:\n")
        f.write(f"   python scripts/train_stage2_progressive.py --lr <custom_lr> --weight-decay <custom_wd>\n\n")
        
        if best_config['optimizer'] == 'adamw':
            f.write("3. Best optimizer is AdamW (better weight decay handling)\n\n")
        
        f.write("4. Expected improvements:\n")
        f.write(f"   - Better Stage 2 convergence\n")
        f.write(f"   - Improved Stage 2 checkpoint for Stage 3\n")
        f.write(f"   - Cascading benefits through training pipeline\n\n")
    
    print(f"✓ Detailed report saved to: {report_path}")


def main():
    """Example usage of Stage 2 optimizer benchmark."""
    from pathlib import Path
    import sys
    
    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from config import Config
    from src.data.data_pipeline import create_dataloaders, calculate_class_weights
    from src.models.vgg16_emotion import build_emotion_model, unfreeze_vgg16_blocks
    
    print(f"""{'='*80}
COMPONENT 5B: STAGE 2 OPTIMIZER BENCHMARK
{'='*80}""")
    
    # Configuration
    STAGE1_CHECKPOINT = Path('models/emotion_stage1_warmup.pth')
    DATA_DIR = Path('data/raw')
    OUTPUT_DIR = Path('results/optimization/stage2_optimizers')
    
    if not STAGE1_CHECKPOINT.exists():
        print(f"""\nError: Stage 1 checkpoint not found: {STAGE1_CHECKPOINT}
Please train Stage 1 first or update checkpoint path""")
        return
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, _, class_names = create_dataloaders(
        DATA_DIR,
        batch_size=64,
        num_workers=4
    )
    
    # Load model from Stage 1
    print("\nLoading Stage 1 model...")
    model = build_emotion_model(num_classes=7, pretrained=True, verbose=False)
    checkpoint = torch.load(STAGE1_CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Unfreeze blocks 4-5 for Stage 2
    model = unfreeze_vgg16_blocks(model, blocks_to_unfreeze=[4, 5], verbose=False)
    model = model.to(device)
    
    print(f"✓ Model loaded from: {STAGE1_CHECKPOINT}")
    print(f"✓ Unfrozen blocks 4-5 for Stage 2 configuration")
    
    # Get class weights
    class_weights = calculate_class_weights(DATA_DIR / 'train')
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Run benchmark
    df, best_config = run_stage2_optimizer_benchmark(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        num_epochs=10,
        output_dir=OUTPUT_DIR
    )
    
    print(f"""\n{'='*80}
STAGE 2 OPTIMIZER BENCHMARK COMPLETE
{'='*80}

Outputs saved to: {OUTPUT_DIR}
Next steps:
1. Review stage2_optimizer_benchmark_report.txt
2. Stage 2 training will auto-use configs/best_stage2_optimizer_config.json
3. Run: python scripts/train_stage2_progressive.py""")
    print("=" * 80)


if __name__ == '__main__':
    main()
