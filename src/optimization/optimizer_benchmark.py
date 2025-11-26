"""
Component 5: Optimizer & Learning Rate Tuning
==============================================

Purpose:
    Compare Adam vs AdamW and find optimal Stage 3 learning rate.
    
Current Setup (from Phase 1):
    - Optimizer: Adam
    - Stage 3 LR: 5e-6
    - Weight decay: 1e-4
    - Validation loss: 1.32 (suggests suboptimal convergence)

Hypotheses:
    - LR might be too conservative (5e-6)
    - AdamW might handle weight decay better for VGG16's 138M parameters
    - Higher LR (1e-5 or 2e-5) could improve convergence

Test Matrix:
    - Adam with [5e-6, 1e-5, 2e-5]
    - AdamW with [5e-6, 1e-5, 2e-5]
    Total: 6 configurations

Expected Gain: +2-3% accuracy, -0.15 loss

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
from typing import Dict
from copy import deepcopy


def run_optimizer_benchmark(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_epochs: int = 10,
    output_dir: Path = None
) -> pd.DataFrame:
    """
    Benchmark different optimizers and learning rates on Stage 3 training.
    
    Args:
        model: VGG16 emotion model (Stage 2 checkpoint loaded)
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function (with class weights)
        device: Device to train on
        num_epochs: Number of epochs per config (default: 10)
        output_dir: Where to save results
    
    Returns:
        DataFrame with benchmark results
    """
    print(f"""{'='*80}
COMPONENT 5: OPTIMIZER BENCHMARK
{'='*80}
Number of epochs per config: {num_epochs}""")
    print(f"Total configurations: 6")
    
    # Define optimizer configurations
    configs = [
        # Current baseline
        {'name': 'Adam_5e-6', 'optimizer': 'adam', 'lr': 5e-6, 'weight_decay': 1e-4},
        
        # Higher LR variants for Adam
        {'name': 'Adam_1e-5', 'optimizer': 'adam', 'lr': 1e-5, 'weight_decay': 1e-4},
        {'name': 'Adam_2e-5', 'optimizer': 'adam', 'lr': 2e-5, 'weight_decay': 1e-4},
        
        # AdamW variants (better weight decay handling)
        {'name': 'AdamW_5e-6', 'optimizer': 'adamw', 'lr': 5e-6, 'weight_decay': 5e-5},
        {'name': 'AdamW_1e-5', 'optimizer': 'adamw', 'lr': 1e-5, 'weight_decay': 5e-5},
        {'name': 'AdamW_2e-5', 'optimizer': 'adamw', 'lr': 2e-5, 'weight_decay': 5e-5},
    ]
    
    results = []
    best_config = None
    best_val_acc = 0.0
    
    # Save initial model state
    initial_state = deepcopy(model.state_dict())
    
    for config_idx, config in enumerate(configs, 1):
        print(f"""\n{'='*80}
TESTING CONFIG {config_idx}/6: {config['name']}
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
            factor=0.3,
            patience=3
        )
        
        # Training loop
        epoch_results = []
        
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
            
            print(f"Epoch {epoch:2d}/{num_epochs} | "
                  f"Train: {train_loss:.4f}/{train_acc:5.2f}% | "
                  f"Val: {val_loss:.4f}/{val_acc:5.2f}%")
        
        # Get final metrics
        final_epoch = epoch_results[-1]
        
        # Check convergence (did it improve from epoch 1 to epoch 10?)
        first_val_loss = epoch_results[0]['val_loss']
        final_val_loss = final_epoch['val_loss']
        converged = final_val_loss < first_val_loss - 0.05
        
        results.append({
            'config': config['name'],
            'optimizer': config['optimizer'],
            'learning_rate': config['lr'],
            'weight_decay': config['weight_decay'],
            'final_val_loss': final_val_loss,
            'final_val_acc': final_epoch['val_acc'],
            'final_train_loss': final_epoch['train_loss'],
            'final_train_acc': final_epoch['train_acc'],
            'converged': converged,
            'loss_improvement': first_val_loss - final_val_loss
        })
        
        # Track best config
        if final_epoch['val_acc'] > best_val_acc:
            best_val_acc = final_epoch['val_acc']
            best_config = config
        
        print(f"""\nConfig {config['name']} Results:
  Final Val Loss: {final_val_loss:.4f}
  Final Val Acc: {final_epoch['val_acc']:.2f}%
  Converged: {'Yes' if converged else 'No'}""")
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    
    # Display summary
    print(f"""\n{'='*80}
OPTIMIZER BENCHMARK SUMMARY
{'='*80}

{df.to_string(index=False)}""")
    
    print(f"""\n{'='*80}
BEST CONFIGURATION
{'='*80}
Config: {best_config['name']}
  Optimizer: {best_config['optimizer'].upper()}
  Learning rate: {best_config['lr']:.0e}
  Weight decay: {best_config['weight_decay']:.0e}
  Final Val Acc: {best_val_acc:.2f}%""")
    
    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save CSV
        csv_path = output_dir / 'optimizer_benchmark_results.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Results saved to: {csv_path}")
        
        # Save best config JSON
        best_config_path = Path('configs/best_optimizer_config.json')
        best_config_path.parent.mkdir(parents=True, exist_ok=True)
        
        best_config_data = {
            'optimizer': best_config['optimizer'],
            'learning_rate': best_config['lr'],
            'weight_decay': best_config['weight_decay'],
            'description': f'Best optimizer config from benchmark (Val Acc: {best_val_acc:.2f}%)',
            'usage': 'Use these settings for Stage 3 training'
        }
        
        with open(best_config_path, 'w') as f:
            json.dump(best_config_data, f, indent=2)
        
        print(f"✓ Best config saved to: {best_config_path}")
        
        # Generate visualizations
        _visualize_optimizer_results(df, output_dir)
    
    return df, best_config


def _visualize_optimizer_results(df: pd.DataFrame, output_dir: Path):
    """Create visualization of optimizer benchmark results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Validation Accuracy
    ax1 = axes[0, 0]
    colors = ['#2E86AB' if 'Adam_' in name else '#A23B72' 
              for name in df['config']]
    
    bars1 = ax1.bar(range(len(df)), df['final_val_acc'], color=colors, alpha=0.8)
    ax1.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Final Validation Accuracy by Config', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels(df['config'], rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, df['final_val_acc'])):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Validation Loss
    ax2 = axes[0, 1]
    bars2 = ax2.bar(range(len(df)), df['final_val_loss'], color=colors, alpha=0.8)
    ax2.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Final Validation Loss by Config', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(df)))
    ax2.set_xticklabels(df['config'], rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars2, df['final_val_loss']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 3: Adam vs AdamW comparison
    ax3 = axes[1, 0]
    adam_df = df[df['optimizer'] == 'adam']
    adamw_df = df[df['optimizer'] == 'adamw']
    
    x_adam = [f"{lr:.0e}" for lr in adam_df['learning_rate']]
    x_adamw = [f"{lr:.0e}" for lr in adamw_df['learning_rate']]
    
    x_pos = range(len(x_adam))
    width = 0.35
    
    bars_adam = ax3.bar([p - width/2 for p in x_pos], adam_df['final_val_acc'], 
                        width, label='Adam', color='#2E86AB', alpha=0.8)
    bars_adamw = ax3.bar([p + width/2 for p in x_pos], adamw_df['final_val_acc'], 
                         width, label='AdamW', color='#A23B72', alpha=0.8)
    
    ax3.set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Adam vs AdamW Comparison', fontsize=14, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(x_adam)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Convergence indicator
    ax4 = axes[1, 1]
    converged_counts = df.groupby('optimizer')['converged'].sum()
    
    ax4.bar(converged_counts.index, converged_counts.values, 
           color=['#2E86AB', '#A23B72'], alpha=0.8)
    ax4.set_xlabel('Optimizer', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Converged Configs (out of 3)', fontsize=12, fontweight='bold')
    ax4.set_title('Convergence by Optimizer Type', fontsize=14, fontweight='bold')
    ax4.set_ylim(0, 3.5)
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (opt, count) in enumerate(converged_counts.items()):
        ax4.text(i, count + 0.1, str(count), ha='center', va='bottom', 
                fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    save_path = output_dir / 'optimizer_benchmark_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualization saved to: {save_path}")


def save_benchmark_results(df: pd.DataFrame, best_config: Dict, output_dir: Path):
    """
    Generate detailed benchmark report.
    
    Args:
        df: Results DataFrame
        best_config: Best configuration dict
        output_dir: Where to save report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / 'optimizer_benchmark_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("OPTIMIZER & LEARNING RATE BENCHMARK REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Objective:\n")
        f.write("  Compare Adam vs AdamW with different learning rates for Stage 3 training.\n")
        f.write("  Current baseline: Adam with LR=5e-6, producing Val Loss=1.32\n\n")
        
        f.write("Hypothesis:\n")
        f.write("  - LR too conservative (5e-6) → slow convergence\n")
        f.write("  - AdamW better weight decay handling for 138M parameters\n")
        f.write("  - Higher LR (1e-5, 2e-5) could improve final performance\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("RESULTS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        # Sort by val_acc
        df_sorted = df.sort_values('final_val_acc', ascending=False)
        
        f.write(f"{'Rank':<6} | {'Config':<12} | {'Val Loss':>10} | {'Val Acc':>10} | {'Converged':>10}\n")
        f.write("-" * 80 + "\n")
        
        for rank, (_, row) in enumerate(df_sorted.iterrows(), 1):
            marker = "***" if rank == 1 else ""
            f.write(f"{rank:<6} | {row['config']:<12} | {row['final_val_loss']:10.4f} | "
                   f"{row['final_val_acc']:9.2f}% | {str(row['converged']):>10} {marker}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("BEST CONFIGURATION\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Config: {best_config['name']}\n")
        f.write(f"  Optimizer: {best_config['optimizer'].upper()}\n")
        f.write(f"  Learning Rate: {best_config['lr']:.0e}\n")
        f.write(f"  Weight Decay: {best_config['weight_decay']:.0e}\n\n")
        
        best_row = df[df['config'] == best_config['name']].iloc[0]
        f.write(f"Performance:\n")
        f.write(f"  Validation Loss: {best_row['final_val_loss']:.4f}\n")
        f.write(f"  Validation Accuracy: {best_row['final_val_acc']:.2f}%\n")
        f.write(f"  Converged: {'Yes' if best_row['converged'] else 'No'}\n\n")
        
        # Comparison to baseline
        baseline_row = df[df['config'] == 'Adam_5e-6'].iloc[0]
        f.write("Improvement over baseline (Adam_5e-6):\n")
        f.write(f"  Val Loss: {baseline_row['final_val_loss']:.4f} -> {best_row['final_val_loss']:.4f} "
               f"({best_row['final_val_loss'] - baseline_row['final_val_loss']:+.4f})\n")
        f.write(f"  Val Acc: {baseline_row['final_val_acc']:.2f}% -> {best_row['final_val_acc']:.2f}% "
               f"({best_row['final_val_acc'] - baseline_row['final_val_acc']:+.2f}%)\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. Use the best configuration for Stage 3 training:\n")
        f.write(f"   python scripts/train_stage3_deep.py \\\n")
        f.write(f"     --lr {best_config['lr']:.0e} \\\n")
        f.write(f"     --weight-decay {best_config['weight_decay']:.0e}\n\n")
        
        if best_config['optimizer'] == 'adamw':
            f.write("2. Switch to AdamW optimizer in training script:\n")
            f.write("   optimizer = optim.AdamW(model.parameters(), ...)\n\n")
        
        f.write("3. Expected improvements:\n")
        f.write(f"   - Better convergence (loss reduction)\n")
        f.write(f"   - Improved validation accuracy\n")
        f.write(f"   - More stable training\n\n")
    
    print(f"✓ Detailed report saved to: {report_path}")


def main():
    """Example usage of optimizer benchmark."""
    from pathlib import Path
    import sys
    
    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from config import Config
    from src.data.data_pipeline import create_dataloaders, calculate_class_weights
    from src.models.vgg16_emotion import build_emotion_model, unfreeze_vgg16_blocks
    
    print(f"""{'='*80}
COMPONENT 5: OPTIMIZER BENCHMARK
{'='*80}""")
    
    # Configuration
    STAGE2_CHECKPOINT = Path('models/emotion_stage2_progressive.pth')
    DATA_DIR = Path('data/raw')
    OUTPUT_DIR = Path('results/optimization/optimizers')
    
    if not STAGE2_CHECKPOINT.exists():
        print(f"""\nError: Stage 2 checkpoint not found: {STAGE2_CHECKPOINT}
Please train Stage 2 first or update checkpoint path""")
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
    
    # Load model from Stage 2
    print("\nLoading Stage 2 model...")
    model = build_emotion_model(num_classes=7, pretrained=True, verbose=False)
    checkpoint = torch.load(STAGE2_CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Unfreeze blocks 2-5 for Stage 3
    model = unfreeze_vgg16_blocks(model, blocks_to_unfreeze=[2, 3, 4, 5], verbose=False)
    model = model.to(device)
    
    print(f"✓ Model loaded from: {STAGE2_CHECKPOINT}")
    
    # Get class weights
    class_weights = calculate_class_weights(DATA_DIR / 'train')
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Run benchmark
    df, best_config = run_optimizer_benchmark(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        num_epochs=10,
        output_dir=OUTPUT_DIR
    )
    
    # Save detailed report
    save_benchmark_results(df, best_config, OUTPUT_DIR)
    
    print(f"""\n{'='*80}
OPTIMIZER BENCHMARK COMPLETE
{'='*80}

Outputs saved to: {OUTPUT_DIR}
Next steps:
1. Review optimizer_benchmark_comparison.png
2. Check optimizer_benchmark_report.txt for recommendations
3. Use configs/best_optimizer_config.json for training""")
    print("=" * 80)


if __name__ == '__main__':
    main()
