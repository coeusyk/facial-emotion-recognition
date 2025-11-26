"""
Component 6: Targeted Grid Search
===================================

Purpose:
    Fine-tune remaining hyperparameters using best configs from Components 1-5.

Search Space (36 configurations):
    - dropout: [0.3, 0.4, 0.5]
    - batch_size: [32, 64]
    - warmup_epochs: [3, 5]
    - stage3_epochs: [10, 15, 20]

Fixed (from Components 1-5):
    - optimizer: AdamW (from Component 5)
    - lr_stage3: 2e-5 (from Component 5)
    - weight_decay: 5e-5 (from Component 5)
    - label_smoothing: 0.1 (from Component 4)
    - class_weights: diagnostic-adjusted (from Component 1)
    - thresholds: optimized per-class (from Component 2)

Expected Gain: +1-2% accuracy (fine-tuning)

Author: FER-2013 Optimization Pipeline
"""

import itertools
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List


def generate_grid_search_configs(
    fixed_params: Dict = None
) -> List[Dict]:
    """
    Generate all grid search configurations.
    
    Args:
        fixed_params: Fixed parameters from previous components
    
    Returns:
        List of configuration dicts
    """
    # Default fixed parameters (from Components 1-5)
    if fixed_params is None:
        fixed_params = {
            'optimizer': 'adamw',
            'learning_rate': 2e-5,
            'weight_decay': 5e-5,
            'label_smoothing': 0.1,
            'use_adjusted_weights': True,
            'use_optimal_thresholds': True
        }
    
    # Search space
    search_space = {
        'dropout': [0.3, 0.4, 0.5],
        'batch_size': [32, 64],
        'warmup_epochs': [3, 5],
        'stage3_epochs': [10, 15, 20]
    }
    
    # Generate all combinations
    param_names = list(search_space.keys())
    param_values = list(search_space.values())
    
    configs = []
    
    for combo in itertools.product(*param_values):
        config = fixed_params.copy()
        
        # Add search space parameters
        for name, value in zip(param_names, combo):
            config[name] = value
        
        # Generate config name
        config['name'] = f"d{combo[0]:.1f}_b{combo[1]}_w{combo[2]}_e{combo[3]}"
        
        configs.append(config)
    
    return configs


def save_grid_search_configs(
    configs: List[Dict],
    output_path: Path = None
):
    """
    Save grid search configurations to JSON.
    
    Args:
        configs: List of configuration dicts
        output_path: Where to save (default: configs/grid_search_configs.json)
    """
    if output_path is None:
        output_path = Path('configs/grid_search_configs.json')
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_data = {
        'total_configs': len(configs),
        'fixed_params': {
            'optimizer': 'adamw',
            'learning_rate': 2e-5,
            'weight_decay': 5e-5,
            'label_smoothing': 0.1
        },
        'search_space': {
            'dropout': [0.3, 0.4, 0.5],
            'batch_size': [32, 64],
            'warmup_epochs': [3, 5],
            'stage3_epochs': [10, 15, 20]
        },
        'configs': configs
    }
    
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"""✓ Grid search configs saved to: {output_path}
  Total configurations: {len(configs)}""")


def run_grid_search(
    configs: List[Dict],
    train_func,
    output_dir: Path = None
) -> pd.DataFrame:
    """
    Run grid search over all configurations.
    
    NOTE: This is a placeholder. Actual implementation would require
    running full 3-stage training for each config (very expensive).
    
    Args:
        configs: List of configuration dicts
        train_func: Function to train model with given config
        output_dir: Where to save results
    
    Returns:
        DataFrame with results
    """
    print(f"""{'='*80}
COMPONENT 6: GRID SEARCH
{'='*80}
Total configurations: {len(configs)}
Estimated runtime: {len(configs) * 2} hours (assuming 2hr per config)

⚠ WARNING: Grid search is very expensive!""")
    print("  Consider running only promising configs first.")
    
    # Placeholder - actual implementation would train each config
    results = []
    
    for i, config in enumerate(configs, 1):
        print(f"""\nConfig {i}/{len(configs)}: {config['name']}
  dropout={config['dropout']}, batch_size={config['batch_size']},""")
        print(f"  warmup_epochs={config['warmup_epochs']}, stage3_epochs={config['stage3_epochs']}")
        
        # Placeholder result (replace with actual training)
        result = {
            'config_name': config['name'],
            'dropout': config['dropout'],
            'batch_size': config['batch_size'],
            'warmup_epochs': config['warmup_epochs'],
            'stage3_epochs': config['stage3_epochs'],
            'val_acc': 0.0,  # Placeholder
            'val_loss': 0.0,  # Placeholder
            'train_time_hours': 0.0  # Placeholder
        }
        
        results.append(result)
    
    df = pd.DataFrame(results)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path = output_dir / 'grid_search_results.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Results saved to: {csv_path}")
    
    return df


def save_grid_search_results(
    df: pd.DataFrame,
    output_dir: Path
):
    """
    Generate grid search report.
    
    Args:
        df: Results DataFrame
        output_dir: Where to save report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / 'grid_search_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("GRID SEARCH REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Search Space:\n")
        f.write("  - dropout: [0.3, 0.4, 0.5]\n")
        f.write("  - batch_size: [32, 64]\n")
        f.write("  - warmup_epochs: [3, 5]\n")
        f.write("  - stage3_epochs: [10, 15, 20]\n\n")
        
        f.write(f"Total configurations tested: {len(df)}\n\n")
        
        # Best config
        if 'val_acc' in df.columns and df['val_acc'].max() > 0:
            best_row = df.loc[df['val_acc'].idxmax()]
            
            f.write("=" * 80 + "\n")
            f.write("BEST CONFIGURATION\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Config: {best_row['config_name']}\n")
            f.write(f"  Dropout: {best_row['dropout']}\n")
            f.write(f"  Batch size: {best_row['batch_size']}\n")
            f.write(f"  Warmup epochs: {best_row['warmup_epochs']}\n")
            f.write(f"  Stage 3 epochs: {best_row['stage3_epochs']}\n")
            f.write(f"  Validation accuracy: {best_row['val_acc']:.2f}%\n")
            f.write(f"  Validation loss: {best_row['val_loss']:.4f}\n\n")
    
    print(f"✓ Grid search report saved to: {report_path}")


def main():
    """Example usage of grid search."""
    
    print(f"""{'='*80}
COMPONENT 6: TARGETED GRID SEARCH""")
    print("=" * 80)
    
    # Generate all configs
    configs = generate_grid_search_configs()
    
    print(f"""\nGenerated {len(configs)} configurations

Sample configs:""")
    for i, config in enumerate(configs[:3], 1):
        print(f"""\n{i}. {config['name']}
   dropout={config['dropout']}, batch_size={config['batch_size']},""")
        print(f"   warmup_epochs={config['warmup_epochs']}, stage3_epochs={config['stage3_epochs']}")
    
    print(f"\n... and {len(configs) - 3} more")
    
    # Save configs
    save_grid_search_configs(configs)
    
    print(f"""\n{'='*80}
GRID SEARCH CONFIGS GENERATED
{'='*80}

To run grid search:
1. Load configs from configs/grid_search_configs.json
2. For each config, run full 3-stage training
3. Compare validation accuracies
4. Select best configuration

Estimated time: ~72 hours for all 36 configs
Consider running subset first (e.g., top 10 promising configs)""")
    print("=" * 80)


if __name__ == '__main__':
    main()
