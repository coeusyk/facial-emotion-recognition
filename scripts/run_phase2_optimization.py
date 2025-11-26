#!/usr/bin/env python3
"""
Phase 2 Master Integration Script
===================================

Orchestrates all 6 optimization components in sequence:
    1. Class Weight Adjustment
    2. Threshold Optimization
    3. Confusion-Aware Augmentation
    4. Label Smoothing (already integrated in training scripts)
    5. Optimizer Benchmark
    6. Grid Search (optional, very expensive)

Usage:
    python scripts/run_phase2_optimization.py [--components COMPONENTS]

Examples:
    # Run all components
    python scripts/run_phase2_optimization.py

    # Run specific components only
    python scripts/run_phase2_optimization.py --components 1,2,5

    # Skip grid search (recommended)
    python scripts/run_phase2_optimization.py --components 1,2,3,4,5

Expected Total Runtime: 18-24 hours (excluding grid search)
Expected Total Gain: +10-15% accuracy (53.76% → 63-68%)

Author: FER-2013 Optimization Pipeline
"""

import sys
from pathlib import Path
import argparse
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_component_1_class_weights(data_dir: Path, output_dir: Path):
    """
    Component 1: Class Weight Adjustment
    Expected gain: +3-4% accuracy
    Runtime: ~2 hours (1 retrain)
    """
    print(f"""\n{'='*80}
COMPONENT 1: CLASS WEIGHT ADJUSTMENT""")
    print("=" * 80)
    
    from src.optimization.class_weight_adjustment import (
        test_weight_strategies,
        save_weight_comparison_report,
        calculate_effective_number_weights,
        calculate_diagnostic_adjusted_weights,
        save_adjusted_weights
    )
    
    # Test all strategies
    comparison_df = test_weight_strategies(
        data_dir=data_dir / 'train',
        strategies=['conservative', 'moderate', 'aggressive'],
        output_dir=output_dir / 'class_weights'
    )
    
    # Generate report
    save_weight_comparison_report(comparison_df, output_dir / 'class_weights')
    
    # Save recommended (moderate) weights
    base_weights, class_names = calculate_effective_number_weights(data_dir / 'train')
    moderate_weights = calculate_diagnostic_adjusted_weights(
        base_weights, class_names, strategy='moderate'
    )
    
    save_adjusted_weights(
        weights=moderate_weights,
        class_names=class_names,
        strategy='moderate',
        output_dir=Path('configs')
    )
    
    print(f"""\n✓ Component 1 complete
  Outputs: {output_dir / 'class_weights'}""")
    print(f"  Weights saved: configs/class_weights_moderate.pth")
    
    return {'status': 'completed', 'output_dir': str(output_dir / 'class_weights')}


def run_component_2_thresholds(checkpoint_path: Path, data_dir: Path, output_dir: Path, device):
    """
    Component 2: Threshold Optimization
    Expected gain: +2-3% accuracy, Disgust F1 +20%
    Runtime: ~30 minutes (no retraining)
    """
    print(f"""\n{'='*80}
COMPONENT 2: THRESHOLD OPTIMIZATION""")
    print("=" * 80)
    
    import torch
    from src.data.data_pipeline import create_dataloaders
    from src.models.vgg16_emotion import build_emotion_model
    from src.optimization.threshold_tuning import (
        optimize_class_thresholds,
        generate_pr_curves,
        save_optimal_thresholds,
        generate_threshold_report
    )
    
    # Load data
    print("Loading validation data...")
    _, val_loader, _, class_names = create_dataloaders(
        data_dir,
        batch_size=64,
        num_workers=4
    )
    
    # Load model
    print("Loading trained model...")
    model = build_emotion_model(num_classes=7, pretrained=False, verbose=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Optimize thresholds
    optimal_thresholds, threshold_details = optimize_class_thresholds(
        model=model,
        val_loader=val_loader,
        class_names=class_names,
        device=device,
        target_metric='f1'
    )
    
    # Generate outputs
    generate_pr_curves(threshold_details, class_names, output_dir / 'thresholds')
    save_optimal_thresholds(optimal_thresholds, Path('configs/optimal_thresholds.json'))
    generate_threshold_report(optimal_thresholds, threshold_details, output_dir / 'thresholds')
    
    print(f"""\n✓ Component 2 complete
  Outputs: {output_dir / 'thresholds'}""")
    print(f"  Thresholds saved: configs/optimal_thresholds.json")
    
    return {'status': 'completed', 'output_dir': str(output_dir / 'thresholds')}


def run_component_3_augmentation(data_dir: Path):
    """
    Component 3: Confusion-Aware Augmentation
    Expected gain: +1-2% accuracy
    Runtime: ~2 hours (1 retrain)
    """
    print(f"""\n{'='*80}
COMPONENT 3: CONFUSION-AWARE AUGMENTATION""")
    print("=" * 80)
    
    from src.optimization.confusion_aware_augmentation import get_confusion_aware_dataloader
    
    # Create confusion-aware dataloader
    train_loader = get_confusion_aware_dataloader(
        data_dir=data_dir / 'train',
        batch_size=64,
        num_workers=4
    )
    
    # Test batch
    images, labels = next(iter(train_loader))
    print(f"""\n✓ Confusion-aware augmentation ready
  Sample batch shape: {images.shape}""")
    print(f"  Specialized transforms for: Fear, Sad, Neutral, Surprise")
    
    print("""\n✓ Component 3 complete
  Note: Use get_confusion_aware_dataloader() in training scripts""")
    
    return {'status': 'completed', 'note': 'Integrated in training via dataloader'}


def run_component_4_label_smoothing():
    """
    Component 4: Label Smoothing
    Expected gain: -0.15 loss, +0.5% accuracy
    Runtime: Already integrated in training scripts
    """
    print(f"""\n{'='*80}
COMPONENT 4: LABEL SMOOTHING""")
    print("=" * 80)
    
    print("""\n✓ Label smoothing already integrated in training scripts
  Usage: python scripts/train_stage3_deep.py --label-smoothing 0.1""")
    print("""\n  Test values: [0.0, 0.05, 0.1, 0.15, 0.2]
  Recommended: 0.1 (standard)""")
    
    print("\n✓ Component 4 complete")
    
    return {'status': 'integrated', 'recommended_value': 0.1}


def run_component_5_optimizer(checkpoint_path: Path, data_dir: Path, output_dir: Path, device):
    """
    Component 5: Optimizer Benchmark
    Expected gain: +2-3% accuracy
    Runtime: ~4 hours (6 configs × 40 min)
    """
    print(f"""\n{'='*80}
COMPONENT 5: OPTIMIZER BENCHMARK""")
    print("=" * 80)
    
    import torch
    import torch.nn as nn
    from src.data.data_pipeline import create_dataloaders, calculate_class_weights
    from src.models.vgg16_emotion import build_emotion_model, unfreeze_vgg16_blocks
    from src.optimization.optimizer_benchmark import run_optimizer_benchmark, save_benchmark_results
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, _, class_names = create_dataloaders(
        data_dir,
        batch_size=64,
        num_workers=4
    )
    
    # Load model from Stage 2
    print("Loading Stage 2 model...")
    model = build_emotion_model(num_classes=7, pretrained=True, verbose=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Unfreeze for Stage 3
    model = unfreeze_vgg16_blocks(model, blocks_to_unfreeze=[2, 3, 4, 5], verbose=False)
    model = model.to(device)
    
    # Get class weights
    class_weights = calculate_class_weights(data_dir / 'train')
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Run benchmark
    df, best_config = run_optimizer_benchmark(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        num_epochs=10,
        output_dir=output_dir / 'optimizers'
    )
    
    # Save detailed report
    save_benchmark_results(df, best_config, output_dir / 'optimizers')
    
    print(f"""\n✓ Component 5 complete
  Outputs: {output_dir / 'optimizers'}""")
    print("  Best config saved: configs/best_optimizer_config.json")
    
    return {'status': 'completed', 'best_config': best_config['name']}


def run_component_6_grid_search():
    """
    Component 6: Grid Search
    Expected gain: +1-2% accuracy
    Runtime: ~72 hours (36 configs × 2 hr)
    """
    print(f"""\n{'='*80}
COMPONENT 6: GRID SEARCH
{'='*80}

⚠ WARNING: Grid search is very expensive (~72 hours)""")
    print("  Consider running after validating Components 1-5 first")
    
    from src.optimization.targeted_grid_search import generate_grid_search_configs, save_grid_search_configs
    
    # Generate configs
    configs = generate_grid_search_configs()
    save_grid_search_configs(configs)
    
    print(f"""\n✓ Generated {len(configs)} grid search configurations
  Configs saved: configs/grid_search_configs.json""")
    print("""\n  To run grid search:
    1. Review configs in JSON file""")
    print("""    2. Run training with each config
    3. Compare results""")
    
    print("\n✓ Component 6 configs generated (not executed)")
    
    return {'status': 'configs_generated', 'total_configs': len(configs)}


def main():
    parser = argparse.ArgumentParser(description='Phase 2 Optimization Pipeline')
    parser.add_argument('--components', type=str, default='1,2,3,4,5',
                       help='Components to run (comma-separated, e.g., "1,2,5"). Default: all except 6 (grid search)')
    parser.add_argument('--data-dir', type=Path, default=Path('data/raw'),
                       help='Path to dataset directory')
    parser.add_argument('--checkpoint', type=Path, default=Path('models/emotion_stage3_deep.pth'),
                       help='Path to trained model checkpoint (for components 2 and 5)')
    parser.add_argument('--stage2-checkpoint', type=Path, default=Path('models/emotion_stage2_progressive.pth'),
                       help='Path to Stage 2 checkpoint (for component 5)')
    parser.add_argument('--output-dir', type=Path, default=Path('results/optimization'),
                       help='Output directory for all results')
    args = parser.parse_args()
    
    # Parse components to run
    components_to_run = [int(c.strip()) for c in args.components.split(',')]
    
    print(f"""{'='*80}
PHASE 2: DATA-DRIVEN OPTIMIZATION PIPELINE""")
    print(f"""{'='*80}

Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}""")
    print(f"""Components to run: {components_to_run}
Data directory: {args.data_dir}""")
    print(f"Output directory: {args.output_dir}")
    
    # Setup
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Track results
    results = {}
    
    # Run components
    if 1 in components_to_run:
        results['component_1'] = run_component_1_class_weights(args.data_dir, args.output_dir)
    
    if 2 in components_to_run:
        if not args.checkpoint.exists():
            print(f"""\n✗ Error: Checkpoint not found: {args.checkpoint}
  Skipping Component 2 (Threshold Optimization)""")
            results['component_2'] = {'status': 'skipped', 'reason': 'checkpoint not found'}
        else:
            results['component_2'] = run_component_2_thresholds(
                args.checkpoint, args.data_dir, args.output_dir, device
            )
    
    if 3 in components_to_run:
        results['component_3'] = run_component_3_augmentation(args.data_dir)
    
    if 4 in components_to_run:
        results['component_4'] = run_component_4_label_smoothing()
    
    if 5 in components_to_run:
        if not args.stage2_checkpoint.exists():
            print(f"""\n✗ Error: Stage 2 checkpoint not found: {args.stage2_checkpoint}
  Skipping Component 5 (Optimizer Benchmark)""")
            results['component_5'] = {'status': 'skipped', 'reason': 'checkpoint not found'}
        else:
            results['component_5'] = run_component_5_optimizer(
                args.stage2_checkpoint, args.data_dir, args.output_dir, device
            )
    
    if 6 in components_to_run:
        results['component_6'] = run_component_6_grid_search()
    
    # Save summary
    summary_path = args.output_dir / 'phase2_summary.json'
    summary = {
        'timestamp': datetime.now().isoformat(),
        'components_run': components_to_run,
        'results': results,
        'expected_improvements': {
            'accuracy': '+10-15% (53.76% -> 63-68%)',
            'macro_f1': '+0.08-0.12 (0.476 -> 0.55-0.60)',
            'fear_f1': '+0.17-0.23 (0.275 -> 0.45-0.50)',
            'disgust_f1': '+0.22-0.27 (0.281 -> 0.50-0.55)'
        }
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Final summary
    print(f"""\n{'='*80}
PHASE 2 OPTIMIZATION COMPLETE""")
    print(f"""{'='*80}

End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}""")
    print(f"""\nResults saved to: {args.output_dir}
Summary: {summary_path}""")
    
    print(f"""\n{'='*80}
NEXT STEPS""")
    print(f"""{'='*80}

1. Review all component outputs:""")
    for comp_num in components_to_run:
        if comp_num in [1, 2, 5]:
            print(f"   - Component {comp_num}: {args.output_dir / ['', 'class_weights', 'thresholds', '', '', 'optimizers'][comp_num]}")
    
    print("""\n2. Train Stage 3 with optimized settings:
   python scripts/train_stage3_deep.py \\""")
    print("     --label-smoothing 0.1 \\")
    
    # Check if optimizer benchmark was run
    if 5 in components_to_run and results.get('component_5', {}).get('status') == 'completed':
        best_config = results['component_5'].get('best_config', 'AdamW_2e-5')
        if 'AdamW' in best_config:
            print("     # Use AdamW optimizer (modify script)")

        lr = '2e-5' if '2e-5' in best_config else '1e-5' if '1e-5' in best_config else '5e-6'

        print(f"     --lr {lr} \\")
    
    print("     # Load optimized class weights in script")
    
    print("""\n3. Evaluate with optimal thresholds:
   # Use configs/optimal_thresholds.json during inference""")
    
    print("""\n4. Monitor improvements:
   - Confusion matrix: Fear->Sad should drop from 24.3% to <18%""")
    print("""   - Disgust F1: Should improve from 0.281 to 0.45-0.50
   - Overall accuracy: Target 63-68% (from 53.76%)""")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
