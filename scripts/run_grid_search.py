#!/usr/bin/env python3
"""
Grid Search CLI Entry Point
============================

Run hyperparameter grid search for the 3-stage progressive training pipeline.

Usage:
    # Phase 1: Minimal search (8 configs, ~2.5 hours)
    python scripts/run_grid_search.py --phase 1
    
    # Phase 2: Comprehensive search (108 configs, ~36 hours)
    python scripts/run_grid_search.py --phase 2
    
    # Resume interrupted search
    python scripts/run_grid_search.py --phase 1 --resume
    
    # Dry run (preview configs without training)
    python scripts/run_grid_search.py --phase 1 --dry-run
    
    # Custom timeout and output directory
    python scripts/run_grid_search.py --phase 1 --timeout 3.0 --output-dir results/custom_search

Author: FER-2013 Optimization Pipeline
"""

import sys
from pathlib import Path
import argparse

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.optimization.grid_search import (
    generate_grid_configs,
    run_grid_search,
    analyze_results,
    print_configs_preview
)


def main():
    parser = argparse.ArgumentParser(
        description='Run hyperparameter grid search for 3-stage training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_grid_search.py --phase 1            # Minimal search (8 configs)
  python scripts/run_grid_search.py --phase 2            # Comprehensive search (108 configs)
  python scripts/run_grid_search.py --phase 1 --resume   # Resume interrupted search
  python scripts/run_grid_search.py --phase 1 --dry-run  # Preview configs only
        """
    )
    
    parser.add_argument(
        '--phase', type=str, default='1',
        help='Search phase: 1=minimal (8 configs), 2m=mini (24 configs), 2=comprehensive (108 configs), 3=full (324 configs)'
    )
    parser.add_argument(
        '--output-dir', type=Path, default=None,
        help='Custom output directory (default: results/grid_search_phase{N})'
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume from previous run (skip completed configs)'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Preview configurations without running training'
    )
    parser.add_argument(
        '--timeout', type=float, default=2.0,
        help='Timeout per configuration in hours (default: 2.0)'
    )
    parser.add_argument(
        '--analyze-only', action='store_true',
        help='Only analyze existing results without running new configs'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Reduce output verbosity'
    )
    parser.add_argument(
        '--no-confirm', action='store_true',
        help='Skip confirmation prompt and start grid search immediately'
    )
    parser.add_argument(
        '--no-stream', action='store_true',
        help='Disable real-time training output (only show summary per stage)'
    )
    
    args = parser.parse_args()
    
    # Parse phase (can be int or '2m')
    phase = args.phase
    if phase.isdigit():
        phase = int(phase)
    
    # Determine output directory (at project root for easy .gitignore)
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = project_root / 'grid_search_results' / f'phase{phase}'
    
    results_csv = output_dir / 'grid_search_results.csv'
    
    print("=" * 80)
    print(f"GRID SEARCH - PHASE {phase}")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Results CSV: {results_csv}")
    
    # Generate configurations
    configs = generate_grid_configs(phase=phase)
    
    print(f"\nGenerated {len(configs)} configurations for Phase {phase}")
    
    # Dry run: just show configs
    if args.dry_run:
        print_configs_preview(configs, max_show=20)
        print("\n[DRY RUN] No training will be performed.")
        print("Remove --dry-run to start training.")
        return
    
    # Analyze only: show results
    if args.analyze_only:
        if not results_csv.exists():
            print(f"\n✗ ERROR: Results file not found: {results_csv}")
            print("  Run grid search first without --analyze-only")
            sys.exit(1)
        
        import pandas as pd
        results_df = pd.read_csv(results_csv)
        # Fill NaN in error_message with empty string
        results_df['error_message'] = results_df['error_message'].fillna('')
        analyze_results(results_df, output_dir)
        return
    
    # Show config preview
    if not args.quiet:
        print_configs_preview(configs, max_show=5)
    
    # Confirm before running (unless resuming or no-confirm flag)
    if not args.resume and not args.no_confirm:
        avg_time = 20  # minutes per config
        total_hours = (len(configs) * avg_time) / 60
        
        print(f"\n⚠ WARNING: This will run {len(configs)} training configurations")
        print(f"  Estimated time: {total_hours:.1f} hours")
        print(f"  Timeout per config: {args.timeout} hours")
        
        response = input("\nProceed with grid search? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Run grid search
    print(f"\n{'='*80}")
    print("STARTING GRID SEARCH")
    print("="*80)
    
    import pandas as pd
    
    results_df = run_grid_search(
        configs=configs,
        output_base_dir=output_dir,
        project_root=project_root,
        results_csv_path=results_csv,
        resume=args.resume,
        timeout_per_config_hours=args.timeout,
        verbose=not args.quiet,
        stream_output=not args.no_stream
    )
    
    # Analyze results
    print(f"\n{'='*80}")
    print("GRID SEARCH COMPLETE - ANALYZING RESULTS")
    print("="*80)
    
    analysis = analyze_results(results_df, output_dir)
    
    # Save best config
    if analysis['best_config']:
        best_config_path = output_dir / 'best_config.json'
        import json
        with open(best_config_path, 'w') as f:
            json.dump(analysis['best_config'], f, indent=2, default=str)
        print(f"\n✓ Best config saved to: {best_config_path}")
    
    print(f"\n{'='*80}")
    print("GRID SEARCH COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {results_csv}")
    print(f"Best config saved to: {output_dir / 'best_config.json'}")
    print(f"\nTo analyze results later:")
    print(f"  python scripts/run_grid_search.py --phase {phase} --analyze-only")
    print("="*80)


if __name__ == '__main__':
    main()
