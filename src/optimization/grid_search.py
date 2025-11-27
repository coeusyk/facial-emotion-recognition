"""
Grid Search Module for Hyperparameter Optimization
====================================================

This module implements a complete grid search system for the 3-stage
progressive training pipeline. It runs full training for each configuration
and tracks results for comparison.

Features:
- Configuration generation with priority ranking
- Subprocess-based training for stability
- Results tracking to CSV
- Resumability for long-running searches
- Best configuration selection

Usage:
    # Generate configs and run grid search
    python scripts/run_grid_search.py --phase 1
    
    # Resume interrupted search
    python scripts/run_grid_search.py --phase 1 --resume
    
    # Dry run to preview configs
    python scripts/run_grid_search.py --phase 1 --dry-run

Author: FER-2013 Optimization Pipeline
"""

import itertools
import json
import subprocess
import sys
import time
import re
import os
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd


class GridSearchConfig:
    """Configuration class for grid search parameters."""
    
    # Phase 1: Minimal search (8 configs, ~2.5 hours)
    PHASE1_SEARCH_SPACE = {
        'dropout': [0.4, 0.5],
        'stage1_epochs': [20, 30],
        'stage3_epochs': [15, 20],
    }
    PHASE1_FIXED = {
        'stage2_epochs': 15,
        'weight_decay': 1e-5,
        'batch_size': 32,
        'label_smoothing': 0.0,
    }
    
    # Phase 2 Mini: Focused search with batch_size=64 (24 configs, ~5 hours)
    # Based on Phase 1 winner analysis - focuses on promising regions
    PHASE2_MINI_SEARCH_SPACE = {
        'dropout': [0.4, 0.5],
        'stage1_epochs': [25, 30],
        'stage2_epochs': [15, 20, 25],
        'stage3_epochs': [20, 25],
    }
    PHASE2_MINI_FIXED = {
        'weight_decay': 1e-5,
        'batch_size': 64,  # Key change from Phase 1
        'label_smoothing': 0.0,
    }
    
    # Phase 2: Comprehensive search (108 configs, ~36 hours)
    PHASE2_SEARCH_SPACE = {
        'dropout': [0.3, 0.4, 0.5, 0.6],
        'stage1_epochs': [20, 25, 30],
        'stage2_epochs': [12, 15, 20],
        'stage3_epochs': [15, 20, 25],
    }
    PHASE2_FIXED = {
        'weight_decay': 1e-5,
        'batch_size': 32,
        'label_smoothing': 0.0,
    }
    
    # Phase 3: Full search including weight decay and batch size
    PHASE3_SEARCH_SPACE = {
        'dropout': [0.3, 0.4, 0.5, 0.6],
        'stage1_epochs': [20, 25, 30],
        'stage2_epochs': [12, 15, 20],
        'stage3_epochs': [15, 20, 25],
        'weight_decay': [1e-6, 1e-5, 5e-5],
    }
    PHASE3_FIXED = {
        'batch_size': 32,
        'label_smoothing': 0.0,
    }


def generate_grid_configs(
    phase = 1,
    custom_search_space: Dict = None,
    custom_fixed: Dict = None
) -> List[Dict]:
    """
    Generate all grid search configurations for a given phase.
    
    Args:
        phase: Search phase (1=minimal, 2=comprehensive, '2m'=mini, 3=full)
        custom_search_space: Optional custom search space dict
        custom_fixed: Optional custom fixed parameters dict
    
    Returns:
        List of configuration dicts with unique IDs
    """
    if custom_search_space:
        search_space = custom_search_space
        fixed_params = custom_fixed or {}
    elif phase == 1:
        search_space = GridSearchConfig.PHASE1_SEARCH_SPACE
        fixed_params = GridSearchConfig.PHASE1_FIXED
    elif phase == '2m' or phase == 21:  # Mini Phase 2
        search_space = GridSearchConfig.PHASE2_MINI_SEARCH_SPACE
        fixed_params = GridSearchConfig.PHASE2_MINI_FIXED
    elif phase == 2:
        search_space = GridSearchConfig.PHASE2_SEARCH_SPACE
        fixed_params = GridSearchConfig.PHASE2_FIXED
    elif phase == 3:
        search_space = GridSearchConfig.PHASE3_SEARCH_SPACE
        fixed_params = GridSearchConfig.PHASE3_FIXED
    else:
        raise ValueError(f"Unknown phase: {phase}. Use 1, 2, '2m', or 3.")
    
    # Generate all combinations
    param_names = list(search_space.keys())
    param_values = list(search_space.values())
    
    configs = []
    
    for idx, combo in enumerate(itertools.product(*param_values), 1):
        config = fixed_params.copy()
        
        # Add search space parameters
        for name, value in zip(param_names, combo):
            config[name] = value
        
        # Generate unique config ID
        config['config_id'] = idx
        config['config_name'] = _generate_config_name(config)
        
        configs.append(config)
    
    return configs


def _generate_config_name(config: Dict) -> str:
    """Generate a human-readable config name."""
    parts = []
    
    if 'dropout' in config:
        parts.append(f"d{config['dropout']:.1f}")
    if 'stage1_epochs' in config:
        parts.append(f"e1_{config['stage1_epochs']}")
    if 'stage2_epochs' in config:
        parts.append(f"e2_{config['stage2_epochs']}")
    if 'stage3_epochs' in config:
        parts.append(f"e3_{config['stage3_epochs']}")
    if 'weight_decay' in config:
        wd = config['weight_decay']
        parts.append(f"wd{wd:.0e}".replace('e-0', 'e-'))
    
    return "_".join(parts)


def _run_subprocess_with_logging(
    cmd: List[str],
    cwd: str,
    env: Dict,
    timeout: float,
    log_file: Path,
    stage_name: str,
    stream_output: bool = True
) -> Tuple[int, str, str]:
    """
    Run subprocess with logging to file.
    
    Args:
        cmd: Command and arguments
        cwd: Working directory
        env: Environment variables
        timeout: Timeout in seconds
        log_file: Path to write log output
        stage_name: Name for display (e.g., "Stage 1")
        stream_output: Whether to print progress indicators
    
    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    if stream_output:
        print(f"    Running... (logging to {log_file.name})")
    
    try:
        # Run subprocess and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=timeout,
            cwd=cwd,
            env=env
        )
        
        # Write complete log to file (stdout only, skip stderr to reduce file size)
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"{'='*60}\n")
            f.write(f"{stage_name} Training Log\n")
            f.write(f"Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Return code: {result.returncode}\n")
            f.write(f"{'='*60}\n\n")
            f.write(result.stdout)
        
        return result.returncode, result.stdout, result.stderr
        
    except subprocess.TimeoutExpired as e:
        error_msg = f"Timeout after {timeout}s"
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"{'='*60}\n")
            f.write(f"{stage_name} Training Log - TIMEOUT\n")
            f.write(f"{'='*60}\n")
            f.write(error_msg)
        raise
    except Exception as e:
        error_msg = str(e)
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"{'='*60}\n")
            f.write(f"{stage_name} Training Log - ERROR\n")
            f.write(f"{'='*60}\n")
            f.write(error_msg)
        return -1, '', error_msg


def run_single_config(
    config: Dict,
    output_dir: Path,
    project_root: Path,
    timeout_hours: float = 2.0,
    verbose: bool = True,
    stream_output: bool = True
) -> Dict:
    """
    Run full 3-stage training for a single configuration.
    
    Args:
        config: Configuration dict with hyperparameters
        output_dir: Directory to save model checkpoints and logs
        project_root: Path to project root directory
        timeout_hours: Maximum time for complete training
        verbose: Whether to print progress
        stream_output: Whether to stream subprocess output in real-time
    
    Returns:
        Results dict with accuracies, losses, and timing
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config_name = config.get('config_name', f"config_{config['config_id']}")
    
    results = {
        'config_id': config['config_id'],
        'config_name': config_name,
        'status': 'pending',
        'stage1_val_acc': 0.0,
        'stage1_val_loss': 0.0,
        'stage2_val_acc': 0.0,
        'stage2_val_loss': 0.0,
        'stage3_val_acc': 0.0,
        'stage3_val_loss': 0.0,
        'total_time_minutes': 0.0,
        'error_message': None,
        **{k: v for k, v in config.items() if k not in ['config_id', 'config_name']}
    }
    
    start_time = time.time()
    timeout_seconds = timeout_hours * 3600
    
    # Set UTF-8 encoding for subprocess to handle Unicode characters on Windows
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    
    try:
        # Stage 1: Warmup Training
        if verbose:
            print(f"\n  [Stage 1] Starting warmup training...")
        
        stage1_cmd = [
            sys.executable, '-u',  # Unbuffered output
            str(project_root / 'scripts' / 'train_stage1_warmup.py'),
            '--epochs', str(config.get('stage1_epochs', 20)),
            '--dropout', str(config.get('dropout', 0.5)),
            '--weight-decay', str(config.get('weight_decay', 1e-5)),
            '--batch-size', str(config.get('batch_size', 32)),
            '--label-smoothing', str(config.get('label_smoothing', 0.0)),
            '--output-dir', str(output_dir),
        ]
        
        stage1_log = output_dir / 'stage1_training.log'
        stage1_retcode, stage1_stdout, stage1_stderr = _run_subprocess_with_logging(
            cmd=stage1_cmd,
            cwd=str(project_root),
            env=env,
            timeout=timeout_seconds,
            log_file=stage1_log,
            stage_name="Stage 1",
            stream_output=stream_output
        )
        
        if stage1_retcode != 0:
            results['status'] = 'failed_stage1'
            results['error_message'] = _extract_error(stage1_stderr)
            return results
        
        # Parse Stage 1 results
        stage1_acc, stage1_loss = _parse_training_output(stage1_stdout)
        results['stage1_val_acc'] = stage1_acc
        results['stage1_val_loss'] = stage1_loss
        
        if verbose:
            print(f"  [Stage 1] Complete: Val Acc={stage1_acc:.2f}%")
        
        # Stage 2: Progressive Fine-tuning
        if verbose:
            print(f"  [Stage 2] Starting progressive fine-tuning...")
        
        stage1_checkpoint = output_dir / 'emotion_stage1_warmup.pth'
        
        stage2_cmd = [
            sys.executable, '-u',  # Unbuffered output
            str(project_root / 'scripts' / 'train_stage2_progressive.py'),
            '--epochs', str(config.get('stage2_epochs', 15)),
            '--dropout', str(config.get('dropout', 0.5)),
            '--weight-decay', str(config.get('weight_decay', 1e-4)),
            '--batch-size', str(config.get('batch_size', 32)),
            '--label-smoothing', str(config.get('label_smoothing', 0.0)),
            '--stage1-checkpoint', str(stage1_checkpoint),
            '--output-dir', str(output_dir),
        ]
        
        remaining_time = timeout_seconds - (time.time() - start_time)
        if remaining_time <= 0:
            results['status'] = 'timeout'
            results['error_message'] = 'Timeout after Stage 1'
            return results
        
        stage2_log = output_dir / 'stage2_training.log'
        stage2_retcode, stage2_stdout, stage2_stderr = _run_subprocess_with_logging(
            cmd=stage2_cmd,
            cwd=str(project_root),
            env=env,
            timeout=remaining_time,
            log_file=stage2_log,
            stage_name="Stage 2",
            stream_output=stream_output
        )
        
        if stage2_retcode != 0:
            results['status'] = 'failed_stage2'
            results['error_message'] = _extract_error(stage2_stderr)
            return results
        
        # Parse Stage 2 results
        stage2_acc, stage2_loss = _parse_training_output(stage2_stdout)
        results['stage2_val_acc'] = stage2_acc
        results['stage2_val_loss'] = stage2_loss
        
        if verbose:
            print(f"  [Stage 2] Complete: Val Acc={stage2_acc:.2f}%")
        
        # Stage 3: Deep Fine-tuning
        if verbose:
            print(f"  [Stage 3] Starting deep fine-tuning...")
        
        stage2_checkpoint = output_dir / 'emotion_stage2_progressive.pth'
        
        stage3_cmd = [
            sys.executable, '-u',  # Unbuffered output
            str(project_root / 'scripts' / 'train_stage3_deep.py'),
            '--epochs', str(config.get('stage3_epochs', 15)),
            '--dropout', str(config.get('dropout', 0.5)),
            '--weight-decay', str(config.get('weight_decay', 1e-4)),
            '--batch-size', str(config.get('batch_size', 32)),
            '--label-smoothing', str(config.get('label_smoothing', 0.0)),
            '--stage2-checkpoint', str(stage2_checkpoint),
            '--output-dir', str(output_dir),
        ]
        
        remaining_time = timeout_seconds - (time.time() - start_time)
        if remaining_time <= 0:
            results['status'] = 'timeout'
            results['error_message'] = 'Timeout after Stage 2'
            return results
        
        stage3_log = output_dir / 'stage3_training.log'
        stage3_retcode, stage3_stdout, stage3_stderr = _run_subprocess_with_logging(
            cmd=stage3_cmd,
            cwd=str(project_root),
            env=env,
            timeout=remaining_time,
            log_file=stage3_log,
            stage_name="Stage 3",
            stream_output=stream_output
        )
        
        if stage3_retcode != 0:
            results['status'] = 'failed_stage3'
            results['error_message'] = _extract_error(stage3_stderr)
            return results
        
        # Parse Stage 3 results
        stage3_acc, stage3_loss = _parse_training_output(stage3_stdout)
        results['stage3_val_acc'] = stage3_acc
        results['stage3_val_loss'] = stage3_loss
        
        if verbose:
            print(f"  [Stage 3] Complete: Val Acc={stage3_acc:.2f}%")
        
        results['status'] = 'success'
        
    except subprocess.TimeoutExpired:
        results['status'] = 'timeout'
        results['error_message'] = f'Exceeded {timeout_hours}h timeout'
    except Exception as e:
        results['status'] = 'error'
        results['error_message'] = str(e)
    
    results['total_time_minutes'] = (time.time() - start_time) / 60
    
    return results


def _parse_training_output(output: str) -> Tuple[float, float]:
    """Extract best validation accuracy and loss from training output."""
    val_acc = 0.0
    val_loss = 0.0
    
    # Look for "Best validation accuracy: XX.XX%"
    acc_match = re.search(r'Best validation accuracy:\s*([\d.]+)%', output)
    if acc_match:
        val_acc = float(acc_match.group(1))
    
    # Look for val_loss in epoch results
    loss_matches = re.findall(r'Val Loss:\s*([\d.]+)', output)
    if loss_matches:
        val_loss = float(loss_matches[-1])  # Last reported val_loss
    
    return val_acc, val_loss


def _extract_error(stderr: str) -> str:
    """Extract meaningful error message from stderr."""
    if not stderr:
        return "Unknown error"
    
    # Look for common error patterns
    if 'CUDA out of memory' in stderr or 'OutOfMemoryError' in stderr:
        return 'CUDA OOM'
    if 'RuntimeError' in stderr:
        match = re.search(r'RuntimeError:\s*(.+?)(?:\n|$)', stderr)
        if match:
            return f'RuntimeError: {match.group(1)[:100]}'
    
    # Return last few lines
    lines = stderr.strip().split('\n')
    return lines[-1][:200] if lines else "Unknown error"


def run_grid_search(
    configs: List[Dict],
    output_base_dir: Path,
    project_root: Path,
    results_csv_path: Path = None,
    resume: bool = False,
    timeout_per_config_hours: float = 2.0,
    verbose: bool = True,
    stream_output: bool = True
) -> pd.DataFrame:
    """
    Run grid search over all configurations.
    
    Args:
        configs: List of configuration dicts
        output_base_dir: Base directory for config outputs
        project_root: Path to project root
        results_csv_path: Path to save/load results CSV
        resume: Whether to resume from previous run
        timeout_per_config_hours: Max time per config
        verbose: Whether to print progress
        stream_output: Whether to stream training output in real-time
    
    Returns:
        DataFrame with all results
    """
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    if results_csv_path is None:
        results_csv_path = output_base_dir / 'grid_search_results.csv'
    
    # Load existing results if resuming
    completed_configs = set()
    existing_results = []
    
    if resume and results_csv_path.exists():
        existing_df = pd.read_csv(results_csv_path)
        # Fill NaN in error_message with empty string to avoid issues
        existing_df['error_message'] = existing_df['error_message'].fillna('')
        completed_configs = set(existing_df['config_id'].tolist())
        existing_results = existing_df.to_dict('records')
        print(f"✓ Resuming: {len(completed_configs)} configs already completed")
    
    # Run remaining configs
    all_results = existing_results.copy()
    remaining_configs = [c for c in configs if c['config_id'] not in completed_configs]
    
    print(f"\n{'='*80}")
    print(f"GRID SEARCH: {len(remaining_configs)}/{len(configs)} configs to run")
    print(f"{'='*80}")
    
    estimated_hours = len(remaining_configs) * timeout_per_config_hours * 0.5  # Assume 50% of timeout
    print(f"Estimated time: {estimated_hours:.1f} hours")
    print(f"Results will be saved to: {results_csv_path}")
    print("="*80)
    
    for idx, config in enumerate(remaining_configs, 1):
        config_id = config['config_id']
        config_name = config['config_name']
        config_dir = output_base_dir / f"config_{config_id:03d}"
        
        print(f"\n[{idx}/{len(remaining_configs)}] Config {config_id}: {config_name}")
        print(f"  Directory: {config_dir}")
        
        # Run training
        result = run_single_config(
            config=config,
            output_dir=config_dir,
            project_root=project_root,
            timeout_hours=timeout_per_config_hours,
            verbose=verbose,
            stream_output=stream_output
        )
        
        all_results.append(result)
        
        # Save results after each config (crash recovery)
        results_df = pd.DataFrame(all_results)
        # Replace None with empty string for CSV, but handle NaN on read
        results_df = results_df.where(pd.notna(results_df), '')
        results_df.to_csv(results_csv_path, index=False)
        
        # Print result summary
        status = result['status']
        if status == 'success':
            print(f"  ✓ SUCCESS: Stage3 Val Acc = {result['stage3_val_acc']:.2f}%")
            print(f"    Time: {result['total_time_minutes']:.1f} min")
        else:
            print(f"  ✗ FAILED: {status}")
            if result['error_message']:
                print(f"    Error: {result['error_message']}")
    
    # Final results DataFrame
    results_df = pd.DataFrame(all_results)
    
    return results_df


def analyze_results(
    results_df: pd.DataFrame,
    output_dir: Path = None
) -> Dict:
    """
    Analyze grid search results and identify best configuration.
    
    Args:
        results_df: DataFrame with grid search results
        output_dir: Directory to save report
    
    Returns:
        Dict with analysis results
    """
    # Filter successful runs
    successful = results_df[results_df['status'] == 'success'].copy()
    
    if len(successful) == 0:
        print("⚠ No successful configurations found!")
        return {'best_config': None, 'success_count': 0}
    
    # Sort by Stage 3 validation accuracy
    successful = successful.sort_values('stage3_val_acc', ascending=False)
    
    # Best configuration
    best = successful.iloc[0]
    best_config = best.to_dict()
    
    # Analysis summary
    analysis = {
        'best_config': best_config,
        'success_count': len(successful),
        'total_count': len(results_df),
        'best_stage3_acc': best['stage3_val_acc'],
        'avg_stage3_acc': successful['stage3_val_acc'].mean(),
        'std_stage3_acc': successful['stage3_val_acc'].std(),
        'top_5_configs': successful.head(5)[['config_name', 'stage3_val_acc', 'dropout', 'stage1_epochs', 'stage3_epochs']].to_dict('records')
    }
    
    # Print summary
    print(f"\n{'='*80}")
    print("GRID SEARCH RESULTS ANALYSIS")
    print("="*80)
    print(f"\nSuccessful runs: {analysis['success_count']}/{analysis['total_count']}")
    print(f"\nStage 3 Validation Accuracy:")
    print(f"  Best:    {analysis['best_stage3_acc']:.2f}%")
    print(f"  Average: {analysis['avg_stage3_acc']:.2f}%")
    print(f"  Std Dev: {analysis['std_stage3_acc']:.2f}%")
    
    print(f"\n{'='*80}")
    print("BEST CONFIGURATION")
    print("="*80)
    print(f"  Config ID: {best_config['config_id']}")
    print(f"  Config Name: {best_config['config_name']}")
    print(f"\nHyperparameters:")
    for key in ['dropout', 'stage1_epochs', 'stage2_epochs', 'stage3_epochs', 'weight_decay', 'batch_size']:
        if key in best_config:
            print(f"  {key}: {best_config[key]}")
    print(f"\nResults:")
    print(f"  Stage 1 Val Acc: {best_config['stage1_val_acc']:.2f}%")
    print(f"  Stage 2 Val Acc: {best_config['stage2_val_acc']:.2f}%")
    print(f"  Stage 3 Val Acc: {best_config['stage3_val_acc']:.2f}%")
    print(f"  Training Time: {best_config['total_time_minutes']:.1f} minutes")
    
    print(f"\n{'='*80}")
    print("TOP 5 CONFIGURATIONS")
    print("="*80)
    for i, cfg in enumerate(analysis['top_5_configs'], 1):
        print(f"  {i}. {cfg['config_name']}: {cfg['stage3_val_acc']:.2f}%")
    
    # Save report if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        report_path = output_dir / 'grid_search_report.json'
        
        with open(report_path, 'w') as f:
            # Convert numpy types for JSON serialization
            analysis_json = {k: (v if not hasattr(v, 'item') else v.item()) 
                           for k, v in analysis.items() if k != 'best_config'}
            analysis_json['best_config'] = {k: (v if not hasattr(v, 'item') else v.item()) 
                                           for k, v in best_config.items()}
            json.dump(analysis_json, f, indent=2, default=str)
        
        print(f"\n✓ Report saved to: {report_path}")
    
    print("="*80)
    
    return analysis


def print_configs_preview(configs: List[Dict], max_show: int = 10):
    """Print preview of grid search configurations."""
    print(f"\n{'='*80}")
    print(f"GRID SEARCH CONFIGURATIONS ({len(configs)} total)")
    print("="*80)
    
    # Show first few configs
    for i, config in enumerate(configs[:max_show], 1):
        param_str = ", ".join([
            f"d={config.get('dropout', 0.5):.1f}",
            f"e1={config.get('stage1_epochs', 20)}",
            f"e2={config.get('stage2_epochs', 15)}",
            f"e3={config.get('stage3_epochs', 15)}",
        ])
        if 'weight_decay' in config and config.get('weight_decay') != 1e-5:
            param_str += f", wd={config['weight_decay']:.0e}"
        
        print(f"  {config['config_id']:3d}. {config['config_name']:<30} [{param_str}]")
    
    if len(configs) > max_show:
        print(f"  ... and {len(configs) - max_show} more configurations")
    
    # Estimate time
    avg_time_per_config = 20  # minutes
    total_time_hours = (len(configs) * avg_time_per_config) / 60
    print(f"\nEstimated total time: {total_time_hours:.1f} hours (assuming ~{avg_time_per_config} min/config)")
    print("="*80)


if __name__ == '__main__':
    # Test config generation
    print("Phase 1 configs:")
    configs = generate_grid_configs(phase=1)
    print_configs_preview(configs)
    
    print("\nPhase 2 configs:")
    configs = generate_grid_configs(phase=2)
    print_configs_preview(configs)
