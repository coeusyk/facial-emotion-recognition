#!/usr/bin/env python3
"""
Quick Test: Phase 1 Winner (Config #8) with batch_size=64
==========================================================

Tests the hypothesis that batch_size=64 improves performance.

Phase 1 Winner Config #8:
- dropout: 0.5
- stage1_epochs: 30
- stage2_epochs: 15  
- stage3_epochs: 20
- weight_decay: 1e-5
- batch_size: 32 → 64 (CHANGED)

Expected runtime: ~20 minutes
Target: Beat 61.02% (Phase 1 best)

Usage:
    python scripts/quick_batch64_test.py
"""

import sys
import os
import subprocess
import time
from pathlib import Path

# Project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test configuration - Phase 1 winner with batch_size=64
CONFIG = {
    'dropout': 0.5,
    'stage1_epochs': 30,
    'stage2_epochs': 15,
    'stage3_epochs': 20,
    'weight_decay': 1e-5,
    'batch_size': 64,  # Changed from 32
    'label_smoothing': 0.0,
}

PHASE1_BEST_ACC = 61.02  # Baseline to beat


def run_stage(stage_num: int, script_name: str, checkpoint_arg: str = None, 
              checkpoint_path: Path = None, output_dir: Path = None) -> tuple:
    """Run a training stage and return (success, val_acc, val_loss)."""
    
    cmd = [
        sys.executable, '-u',
        str(project_root / 'scripts' / script_name),
        '--epochs', str(CONFIG[f'stage{stage_num}_epochs']),
        '--dropout', str(CONFIG['dropout']),
        '--weight-decay', str(CONFIG['weight_decay']),
        '--batch-size', str(CONFIG['batch_size']),
        '--label-smoothing', str(CONFIG['label_smoothing']),
        '--output-dir', str(output_dir),
    ]
    
    if checkpoint_arg and checkpoint_path:
        cmd.extend([checkpoint_arg, str(checkpoint_path)])
    
    print(f"\n  Running Stage {stage_num}...")
    print(f"  Command: {' '.join(cmd[:5])}...")
    
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    
    start = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        cwd=str(project_root),
        env=env
    )
    elapsed = time.time() - start
    
    if result.returncode != 0:
        print(f"  ✗ Stage {stage_num} FAILED")
        print(f"  Error: {result.stderr[-500:] if result.stderr else 'Unknown'}")
        return False, 0.0, 0.0
    
    # Parse results
    import re
    acc_match = re.search(r'Best validation accuracy:\s*([\d.]+)%', result.stdout)
    loss_matches = re.findall(r'Val Loss:\s*([\d.]+)', result.stdout)
    
    val_acc = float(acc_match.group(1)) if acc_match else 0.0
    val_loss = float(loss_matches[-1]) if loss_matches else 0.0
    
    print(f"  ✓ Stage {stage_num} Complete: Val Acc={val_acc:.2f}%, Time={elapsed/60:.1f}min")
    
    return True, val_acc, val_loss


def main():
    print("=" * 80)
    print("QUICK TEST: Phase 1 Winner with batch_size=64")
    print("=" * 80)
    print(f"\nConfiguration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print(f"\nBaseline to beat: {PHASE1_BEST_ACC}% (Phase 1 best with batch_size=32)")
    print("=" * 80)
    
    output_dir = project_root / 'grid_search_results' / 'quick_batch64_test'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    start_time = time.time()
    results = {}
    
    # Stage 1
    success, acc, loss = run_stage(
        1, 'train_stage1_warmup.py',
        output_dir=output_dir
    )
    if not success:
        print("\n✗ Quick test FAILED at Stage 1")
        return
    results['stage1'] = {'acc': acc, 'loss': loss}
    
    # Stage 2
    stage1_ckpt = output_dir / 'emotion_stage1_warmup.pth'
    success, acc, loss = run_stage(
        2, 'train_stage2_progressive.py',
        checkpoint_arg='--stage1-checkpoint',
        checkpoint_path=stage1_ckpt,
        output_dir=output_dir
    )
    if not success:
        print("\n✗ Quick test FAILED at Stage 2")
        return
    results['stage2'] = {'acc': acc, 'loss': loss}
    
    # Stage 3
    stage2_ckpt = output_dir / 'emotion_stage2_progressive.pth'
    success, acc, loss = run_stage(
        3, 'train_stage3_deep.py',
        checkpoint_arg='--stage2-checkpoint',
        checkpoint_path=stage2_ckpt,
        output_dir=output_dir
    )
    if not success:
        print("\n✗ Quick test FAILED at Stage 3")
        return
    results['stage3'] = {'acc': acc, 'loss': loss}
    
    total_time = (time.time() - start_time) / 60
    
    # Summary
    print("\n" + "=" * 80)
    print("QUICK TEST RESULTS")
    print("=" * 80)
    print(f"\nStage Results:")
    print(f"  Stage 1: {results['stage1']['acc']:.2f}%")
    print(f"  Stage 2: {results['stage2']['acc']:.2f}%")
    print(f"  Stage 3: {results['stage3']['acc']:.2f}%")
    print(f"\nTotal time: {total_time:.1f} minutes")
    
    final_acc = results['stage3']['acc']
    improvement = final_acc - PHASE1_BEST_ACC
    
    print("\n" + "=" * 80)
    print("COMPARISON WITH PHASE 1 BEST")
    print("=" * 80)
    print(f"  Phase 1 best (batch_size=32): {PHASE1_BEST_ACC:.2f}%")
    print(f"  This test (batch_size=64):    {final_acc:.2f}%")
    print(f"  Improvement: {improvement:+.2f}%")
    
    if improvement > 0:
        print(f"\n✓ SUCCESS! batch_size=64 improved performance by {improvement:.2f}%")
        print("  → Proceed with Mini Phase 2 focusing on batch_size=64")
    else:
        print(f"\n✗ batch_size=64 did not improve performance")
        print("  → Consider other strategies (ensemble, TTA, etc.)")
    
    print("=" * 80)
    
    # Save results
    import json
    results_file = output_dir / 'quick_test_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'config': CONFIG,
            'results': results,
            'total_time_minutes': total_time,
            'phase1_baseline': PHASE1_BEST_ACC,
            'improvement': improvement,
        }, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
