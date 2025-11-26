"""
Adaptive Class Weighting for Multi-Stage Training
==================================================

Dynamically adjusts loss function class weights between training stages based on
per-class validation performance. Addresses class imbalance by automatically boosting
weights for underperforming emotions while maintaining training stability.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import csv


def calculate_adaptive_weights(
    per_class_accuracy: Dict[int, float],
    base_weights: torch.Tensor,
    class_names: List[str],
    stage: int,
    accuracy_threshold: Optional[float] = None,
    boost_strength: Optional[float] = None,
    max_weight: float = 5.0,
    min_weight: float = 0.3
) -> Tuple[torch.Tensor, Dict]:
    """
    Calculate adaptive class weights based on per-class validation performance.
    
    Boosts weights for classes performing below threshold, with boost magnitude
    proportional to performance gap. Uses stage-specific defaults for conservative
    early stages and more aggressive later stages.
    
    Args:
        per_class_accuracy: Dict mapping class index to validation accuracy (%)
                           e.g., {0: 45.5, 1: 12.3, 2: 8.7, ...}
        base_weights: Original Effective Number weights (torch.Tensor)
        class_names: List of emotion class names for logging
                    e.g., ['angry', 'disgust', 'fear', 'happy', ...]
        stage: Training stage number (1 for Stage 1→2, 2 for Stage 2→3)
        accuracy_threshold: Performance threshold in % (default: stage-dependent)
                           Classes below this get boosted
        boost_strength: Multiplier for boost calculation (default: stage-dependent)
                       Higher = more aggressive boosting
        max_weight: Maximum allowed weight value (prevents gradient explosion)
        min_weight: Minimum allowed weight value (prevents class suppression)
    
    Returns:
        Tuple of (adapted_weights, metadata):
            adapted_weights: torch.Tensor of adjusted weights ready for CrossEntropyLoss
            metadata: Dict containing boost factors and status for each class
    
    Stage-Specific Defaults:
        Stage 1→2: threshold=20%, boost_strength=2.0 (aggressive early boost)
        Stage 2→3: threshold=40%, boost_strength=1.5 (conservative refinement)
    
    Example:
        >>> per_class_acc = {0: 45.2, 1: 12.5, 2: 8.7, 3: 82.1, 4: 55.3, 5: 48.9, 6: 38.2}
        >>> base_weights = torch.tensor([1.08, 2.73, 1.07, 0.60, 0.84, 0.86, 1.19])
        >>> class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        >>> 
        >>> weights, meta = calculate_adaptive_weights(
        ...     per_class_acc, base_weights, class_names, stage=1
        ... )
        >>> # Classes below 20% (disgust, fear) get boosted
        >>> # Classes above 20% maintain base weights
    """
    # Stage-specific default parameters
    stage_defaults = {
        1: {'threshold': 20.0, 'boost': 2.0},  # Stage 1→2: Aggressive early correction
        2: {'threshold': 40.0, 'boost': 1.5}   # Stage 2→3: Conservative refinement
    }
    
    if stage not in stage_defaults:
        raise ValueError(f"Invalid stage {stage}. Must be 1 (Stage 1→2) or 2 (Stage 2→3)")
    
    # Use stage defaults if not explicitly provided
    if accuracy_threshold is None:
        accuracy_threshold = stage_defaults[stage]['threshold']
    if boost_strength is None:
        boost_strength = stage_defaults[stage]['boost']
    
    num_classes = len(base_weights)
    adapted_weights = base_weights.clone()
    metadata = {}
    
    print(f"\n{'='*80}")
    print(f"STAGE {stage} → STAGE {stage+1} ADAPTIVE WEIGHT ADJUSTMENT")
    print(f"{'='*80}")
    print(f"Performance Threshold: {accuracy_threshold:.1f}%")
    print(f"Boost Strength: {boost_strength}x")
    print(f"Weight Bounds: [{min_weight:.2f}, {max_weight:.2f}]")
    
    print(f"\n{'='*80}")
    print("Class Performance Analysis:")
    print(f"{'='*80}")
    print(f"{'Emotion':<12} {'Accuracy':<10} {'Base':<8} {'Boost':<8} {'New':<8} {'Status':<15}")
    print(f"{'-'*80}")
    
    for class_idx in range(num_classes):
        class_name = class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"
        base_weight = base_weights[class_idx].item()
        
        # Get accuracy (handle missing classes)
        accuracy = per_class_accuracy.get(class_idx, 0.0)
        
        # Calculate boost factor
        if accuracy < accuracy_threshold:
            # Performance gap: how far below threshold
            gap = accuracy_threshold - accuracy
            
            # Boost factor proportional to gap
            # Larger gap → stronger boost
            boost_factor = 1.0 + (gap / 100.0) * boost_strength
            
            # Apply boost
            new_weight = base_weight * boost_factor
            
            # Clip to safety bounds
            clipped_weight = np.clip(new_weight, min_weight, max_weight)
            
            # Determine status
            if clipped_weight == max_weight and new_weight > max_weight:
                status = "! CAPPED MAX"
            elif clipped_weight == min_weight and new_weight < min_weight:
                status = "! CAPPED MIN"
            else:
                status = "✓ BOOSTED"
            
            adapted_weights[class_idx] = clipped_weight
            
        else:
            # Above threshold: maintain base weight
            boost_factor = 1.0
            clipped_weight = base_weight
            status = "→ STABLE"
        
        # Store metadata
        metadata[class_idx] = {
            'class_name': class_name,
            'accuracy': accuracy,
            'base_weight': base_weight,
            'boost_factor': boost_factor,
            'final_weight': clipped_weight,
            'status': status
        }
        
        # Print row
        print(f"{class_name:<12} {accuracy:>6.2f}%   "
              f"{base_weight:>6.2f}  {boost_factor:>6.2f}x  "
              f"{clipped_weight:>6.2f}  {status:<15}")
    
    print(f"{'-'*80}")
    
    # Stability checks
    weight_ratio = adapted_weights.max().item() / adapted_weights.min().item()
    if weight_ratio > 20.0:
        print(f"\n⚠ WARNING: High weight ratio ({weight_ratio:.1f}:1)")
        print(f"  Training may be unstable. Consider reducing boost_strength.")
    
    # Verify all weights are valid
    if not torch.all(torch.isfinite(adapted_weights)):
        raise ValueError("Non-finite weights detected after adaptation")
    
    if not torch.all(adapted_weights > 0):
        raise ValueError("Non-positive weights detected after adaptation")
    
    print(f"\n✓ Adaptive weights calculated successfully")
    print(f"  Weight range: [{adapted_weights.min().item():.2f}, {adapted_weights.max().item():.2f}]")
    print(f"  Weight ratio: {weight_ratio:.2f}:1")
    print(f"{'='*80}\n")
    
    return adapted_weights, metadata


def save_weight_history(
    metadata: Dict,
    stage: int,
    log_file: Path = Path("logs/adaptive_weights_history.csv")
):
    """
    Save adaptive weight history to CSV for analysis.
    
    Args:
        metadata: Metadata dict from calculate_adaptive_weights()
        stage: Current stage number (1 or 2)
        log_file: Path to CSV log file
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file exists to determine if we need header
    file_exists = log_file.exists()
    
    with open(log_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header if new file
        if not file_exists:
            writer.writerow([
                'stage_transition',
                'emotion',
                'accuracy_pct',
                'base_weight',
                'boost_factor',
                'final_weight',
                'status'
            ])
        
        # Write data rows
        for class_idx in sorted(metadata.keys()):
            meta = metadata[class_idx]
            writer.writerow([
                f"{stage}->{stage+1}",
                meta['class_name'],
                f"{meta['accuracy']:.2f}",
                f"{meta['base_weight']:.4f}",
                f"{meta['boost_factor']:.4f}",
                f"{meta['final_weight']:.4f}",
                meta['status']
            ])
    
    print(f"✓ Weight history saved to: {log_file}")


def print_weight_comparison(
    base_weights: torch.Tensor,
    adapted_weights: torch.Tensor,
    class_names: List[str]
):
    """
    Print side-by-side comparison of base vs adapted weights.
    
    Args:
        base_weights: Original Effective Number weights
        adapted_weights: Adapted weights after boosting
        class_names: List of emotion class names
    """
    print(f"\n{'='*60}")
    print("WEIGHT COMPARISON: Base vs Adapted")
    print(f"{'='*60}")
    print(f"{'Emotion':<12} {'Base Weight':<15} {'Adapted Weight':<15} {'Change':<10}")
    print(f"{'-'*60}")
    
    for idx, class_name in enumerate(class_names):
        base = base_weights[idx].item()
        adapted = adapted_weights[idx].item()
        change = ((adapted - base) / base) * 100 if base > 0 else 0
        
        change_str = f"+{change:.1f}%" if change >= 0 else f"{change:.1f}%"
        
        print(f"{class_name:<12} {base:>8.4f}       {adapted:>8.4f}       {change_str:>8}")
    
    print(f"{'='*60}\n")
