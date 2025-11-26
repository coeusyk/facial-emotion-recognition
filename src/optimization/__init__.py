"""
Phase 2: Data-Driven Optimization for FER-2013 Emotion Recognition
====================================================================

This module contains targeted optimization components to address specific
failure modes identified in Phase 1 diagnostics:

1. Class Weight Adjustment - Fix Sad over-prediction and Fear under-detection
2. Threshold Optimization - Fix Disgust paradox (high AUC, low F1)
3. Confusion-Aware Augmentation - Target specific confusion pairs
4. Label Smoothing - Reduce overconfidence and improve calibration
5. Optimizer Benchmark - Test AdamW vs Adam with different learning rates
6. Targeted Grid Search - Fine-tune remaining hyperparameters

Author: FER-2013 Optimization Pipeline
Date: November 2025
"""

from .class_weight_adjustment import (
    calculate_diagnostic_adjusted_weights,
    test_weight_strategies,
    save_weight_comparison_report
)

from .threshold_tuning import (
    optimize_class_thresholds,
    generate_pr_curves,
    save_optimal_thresholds
)

from .confusion_aware_augmentation import (
    get_confusion_aware_transforms,
    ConfusionAwareDataset,
    get_confusion_aware_dataloader
)

from .optimizer_benchmark import (
    run_optimizer_benchmark,
    save_benchmark_results
)

from .targeted_grid_search import (
    run_grid_search,
    save_grid_search_results
)

__all__ = [
    # Component 1: Class Weight Adjustment
    'calculate_diagnostic_adjusted_weights',
    'test_weight_strategies',
    'save_weight_comparison_report',
    
    # Component 2: Threshold Optimization
    'optimize_class_thresholds',
    'generate_pr_curves',
    'save_optimal_thresholds',
    
    # Component 3: Confusion-Aware Augmentation
    'get_confusion_aware_transforms',
    'ConfusionAwareDataset',
    'get_confusion_aware_dataloader',
    
    # Component 4: Optimizer Benchmark
    'run_optimizer_benchmark',
    'save_benchmark_results',
    
    # Component 5: Grid Search
    'run_grid_search',
    'save_grid_search_results',
]
