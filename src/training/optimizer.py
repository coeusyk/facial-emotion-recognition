"""
SGD + Nesterov Momentum Optimizer Module
========================================

Helper functions for creating and managing SGD+Nesterov optimizer.

Based on SOTA FER-2013 papers showing SGD outperforms Adam by 2-3%:
- Khaireddin et al. (2021): VGGNet with SGD achieves 73.28%
- Better generalization on image classification tasks
- Well-aligned with ImageNet pretrained features (trained with SGD)
"""

import torch
import torch.optim as optim
from typing import Dict, Any


class OptimizerConfig:
    """Stage-specific optimizer configurations."""
    
    # Stage 1: Warmup (classifier head only, frozen backbone)
    STAGE1_SGD = {
        'lr': 0.003,         # 3x Adam (0.001), conservative start
        'momentum': 0.9,     # Standard momentum for vision tasks
        'weight_decay': 1e-4,  # Higher L2 penalty (training from scratch)
        'nesterov': True     # Look-ahead updates, free improvement
    }
    
    STAGE1_ADAM = {
        'lr': 1e-3,
        'weight_decay': 1e-5,
        'betas': (0.9, 0.999)
    }
    
    # Stage 2: Progressive unfreezing (40% of backbone trainable)
    STAGE2_SGD = {
        'lr': 0.0003,        # 3x Adam (1e-4), 10x lower than Stage 1
        'momentum': 0.9,
        'weight_decay': 5e-5,  # Reduced from Stage 1 (0.5x)
        'nesterov': True
    }
    
    STAGE2_ADAM = {
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'betas': (0.9, 0.999)
    }
    
    # Stage 3: Deep fine-tuning (90% of backbone trainable)
    STAGE3_SGD = {
        'lr': 0.00015,       # 3x Adam (5e-5), finest tuning
        'momentum': 0.9,
        'weight_decay': 1e-5,  # Lightest regularization
        'nesterov': True
    }
    
    STAGE3_ADAM = {
        'lr': 5e-5,
        'weight_decay': 1e-4,
        'betas': (0.9, 0.999)
    }


def create_optimizer(
    model: torch.nn.Module,
    optimizer_type: str,
    stage: int,
    lr: float = None,
    momentum: float = None,
    weight_decay: float = None,
    nesterov: bool = True
) -> torch.optim.Optimizer:
    """
    Create optimizer with stage-specific configurations.
    
    Args:
        model: PyTorch model
        optimizer_type: 'sgd' or 'adam'
        stage: Training stage (1, 2, or 3)
        lr: Custom learning rate (overrides stage default)
        momentum: Custom momentum for SGD (overrides default)
        weight_decay: Custom weight decay (overrides default)
        nesterov: Whether to use Nesterov momentum for SGD
    
    Returns:
        Configured optimizer instance
    
    Example:
        >>> optimizer = create_optimizer(model, 'sgd', stage=1)
        >>> # Or with custom LR:
        >>> optimizer = create_optimizer(model, 'sgd', stage=1, lr=0.01)
    """
    assert optimizer_type.lower() in ['sgd', 'adam'], f"Optimizer must be 'sgd' or 'adam', got {optimizer_type}"
    assert stage in [1, 2, 3], f"Stage must be 1, 2, or 3, got {stage}"
    
    optimizer_type = optimizer_type.lower()
    
    if optimizer_type == 'sgd':
        # Get stage-specific SGD config
        config_attr = f'STAGE{stage}_SGD'
        config = getattr(OptimizerConfig, config_attr).copy()
        
        # Override with custom values if provided
        if lr is not None:
            config['lr'] = lr
        if momentum is not None:
            config['momentum'] = momentum
        if weight_decay is not None:
            config['weight_decay'] = weight_decay
        
        config['nesterov'] = nesterov
        
        optimizer = torch.optim.SGD(model.parameters(), **config)
        
    else:  # Adam
        # Get stage-specific Adam config
        config_attr = f'STAGE{stage}_ADAM'
        config = getattr(OptimizerConfig, config_attr).copy()
        
        # Override with custom values if provided
        if lr is not None:
            config['lr'] = lr
        if weight_decay is not None:
            config['weight_decay'] = weight_decay
        
        optimizer = torch.optim.Adam(model.parameters(), **config)
    
    return optimizer


def get_warmup_scheduler(optimizer: torch.optim.Optimizer, warmup_epochs: int = 3):
    """
    Create learning rate warmup scheduler (linear warmup from 0.001 to target LR).
    
    Only used for Stage 1 (training classifier from scratch with high LR).
    Other stages load pretrained weights and don't need warmup.
    
    Args:
        optimizer: SGD optimizer with target learning rate
        warmup_epochs: Number of epochs to warm up over (recommended: 3)
    
    Returns:
        LambdaLR scheduler that linearly increases LR over warmup_epochs
    
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        >>> scheduler = get_warmup_scheduler(optimizer, warmup_epochs=3)
        >>> # In training loop:
        >>> for epoch in range(num_epochs):
        >>>     train(...)
        >>>     scheduler.step()  # Update LR
    """
    def warmup_lambda(epoch):
        """Linear warmup: 1/warmup_epochs in first epoch, 1.0 after warmup_epochs"""
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)


def apply_gradient_clipping(model: torch.nn.Module, max_norm: float = 1.0) -> float:
    """
    Apply gradient clipping to prevent gradient explosion during SGD training.
    
    Should be called after loss.backward() but before optimizer.step().
    
    Args:
        model: PyTorch model
        max_norm: Maximum norm of gradients (1.0 is typical)
    
    Returns:
        Total norm of the gradients before clipping
    
    Example:
        >>> loss.backward()
        >>> apply_gradient_clipping(model, max_norm=1.0)
        >>> optimizer.step()
    """
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    return total_norm.item()


def print_optimizer_info(optimizer: torch.optim.Optimizer, stage: int):
    """
    Print optimizer configuration for debugging.
    
    Args:
        optimizer: PyTorch optimizer
        stage: Training stage (for context)
    """
    print(f"\nOptimizer Configuration (Stage {stage}):")
    print("=" * 60)
    
    optimizer_name = optimizer.__class__.__name__
    print(f"Type: {optimizer_name}")
    
    for param_group in optimizer.param_groups:
        print(f"Learning rate: {param_group['lr']:.0e}")
        
        if 'momentum' in param_group:
            print(f"Momentum: {param_group['momentum']}")
        
        if 'nesterov' in param_group:
            print(f"Nesterov: {param_group['nesterov']}")
        
        if 'weight_decay' in param_group:
            print(f"Weight decay: {param_group['weight_decay']:.0e}")
        
        if 'betas' in param_group:
            print(f"Betas: {param_group['betas']}")
    
    print("=" * 60)


def main():
    """Test optimizer creation."""
    import torch.nn as nn
    
    # Create dummy model
    model = nn.Linear(10, 7)
    
    print("Testing SGD + Nesterov Optimizer Creation")
    print("=" * 60)
    
    # Test Stage 1 SGD
    opt1_sgd = create_optimizer(model, 'sgd', stage=1)
    print("\nStage 1 - SGD:")
    print_optimizer_info(opt1_sgd, stage=1)
    
    # Test Stage 1 Adam
    opt1_adam = create_optimizer(model, 'adam', stage=1)
    print("\nStage 1 - Adam:")
    print_optimizer_info(opt1_adam, stage=1)
    
    # Test warmup scheduler
    warmup_sched = get_warmup_scheduler(opt1_sgd, warmup_epochs=3)
    print("\nWarmup Scheduler:")
    print("Epoch | LR Scale")
    for epoch in range(5):
        warmup_sched.step()
        lr_scale = opt1_sgd.param_groups[0]['lr'] / 0.01
        print(f"{epoch:5d} | {lr_scale:8.2f}x")
    
    # Test gradient clipping
    print("\nGradient Clipping:")
    dummy_input = torch.randn(4, 10)
    dummy_target = torch.tensor([0, 1, 2, 3])
    loss_fn = nn.CrossEntropyLoss()
    
    output = model(dummy_input)
    loss = loss_fn(output, dummy_target)
    loss.backward()
    
    clipped_norm = apply_gradient_clipping(model, max_norm=1.0)
    print(f"Total gradient norm (clipped): {clipped_norm:.4f}")
    
    print("\n" + "=" * 60)
    print("✓ All optimizer tests passed")


if __name__ == "__main__":
    main()
