"""
Part 3: Transfer Learning Model Architecture
Build VGG16-based emotion recognition model with custom classifier.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import VGG16_Weights


def build_emotion_model(num_classes=7, pretrained=True, verbose=True):
    """
    Build emotion recognition model using VGG16 transfer learning.
    
    Modifications:
    - First conv layer modified to accept 1-channel (grayscale) input
    - Custom classification head with dropout and batch normalization
    - Feature extraction layers frozen initially
    
    Args:
        num_classes (int): Number of emotion classes
        pretrained (bool): Whether to use pretrained ImageNet weights
        verbose (bool): Whether to print model information
        
    Returns:
        torch.nn.Module: Modified VGG16 model
    """
    if verbose:
        print("=" * 60)
        print("BUILDING VGG16 EMOTION RECOGNITION MODEL")
        print("=" * 60)
    
    # Load pretrained VGG16
    if pretrained:
        weights = VGG16_Weights.IMAGENET1K_V1
        if verbose:
            print("\n✓ Loading VGG16 with ImageNet pretrained weights")
    else:
        weights = None
        if verbose:
            print("\n✓ Loading VGG16 without pretrained weights")
    
    model = models.vgg16(weights=weights)
    
    if verbose:
        print("✓ VGG16 base model loaded")
    
    # Modify first convolutional layer for grayscale input
    # Original: Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
    # New: Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
    
    original_conv = model.features[0]
    
    if verbose:
        print(f"\nModifying first conv layer for grayscale input:")
        print(f"  Original: {original_conv}")
    
    # Create new conv layer with 1 input channel
    new_conv = nn.Conv2d(
        in_channels=1,  # Grayscale instead of RGB
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )
    
    # Copy pretrained weights by averaging across RGB channels
    if pretrained:
        # original_conv.weight shape: [64, 3, 3, 3]
        # new_conv.weight shape: [64, 1, 3, 3]
        # Average the weights across the 3 RGB channels
        with torch.no_grad():
            new_conv.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
    
    # Replace the first conv layer
    model.features[0] = new_conv
    
    if verbose:
        print(f"  Modified: {new_conv}")
        print("  ✓ Pretrained weights averaged across RGB channels")
    
    # Freeze all feature extraction layers initially
    for param in model.features.parameters():
        param.requires_grad = False
    
    if verbose:
        print("\n✓ All feature extraction layers frozen")
    
    # Get input features to classifier
    num_features = model.classifier[0].in_features
    
    if verbose:
        print(f"\nOriginal classifier input features: {num_features}")
    
    # Replace VGG16's classifier with custom classification head
    model.classifier = nn.Sequential(
        # First dense layer
        nn.Linear(num_features, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),
        
        # Second dense layer
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(256),
        nn.Dropout(0.5),
        
        # Output layer
        nn.Linear(256, num_classes)
    )
    
    if verbose:
        print("\n✓ Custom classifier head created:")
        print("  Layer 1: Linear(25088 -> 512) + ReLU + BatchNorm + Dropout(0.5)")
        print("  Layer 2: Linear(512 -> 256) + ReLU + BatchNorm + Dropout(0.5)")
        print(f"  Layer 3: Linear(256 -> {num_classes}) [Output]")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    if verbose:
        print(f"\n{'='*60}")
        print("MODEL PARAMETER SUMMARY")
        print(f"{'='*60}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        print(f"Frozen parameters: {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")
        print(f"{'='*60}")
    
    return model


def unfreeze_layers(model, unfreeze_from_layer=10, verbose=True):
    """
    Unfreeze VGG16 feature layers for fine-tuning.
    
    Args:
        model (torch.nn.Module): The model to unfreeze
        unfreeze_from_layer (int): Index to start unfreezing from (0-based)
        verbose (bool): Whether to print information
        
    Returns:
        torch.nn.Module: Model with unfrozen layers
    """
    if verbose:
        print("=" * 60)
        print("UNFREEZING LAYERS FOR FINE-TUNING")
        print("=" * 60)
    
    # Get total number of feature layers
    total_feature_layers = len(list(model.features.parameters()))
    
    if verbose:
        print(f"\nTotal feature layers: {total_feature_layers}")
        print(f"Unfreezing from layer: {unfreeze_from_layer}")
    
    # Unfreeze layers from specified index onwards
    for i, param in enumerate(model.features.parameters()):
        if i >= unfreeze_from_layer:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    # Count parameters again
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    unfrozen_layers = sum(1 for p in model.features.parameters() if p.requires_grad)
    
    if verbose:
        print(f"\n✓ Unfrozen {unfrozen_layers} feature layers")
        print(f"\n{'='*60}")
        print("UPDATED PARAMETER SUMMARY")
        print(f"{'='*60}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        print(f"Frozen parameters: {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")
        print(f"{'='*60}")
    
    return model


def unfreeze_vgg16_blocks(model, blocks_to_unfreeze=[4, 5], verbose=True):
    """
    Unfreeze specific VGG16 blocks for progressive fine-tuning.
    
    VGG16 Block Structure:
        Block 1: features[0-4]   - Early edges/textures
        Block 2: features[5-9]   - Basic patterns  
        Block 3: features[10-16] - Mid-level features
        Block 4: features[17-23] - Complex features
        Block 5: features[24-30] - High-level features
    
    Args:
        model (torch.nn.Module): VGG16 model to unfreeze
        blocks_to_unfreeze (list): Which blocks to unfreeze (1-5)
        verbose (bool): Whether to print information
        
    Returns:
        torch.nn.Module: Model with specified blocks unfrozen
    
    Examples:
        # Stage 2: Unfreeze last 2 blocks
        model = unfreeze_vgg16_blocks(model, blocks_to_unfreeze=[4, 5])
        
        # Stage 3: Unfreeze blocks 2-5, keep block 1 frozen
        model = unfreeze_vgg16_blocks(model, blocks_to_unfreeze=[2, 3, 4, 5])
    """
    # VGG16 block boundaries (layer indices in features)
    block_ranges = {
        1: (0, 5),    # Block 1: layers 0-4
        2: (5, 10),   # Block 2: layers 5-9
        3: (10, 17),  # Block 3: layers 10-16
        4: (17, 24),  # Block 4: layers 17-23
        5: (24, 31),  # Block 5: layers 24-30
    }
    
    if verbose:
        print("=" * 60)
        print("UNFREEZING VGG16 BLOCKS FOR PROGRESSIVE FINE-TUNING")
        print("=" * 60)
        print(f"\nBlocks to unfreeze: {blocks_to_unfreeze}")
        print("\nVGG16 Block Structure:")
        for block_num, (start, end) in block_ranges.items():
            status = "UNFROZEN" if block_num in blocks_to_unfreeze else "FROZEN"
            print(f"  Block {block_num}: features[{start:2d}-{end-1:2d}] - {status}")
    
    # First, freeze all feature layers
    for param in model.features.parameters():
        param.requires_grad = False
    
    # Then unfreeze specified blocks
    for block_num in blocks_to_unfreeze:
        if block_num not in block_ranges:
            raise ValueError(f"Invalid block number: {block_num}. Must be 1-5.")
        
        start_idx, end_idx = block_ranges[block_num]
        
        # Unfreeze parameters in this block
        for i, param in enumerate(model.features.parameters()):
            if start_idx <= i < end_idx:
                param.requires_grad = True
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    unfrozen_feature_layers = sum(1 for p in model.features.parameters() if p.requires_grad)
    total_feature_layers = len(list(model.features.parameters()))
    
    if verbose:
        print(f"\n✓ Unfrozen {unfrozen_feature_layers}/{total_feature_layers} feature layers")
        print(f"\n{'='*60}")
        print("UPDATED PARAMETER SUMMARY")
        print(f"{'='*60}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        print(f"Frozen parameters: {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")
        print(f"{'='*60}")
    
    return model


def print_model_summary(model, input_size=(1, 1, 48, 48), device='cpu'):
    """
    Print detailed model summary.
    
    Args:
        model (torch.nn.Module): Model to summarize
        input_size (tuple): Input tensor size (batch, channels, height, width)
        device (str): Device to run model on
    """
    try:
        from torchsummary import summary
        print(f"\n{'='*60}")
        print("DETAILED MODEL SUMMARY")
        print(f"{'='*60}")
        model = model.to(device)
        summary(model, input_size[1:])  # torchsummary expects (C, H, W)
    except ImportError:
        print("\n⚠ torchsummary not installed. Install with: pip install torchsummary")
        print("Displaying basic model structure:\n")
        print(model)


def test_model_forward_pass(model, input_size=(1, 1, 48, 48), device='cpu'):
    """
    Test model with a dummy forward pass.
    
    Args:
        model (torch.nn.Module): Model to test
        input_size (tuple): Input tensor size
        device (str): Device to run model on
    """
    print(f"\n{'='*60}")
    print("TESTING MODEL FORWARD PASS")
    print(f"{'='*60}")
    
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_size).to(device)
    
    print(f"\nInput shape: {dummy_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Output values (logits): {output[0]}")
    
    # Apply softmax to get probabilities
    probs = torch.softmax(output, dim=1)
    print(f"Output probabilities: {probs[0]}")
    print(f"Sum of probabilities: {probs[0].sum():.6f}")
    
    # Get predicted class
    pred_class = torch.argmax(output, dim=1)
    print(f"Predicted class: {pred_class.item()}")
    
    print("\n✓ Forward pass successful!")


def main():
    """Main function to test model architecture."""
    print("=" * 60)
    print("VGG16 EMOTION RECOGNITION MODEL - ARCHITECTURE TEST")
    print("=" * 60)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Build model
    print("\n" + "="*60)
    print("STAGE 1: BUILDING MODEL (FROZEN FEATURES)")
    print("="*60)
    
    model = build_emotion_model(num_classes=7, pretrained=True, verbose=True)
    
    # Print summary
    print_model_summary(model, device=device)
    
    # Test forward pass
    test_model_forward_pass(model, device=device)
    
    # Test unfreezing
    print("\n" + "="*60)
    print("STAGE 2: UNFREEZING LAYERS FOR FINE-TUNING")
    print("="*60)
    
    # Unfreeze last 50% of layers (VGG16 has ~26 layers, so unfreeze from layer 13)
    model = unfreeze_layers(model, unfreeze_from_layer=13, verbose=True)
    
    # Test forward pass again
    test_model_forward_pass(model, device=device)
    
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE TEST COMPLETE")
    print("="*60)
    print("\nThe model is ready for training!")
    print("Next step: python train_emotion_model.py")
    print("="*60)


if __name__ == "__main__":
    main()
