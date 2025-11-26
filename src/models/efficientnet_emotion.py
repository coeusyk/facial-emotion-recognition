"""
EfficientNet-B0 Transfer Learning Model for Emotion Recognition
Efficient alternative to VGG16: 5.3M parameters vs 138M (26× smaller)
Expected accuracy: 62-67% on FER2013 dataset
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


def build_emotion_model(num_classes=7, pretrained=True, dropout=0.5, verbose=True):
    """
    Build emotion recognition model using EfficientNet-B0 transfer learning.
    
    EfficientNet-B0 offers excellent efficiency with only 5.3M parameters
    compared to VGG16's 138M, making it ideal for FER2013's 28k training samples.
    
    Architecture:
    - Base: EfficientNet-B0 with ImageNet pretrained weights
    - Input: 48×48 grayscale (1 channel)
    - Feature extractor: Frozen initially for Stage 1 training
    - Classifier: Simple Dropout + Linear (more efficient than VGG16's 3-layer head)
    - Output: 7 emotion classes
    
    Modifications:
    - First conv layer modified to accept 1-channel (grayscale) input
    - RGB pretrained weights averaged for grayscale initialization
    - Custom classification head with configurable dropout
    - Feature extraction layers frozen initially
    
    Args:
        num_classes (int): Number of emotion classes (default: 7)
        pretrained (bool): Whether to use pretrained ImageNet weights (default: True)
        dropout (float): Dropout probability for classifier (default: 0.5)
        verbose (bool): Whether to print model information (default: True)
        
    Returns:
        torch.nn.Module: Modified EfficientNet-B0 model ready for training
    """
    if verbose:
        print("=" * 60)
        print("BUILDING EFFICIENTNET-B0 EMOTION RECOGNITION MODEL")
        print("=" * 60)
    
    # Load pretrained EfficientNet-B0
    if pretrained:
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        if verbose:
            print("\n✓ Loading EfficientNet-B0 with ImageNet pretrained weights")
    else:
        weights = None
        if verbose:
            print("\n✓ Loading EfficientNet-B0 without pretrained weights")
    
    model = models.efficientnet_b0(weights=weights)
    
    if verbose:
        print("✓ EfficientNet-B0 base model loaded")
    
    # Modify first convolutional layer for grayscale input
    # EfficientNet structure: features[0] is a Conv2dNormActivation block
    # features[0][0] is the actual Conv2d layer we need to modify
    # Original: Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
    # New: Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
    
    original_conv = model.features[0][0]
    
    if verbose:
        print(f"\nModifying first conv layer for grayscale input:")
        print(f"  Original: {original_conv}")
    
    # Create new conv layer with 1 input channel
    new_conv = nn.Conv2d(
        in_channels=1,  # Grayscale instead of RGB
        out_channels=32,
        kernel_size=3,
        stride=2,
        padding=1,
        bias=False  # EfficientNet uses bias=False with BatchNorm
    )
    
    # Copy pretrained weights by averaging across RGB channels
    if pretrained:
        # original_conv.weight shape: [32, 3, 3, 3]
        # new_conv.weight shape: [32, 1, 3, 3]
        # Average the weights across the 3 RGB channels
        with torch.no_grad():
            new_conv.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
    
    # Replace the first conv layer (inside Conv2dNormActivation block)
    model.features[0][0] = new_conv
    
    if verbose:
        print(f"  Modified: {new_conv}")
        print("  ✓ Pretrained weights averaged across RGB channels")
    
    # Freeze all feature extraction layers initially
    for param in model.features.parameters():
        param.requires_grad = False
    
    if verbose:
        print("\n✓ All feature extraction layers frozen")
    
    # Get input features to classifier
    # EfficientNet-B0 has 1280 features after adaptive pooling
    num_features = model.classifier[1].in_features
    
    if verbose:
        print(f"\nOriginal classifier input features: {num_features}")
    
    # Replace EfficientNet's classifier with custom head
    # Simple architecture: Dropout + Linear (more efficient than VGG16's 3-layer)
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(num_features, num_classes)
    )
    
    if verbose:
        print("\n✓ Custom classifier head created:")
        print(f"  Dropout(p={dropout})")
        print(f"  Linear({num_features} -> {num_classes}) [Output]")
        print("  (Simpler than VGG16's 3-layer head for better efficiency)")
    
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
        print(f"\nCompared to VGG16:")
        print(f"  EfficientNet-B0: {total_params/1e6:.1f}M parameters")
        print(f"  VGG16: 138M parameters")
        print(f"  Reduction: {138/(total_params/1e6):.1f}× smaller!")
        print(f"{'='*60}")
    
    return model


def unfreeze_layers(model, unfreeze_from_layer=4, verbose=True):
    """
    Unfreeze EfficientNet-B0 feature layers for fine-tuning.
    
    EfficientNet-B0 has 9 main blocks (features[0] through features[8]).
    Typical unfreezing strategy:
    - Stage 1: All frozen (unfreeze_from_layer=9, all layers frozen)
    - Stage 2 Phase 1: Unfreeze last 50% (unfreeze_from_layer=4, blocks 4-8 unfrozen)
    - Stage 2 Phase 2: Unfreeze last 75% (unfreeze_from_layer=2, blocks 2-8 unfrozen)
    
    Args:
        model (torch.nn.Module): The model to unfreeze
        unfreeze_from_layer (int): Block index to start unfreezing from (0-8)
        verbose (bool): Whether to print information
        
    Returns:
        torch.nn.Module: Model with unfrozen layers
    """
    if verbose:
        print("=" * 60)
        print("UNFREEZING LAYERS FOR FINE-TUNING")
        print("=" * 60)
    
    # Get total number of feature blocks (EfficientNet has 9 blocks)
    total_blocks = len(model.features)
    
    if verbose:
        print(f"\nTotal feature blocks: {total_blocks}")
        print(f"Unfreezing from block: {unfreeze_from_layer}")
        print(f"Blocks to unfreeze: {total_blocks - unfreeze_from_layer}")
    
    # Freeze/unfreeze blocks
    for i, block in enumerate(model.features):
        if i >= unfreeze_from_layer:
            # Unfreeze this block
            for param in block.parameters():
                param.requires_grad = True
        else:
            # Keep frozen
            for param in block.parameters():
                param.requires_grad = False
    
    # Count parameters again
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    unfrozen_blocks = total_blocks - unfreeze_from_layer
    
    if verbose:
        print(f"\n✓ Unfrozen {unfrozen_blocks} feature blocks (blocks {unfreeze_from_layer}-{total_blocks-1})")
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
        input_size (tuple): Input tensor size (batch, channels, height, width)
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
    print("EFFICIENTNET-B0 EMOTION RECOGNITION MODEL - ARCHITECTURE TEST")
    print("=" * 60)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Build model
    print("\n" + "="*60)
    print("STAGE 1: BUILDING MODEL (FROZEN FEATURES)")
    print("="*60)
    
    model = build_emotion_model(num_classes=7, pretrained=True, dropout=0.5, verbose=True)
    
    # Print summary
    print_model_summary(model, device=device)
    
    # Test forward pass
    test_model_forward_pass(model, device=device)
    
    # Test unfreezing
    print("\n" + "="*60)
    print("STAGE 2: UNFREEZING LAYERS FOR FINE-TUNING")
    print("="*60)
    
    # Unfreeze last 50% of blocks (EfficientNet has 9 blocks, so unfreeze from block 4)
    model = unfreeze_layers(model, unfreeze_from_layer=4, verbose=True)
    
    # Test forward pass again
    test_model_forward_pass(model, device=device)
    
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE TEST COMPLETE")
    print("="*60)
    print("\nEfficientNet-B0 is ready for training!")
    print("\nKey advantages over VGG16:")
    print("  • 26× fewer parameters (5.3M vs 138M)")
    print("  • Faster training and inference")
    print("  • Better suited for small datasets (28k images)")
    print("  • Expected accuracy: 62-67% (similar to VGG16's 61.5%)")
    print("\nNext step: python scripts/train/train_stage1.py")
    print("="*60)


if __name__ == "__main__":
    main()
