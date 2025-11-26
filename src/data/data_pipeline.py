"""
Part 2: Data Pipeline with PyTorch
Create data transformation pipelines and DataLoaders for training and testing.
"""

import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_train_transforms(img_size=48, normalize_type='imagenet'):
    """
    Create training data transformation pipeline with augmentation.
    Based on state-of-the-art FER2013 preprocessing techniques:
    - Khaireddin et al. (2021): "Facial Emotion Recognition: State of the Art Performance on FER2013"
    - Random rescaling (±20%), rotation (±10°), shifts (±20%)
    - Random erasing with 50% probability
    - Horizontal flips
    
    Args:
        img_size (int): Target image size (width and height)
        normalize_type (str): Normalization type:
            - 'imagenet': ImageNet-style normalization (mean=[0.485], std=[0.229]) - RECOMMENDED for VGG16
            - 'zero_one': [0,1] range (ToTensor default)
            - 'neg_one_one': [-1,1] range
        
    Returns:
        torchvision.transforms.Compose: Training transformation pipeline
    """
    transform_list = [
        # Ensure grayscale (1 channel)
        transforms.Grayscale(num_output_channels=1),
        
        # Resize to target size
        transforms.Resize((img_size, img_size)),
        
        # Data augmentation transforms (based on FER2013 best practices)
        transforms.RandomRotation(degrees=10),  # Rotate ±10 degrees (research-backed)
        transforms.RandomHorizontalFlip(p=0.5),  # 50% chance to flip
        
        # Random rescaling and shifts (±20% as per research)
        transforms.RandomAffine(
            degrees=0,  # No rotation (already done above)
            translate=(0.2, 0.2),  # Shift up to 20% in x and y (research: ±20%)
            scale=(0.8, 1.2),  # Random rescaling ±20% (research-backed)
            shear=None   # No shearing
        ),
        
        # Convert to tensor (automatically scales to [0, 1])
        transforms.ToTensor(),
        
        # Random erasing (50% probability) - research-backed augmentation
        transforms.RandomErasing(
            p=0.5,  # 50% probability
            scale=(0.02, 0.33),  # Erase 2-33% of image
            ratio=(0.3, 3.3),  # Aspect ratio
            value=0,  # Fill with black
            inplace=False
        ),
    ]
    
    # Add normalization based on type
    if normalize_type == 'imagenet':
        # ImageNet-style normalization for VGG16 transfer learning
        # Uses grayscale equivalent of ImageNet RGB normalization
        # This aligns with VGG16's pretrained weights for better feature extraction
        transform_list.append(transforms.Normalize(mean=[0.485], std=[0.229]))
    elif normalize_type == 'zero_one':
        # No normalization needed - ToTensor already maps to [0, 1]
        pass
    elif normalize_type == 'neg_one_one':
        # Normalize to [-1, 1] range
        transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    else:
        raise ValueError(f"normalize_type must be 'imagenet', 'zero_one' or 'neg_one_one', got {normalize_type}")
    
    train_transforms = transforms.Compose(transform_list)
    
    return train_transforms


def get_test_transforms(img_size=48, normalize_type='imagenet'):
    """
    Create test/validation data transformation pipeline (NO augmentation).
    
    Args:
        img_size (int): Target image size (width and height)
        normalize_type (str): Normalization type:
            - 'imagenet': ImageNet-style normalization (mean=[0.485], std=[0.229]) - RECOMMENDED for VGG16
            - 'zero_one': [0,1] range (ToTensor default)
            - 'neg_one_one': [-1,1] range
        
    Returns:
        torchvision.transforms.Compose: Test transformation pipeline
    """
    transform_list = [
        # Ensure grayscale (1 channel)
        transforms.Grayscale(num_output_channels=1),
        
        # Resize to target size
        transforms.Resize((img_size, img_size)),
        
        # Convert to tensor
        transforms.ToTensor(),
    ]
    
    # Add normalization based on type (must match training)
    if normalize_type == 'imagenet':
        # ImageNet-style normalization for VGG16 transfer learning
        # Uses grayscale equivalent of ImageNet RGB normalization
        transform_list.append(transforms.Normalize(mean=[0.485], std=[0.229]))
    elif normalize_type == 'zero_one':
        # No normalization needed - ToTensor already maps to [0, 1]
        pass
    elif normalize_type == 'neg_one_one':
        # Normalize to [-1, 1] range
        transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    else:
        raise ValueError(f"normalize_type must be 'imagenet', 'zero_one' or 'neg_one_one', got {normalize_type}")
    
    test_transforms = transforms.Compose(transform_list)
    
    return test_transforms


def create_dataloaders(data_dir, batch_size=64, img_size=48, num_workers=4, val_split=0.2, normalize_type='imagenet'):
    """
    Create PyTorch DataLoaders for training and testing.
    
    Args:
        data_dir (str): Path to data directory (should contain 'train' and 'test' subdirectories)
        batch_size (int): Batch size for DataLoader
        img_size (int): Target image size
        num_workers (int): Number of worker processes for data loading
        val_split (float): Fraction of training data to use for validation (0.0-1.0)
        normalize_type (str): Normalization type:
            - 'imagenet': ImageNet-style (mean=[0.485], std=[0.229]) - DEFAULT for VGG16 transfer learning
            - 'zero_one': [0,1] range
            - 'neg_one_one': [-1,1] range
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, class_names)
    """
    print("=" * 60)
    print("CREATING PYTORCH DATALOADERS")
    print("=" * 60)
    
    # Define paths
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    # Check if directories exist
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    
    # Get transforms with specified normalization type
    train_transforms = get_train_transforms(img_size, normalize_type=normalize_type)
    test_transforms = get_test_transforms(img_size, normalize_type=normalize_type)
    
    print(f"\nPreprocessing configuration:")
    print(f"  Normalization: {normalize_type}")
    print(f"  Image size: {img_size}x{img_size}")
    print(f"  Augmentation: Random rotation (±10°), rescaling (±20%), shifts (±20%), random erasing (50%)")
    
    print(f"\n{'='*60}")
    print("LOADING DATASETS")
    print(f"{'='*60}")
    
    # Load full training dataset
    full_train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=train_transforms
    )
    
    # Get class names
    class_names = full_train_dataset.classes
    num_classes = len(class_names)
    
    print(f"\nDataset loaded from: {train_dir}")
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    print(f"Total training images: {len(full_train_dataset)}")
    
    # Split training data into train and validation
    if val_split > 0:
        train_size = int((1 - val_split) * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        
        # Get indices for split (not the data itself)
        train_indices, val_indices = torch.utils.data.random_split(
            range(len(full_train_dataset)),
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create train dataset with augmentation (from full_train_dataset)
        train_dataset = torch.utils.data.Subset(
            full_train_dataset,  # Uses train_transforms with augmentation
            train_indices
        )
        
        # Create validation dataset WITHOUT augmentation
        val_dataset_base = datasets.ImageFolder(
            root=train_dir,
            transform=test_transforms  # NO augmentation
        )
        val_dataset = torch.utils.data.Subset(
            val_dataset_base,
            val_indices  # Use the VALIDATION indices (20%)
        )
        
        print(f"\nTraining set size: {train_size} ({(1-val_split)*100:.0f}%)")
        print(f"Validation set size: {val_size} ({val_split*100:.0f}%)")
    else:
        train_dataset = full_train_dataset
        val_dataset = None
        print(f"\nNo validation split. Using full training set: {len(train_dataset)}")
    
    # Load test dataset
    test_dataset = None
    if os.path.exists(test_dir):
        test_dataset = datasets.ImageFolder(
            root=test_dir,
            transform=test_transforms
        )
        print(f"Test set size: {len(test_dataset)}")
    else:
        print(f"⚠ Test directory not found: {test_dir}")
    
    print(f"\n{'='*60}")
    print("CREATING DATALOADERS")
    print(f"{'='*60}")
    
    # Create DataLoader for training
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        pin_memory=True,  # Faster GPU transfer
        persistent_workers=True if num_workers > 0 else False  # Keep workers alive
    )
    
    print(f"\n✓ Training DataLoader created:")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of batches: {len(train_loader)}")
    print(f"  Shuffle: True")
    print(f"  Augmentation: Enabled")
    
    # Create DataLoader for validation
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle validation data
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )
        print(f"\n✓ Validation DataLoader created:")
        print(f"  Batch size: {batch_size}")
        print(f"  Number of batches: {len(val_loader)}")
        print(f"  Shuffle: False")
        print(f"  Augmentation: Disabled")
    
    # Create DataLoader for testing
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )
        print(f"\n✓ Test DataLoader created:")
        print(f"  Batch size: {batch_size}")
        print(f"  Number of batches: {len(test_loader)}")
        print(f"  Shuffle: False")
        print(f"  Augmentation: Disabled")
    
    print(f"\n{'='*60}")
    print("SAMPLE BATCH INFORMATION")
    print(f"{'='*60}")
    
    # Get a sample batch
    sample_images, sample_labels = next(iter(train_loader))
    print(f"\nSample batch shape: {sample_images.shape}")
    print(f"  Batch size: {sample_images.shape[0]}")
    print(f"  Channels: {sample_images.shape[1]} (grayscale)")
    print(f"  Height: {sample_images.shape[2]}")
    print(f"  Width: {sample_images.shape[3]}")
    print(f"\nSample labels shape: {sample_labels.shape}")
    print(f"Pixel value range: [{sample_images.min():.3f}, {sample_images.max():.3f}]")
    print(f"Label range: [{sample_labels.min()}, {sample_labels.max()}]")
    
    # Class distribution in the batch
    unique, counts = torch.unique(sample_labels, return_counts=True)
    print(f"\nClass distribution in sample batch:")
    for class_idx, count in zip(unique, counts):
        print(f"  {class_names[class_idx]:12s}: {count} images")
    
    print(f"\n{'='*60}")
    print("DATALOADER CREATION COMPLETE")
    print(f"{'='*60}")
    
    return train_loader, val_loader, test_loader, class_names


def calculate_class_weights(data_dir, beta=0.9999):
    """
    Calculate class weights using Effective Number of Samples.
    More stable than inverse frequency for severe imbalance (16:1 ratio).
    
    Based on: Cui et al. (2019) "Class-Balanced Loss Based on Effective Number of Samples"
    
    Args:
        data_dir: Path to training data
        beta: Smoothing parameter (0.999-0.9999). Higher = more conservative.
    
    Returns:
        torch.Tensor: Normalized class weights
    """
    import numpy as np
    from collections import Counter
    
    dataset = datasets.ImageFolder(root=data_dir)
    labels = [label for _, label in dataset.samples]
    class_counts = Counter(labels)
    
    samples_per_class = np.array([class_counts[i] for i in range(len(class_counts))])
    
    # Effective number formula
    effective_num = 1.0 - np.power(beta, samples_per_class)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * len(class_counts)  # Normalize
    
    weights_tensor = torch.FloatTensor(weights)
    
    print("\nClass weights (Effective Number, beta={:.4f}):".format(beta))
    for i, (class_name, weight) in enumerate(zip(dataset.classes, weights_tensor)):
        count = class_counts[i]
        print(f"  {class_name:12s}: {weight:5.2f} (samples: {count:5d})")
    
    return weights_tensor


def calculate_class_weights_custom_boost(data_dir, beta=0.9999, boost_classes=None):
    """
    Calculate class weights with manual boost for specific emotions.
    
    Useful for fine-tuning per-class performance when standard Effective Number
    weighting doesn't provide enough focus on underperforming minority classes.
    
    Args:
        data_dir: Path to training data
        beta: Effective number beta (0.9999 recommended)
        boost_classes: Dict of {emotion_name: boost_factor}
                       e.g., {'angry': 1.5, 'fear': 1.5}
    
    Returns:
        torch.Tensor: Adjusted class weights
    
    Example:
        >>> boost_config = {'angry': 1.5, 'fear': 1.5}
        >>> weights = calculate_class_weights_custom_boost(
        ...     'data/raw/train',
        ...     boost_classes=boost_config
        ... )
    """
    import numpy as np
    from collections import Counter
    
    dataset = datasets.ImageFolder(root=data_dir)
    labels = [label for _, label in dataset.samples]
    class_counts = Counter(labels)
    
    samples_per_class = np.array([class_counts[i] for i in range(len(class_counts))])
    
    # Effective number base weights
    effective_num = 1.0 - np.power(beta, samples_per_class)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * len(class_counts)
    
    print("\nClass weights (Effective Number base):")
    for i, (class_name, weight) in enumerate(zip(dataset.classes, weights)):
        count = class_counts[i]
        print(f"  {class_name:12s}: {weight:5.2f} (samples: {count:5d})")
    
    # Apply manual boosts
    if boost_classes:
        print(f"\nApplying custom boosts:")
        emotion_to_idx = {name: idx for idx, name in enumerate(dataset.classes)}
        for emotion, boost_factor in boost_classes.items():
            if emotion in emotion_to_idx:
                idx = emotion_to_idx[emotion]
                old_weight = weights[idx]
                weights[idx] *= boost_factor
                print(f"  ✓ {emotion:12s}: {old_weight:.2f} → {weights[idx]:.2f} ({boost_factor}x boost)")
            else:
                print(f"  ✗ {emotion}: not found in dataset classes")
    
    weights_tensor = torch.FloatTensor(weights)
    
    print("\nFinal class weights after boosts:")
    for i, (class_name, weight) in enumerate(zip(dataset.classes, weights_tensor)):
        count = class_counts[i]
        print(f"  {class_name:12s}: {weight:5.2f} (samples: {count:5d})")
    
    return weights_tensor


def main():
    """Main function to test data pipeline."""
    # Configuration
    DATA_DIR = "data/raw"
    BATCH_SIZE = 64
    IMG_SIZE = 48
    NUM_WORKERS = 4
    VAL_SPLIT = 0.2
    
    # Create dataloaders
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        num_workers=NUM_WORKERS,
        val_split=VAL_SPLIT
    )
    
    # Calculate class weights
    print(f"\n{'='*60}")
    print("CALCULATING CLASS WEIGHTS")
    print(f"{'='*60}")
    
    train_dir = os.path.join(DATA_DIR, 'train')
    class_weights = calculate_class_weights(train_dir)
    
    print(f"\n{'='*60}")
    print("DATA PIPELINE TEST COMPLETE")
    print(f"{'='*60}")
    print("\nYou can now use these dataloaders for training:")
    print("  - train_loader: For training with augmentation")
    print("  - val_loader: For validation without augmentation")
    print("  - test_loader: For final evaluation")
    print("  - class_weights: For weighted loss function")
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
