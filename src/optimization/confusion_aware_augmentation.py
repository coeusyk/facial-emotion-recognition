"""
Component 3: Confusion-Targeted Data Augmentation
===================================================

Purpose:
    Apply specialized augmentation to address top confusion pairs from Phase 1.

Top 3 Confusion Pairs:
    1. Fear → Sad (24.3%): Both have downturned mouths, differ in eye openness
    2. Neutral ↔ Sad (19.7% + 18.4%): Minimal expressiveness, subtle differences
    3. Fear → Surprise (15.8%): Both have wide eyes, differ in mouth shape

Strategy:
    Create class-specific augmentation transforms that emphasize the distinguishing
    features for confused emotion pairs.

Expected Gain: +1-2% accuracy, reduced confusion

Author: FER-2013 Optimization Pipeline
"""

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from typing import Dict


def get_confusion_aware_transforms(emotion_name: str, img_size: int = 48):
    """
    Get class-specific augmentation transform for an emotion.
    
    Args:
        emotion_name: Name of the emotion class
        img_size: Target image size
    
    Returns:
        torchvision.transforms.Compose object
    """
    
    # Fear-Sad augmentation: Emphasize eye region differences
    if emotion_name in ['fear']:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(degrees=15),  # Stronger rotation
            transforms.ColorJitter(brightness=0.3, contrast=0.3),  # Emphasize intensity
            transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),  # Focus on upper face
            transforms.RandomHorizontalFlip(p=0.3),  # Reduced (asymmetry helps)
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33)),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
    
    # Neutral-Sad augmentation: Emphasize subtle features
    elif emotion_name in ['neutral', 'sad']:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.15, 0.15),  # Subtle position shifts
                scale=(0.95, 1.05)  # Minimal scaling
            ),
            transforms.ColorJitter(contrast=0.4, saturation=0.2),  # Sharpen contrast
            transforms.CenterCrop(42),  # Focus on lower face (mouth)
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33)),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
    
    # Fear-Surprise augmentation: Emphasize lower face (mouth)
    elif emotion_name in ['surprise']:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop(40),  # Aggressive crop
            transforms.Resize(img_size),
            transforms.RandomRotation(degrees=5),  # Minimal rotation (mouth shape critical)
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33)),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
    
    # Default augmentation for other classes (Angry, Disgust, Happy)
    else:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(degrees=10),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.2, 0.2),
                scale=(0.8, 1.2)
            ),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33)),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])


class ConfusionAwareDataset(Dataset):
    """
    Custom dataset that applies class-specific augmentation strategies.
    """
    
    def __init__(self, data_dir: Path, class_augmentations: Dict = None, img_size: int = 48):
        """
        Args:
            data_dir: Path to training data directory
            class_augmentations: Optional dict of {class_name: transform}
            img_size: Image size
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        
        # Load dataset without transforms first
        self.dataset = datasets.ImageFolder(root=self.data_dir)
        self.class_names = self.dataset.classes
        
        # Create class-specific transforms
        if class_augmentations is None:
            # Use default confusion-aware transforms
            self.class_transforms = {
                class_name: get_confusion_aware_transforms(class_name, img_size)
                for class_name in self.class_names
            }
        else:
            self.class_transforms = class_augmentations
        
        print(f"""\n✓ ConfusionAwareDataset created
  Classes: {self.class_names}
  Samples: {len(self.dataset)}
  Class-specific augmentation: Enabled""")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Get original image and label
        img_path, label = self.dataset.samples[idx]
        img = self.dataset.loader(img_path)
        
        # Get class name
        class_name = self.class_names[label]
        
        # Apply class-specific transform
        transform = self.class_transforms[class_name]
        img = transform(img)
        
        return img, label


def get_confusion_aware_dataloader(
    data_dir: Path,
    batch_size: int = 64,
    num_workers: int = 4,
    img_size: int = 48
) -> DataLoader:
    """
    Create dataloader with confusion-aware augmentation.
    
    Args:
        data_dir: Path to training data
        batch_size: Batch size
        num_workers: Number of worker processes
        img_size: Image size
    
    Returns:
        DataLoader with confusion-aware augmentation
    
    Example:
        >>> train_loader = get_confusion_aware_dataloader('data/raw/train')
        >>> # Use this instead of regular dataloader for training
    """
    dataset = ConfusionAwareDataset(data_dir, img_size=img_size)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    print(f"""\n✓ Confusion-aware DataLoader created:
  Batch size: {batch_size}
  Number of batches: {len(dataloader)}
  Specialized augmentation for: Fear, Sad, Neutral, Surprise""")
    
    return dataloader


def main():
    """Example usage of confusion-aware augmentation."""
    
    DATA_DIR = Path('data/raw/train')
    
    if not DATA_DIR.exists():
        print(f"""Error: Data directory not found: {DATA_DIR}
Please update DATA_DIR in this script""")
        return
    
    print(f"""{'='*80}
COMPONENT 3: CONFUSION-AWARE AUGMENTATION""")
    print("=" * 80)
    
    # Create confusion-aware dataloader
    train_loader = get_confusion_aware_dataloader(
        data_dir=DATA_DIR,
        batch_size=64,
        num_workers=4
    )
    
    # Test by getting a batch
    images, labels = next(iter(train_loader))
    
    print(f"""\nSample batch:
  Images shape: {images.shape}
  Labels shape: {labels.shape}
  Pixel range: [{images.min():.3f}, {images.max():.3f}]""")
    
    print(f"""\n{'='*80}
CONFUSION-AWARE AUGMENTATION READY
{'='*80}

Usage in training:
  from src.optimization.confusion_aware_augmentation import get_confusion_aware_dataloader
  train_loader = get_confusion_aware_dataloader('data/raw/train')
  # Use this instead of regular create_dataloaders()
{'='*80}""")


if __name__ == '__main__':
    main()
