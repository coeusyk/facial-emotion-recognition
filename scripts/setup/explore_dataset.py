"""
Dataset exploration script for Facial Emotion Recognition project.
Analyzes and visualizes the FER-2013 dataset structure and content.
"""

import os
import sys
from pathlib import Path
import random

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def check_dataset_exists(base_path="data/raw"):
    """Check if dataset has been downloaded."""
    base_path = Path(base_path)
    
    if not base_path.exists():
        print(f"✗ Dataset directory not found: {base_path}")
        print("\nPlease run: python scripts/setup/download_dataset.py")
        return False
    
    train_path = base_path / "train"
    test_path = base_path / "test"
    
    if not train_path.exists() or not test_path.exists():
        print("✗ Dataset not properly downloaded (missing train/test directories)")
        print("\nPlease run: python scripts/setup/download_dataset.py")
        return False
    
    return True


def count_images_per_class(base_path="data/raw"):
    """Count images for each emotion class in train and test sets."""
    base_path = Path(base_path)
    emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    
    stats = {
        "train": {},
        "test": {}
    }
    
    print("=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    print()
    
    for split in ["train", "test"]:
        split_path = base_path / split
        print(f"{split.upper()} SET:")
        
        total = 0
        for emotion in emotions:
            emotion_path = split_path / emotion
            
            if emotion_path.exists():
                # Count image files
                images = list(emotion_path.glob("*.jpg")) + \
                        list(emotion_path.glob("*.png"))
                count = len(images)
                stats[split][emotion] = count
                total += count
                print(f"  {emotion:10s}: {count:5d} images")
            else:
                stats[split][emotion] = 0
                print(f"  {emotion:10s}: 0 images (missing!)")
        
        stats[split]["total"] = total
        print(f"  {'TOTAL':10s}: {total:5d} images")
        print()
    
    return stats


def plot_distribution(stats, output_dir="results"):
    """Create bar chart showing emotion distribution."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    
    # Prepare data
    train_counts = [stats["train"].get(emotion, 0) for emotion in emotions]
    test_counts = [stats["test"].get(emotion, 0) for emotion in emotions]
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Colors for each emotion
    colors = ['#e74c3c', '#9b59b6', '#34495e', '#f39c12', 
              '#95a5a6', '#3498db', '#2ecc71']
    
    # Plot training set
    axes[0].bar(emotions, train_counts, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_title('Training Set Distribution', fontsize=16, fontweight='bold')
    axes[0].set_xlabel('Emotion', fontsize=12)
    axes[0].set_ylabel('Number of Images', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(train_counts):
        axes[0].text(i, v + 50, str(v), ha='center', va='bottom', fontweight='bold')
    
    # Plot test set
    axes[1].bar(emotions, test_counts, color=colors, alpha=0.8, edgecolor='black')
    axes[1].set_title('Test Set Distribution', fontsize=16, fontweight='bold')
    axes[1].set_xlabel('Emotion', fontsize=12)
    axes[1].set_ylabel('Number of Images', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(test_counts):
        axes[1].text(i, v + 50, str(v), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / "emotion_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved distribution chart: {output_path}")
    plt.close()


def plot_sample_images(base_path="data/raw", output_dir="results", num_samples=5):
    """Create grid of sample images from each emotion class."""
    base_path = Path(base_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    
    # Create figure
    fig, axes = plt.subplots(len(emotions), num_samples, figsize=(15, 2.5 * len(emotions)))
    fig.suptitle('Sample Images from Training Set', fontsize=18, fontweight='bold', y=0.995)
    
    for i, emotion in enumerate(emotions):
        emotion_path = base_path / "train" / emotion
        
        if not emotion_path.exists():
            continue
        
        # Get random sample images
        image_files = list(emotion_path.glob("*.jpg")) + \
                     list(emotion_path.glob("*.png"))
        
        if len(image_files) == 0:
            continue
        
        # Sample random images
        sample_files = random.sample(image_files, min(num_samples, len(image_files)))
        
        for j, img_file in enumerate(sample_files):
            try:
                img = Image.open(img_file).convert('L')  # Convert to grayscale
                
                axes[i, j].imshow(img, cmap='gray')
                axes[i, j].axis('off')
                
                # Add emotion label to first column
                if j == 0:
                    axes[i, j].set_ylabel(emotion.upper(), 
                                         fontsize=12, 
                                         fontweight='bold',
                                         rotation=0,
                                         ha='right',
                                         va='center')
            except Exception as e:
                print(f"  Warning: Could not load {img_file.name}: {e}")
                axes[i, j].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / "sample_images.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved sample images: {output_path}")
    plt.close()


def plot_class_balance(stats, output_dir="results"):
    """Create visualization showing class imbalance."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    train_counts = [stats["train"].get(emotion, 0) for emotion in emotions]
    
    # Calculate percentages
    total = sum(train_counts)
    percentages = [(count / total * 100) if total > 0 else 0 for count in train_counts]
    
    # Create pie chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Pie chart
    colors = ['#e74c3c', '#9b59b6', '#34495e', '#f39c12', 
              '#95a5a6', '#3498db', '#2ecc71']
    
    wedges, texts, autotexts = ax1.pie(
        train_counts,
        labels=emotions,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )
    ax1.set_title('Training Set Class Distribution', fontsize=16, fontweight='bold')
    
    # Horizontal bar chart showing imbalance
    sorted_idx = np.argsort(train_counts)
    sorted_emotions = [emotions[i] for i in sorted_idx]
    sorted_counts = [train_counts[i] for i in sorted_idx]
    sorted_colors = [colors[i] for i in sorted_idx]
    
    y_pos = np.arange(len(sorted_emotions))
    ax2.barh(y_pos, sorted_counts, color=sorted_colors, alpha=0.8, edgecolor='black')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(sorted_emotions)
    ax2.set_xlabel('Number of Images', fontsize=12)
    ax2.set_title('Class Imbalance Analysis', fontsize=16, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(sorted_counts):
        ax2.text(v + 50, i, str(v), va='center', fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / "class_balance.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved class balance chart: {output_path}")
    plt.close()


def print_summary(stats):
    """Print comprehensive dataset summary."""
    print("=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    print()
    
    total_train = stats["train"]["total"]
    total_test = stats["test"]["total"]
    total_all = total_train + total_test
    
    print(f"Total images: {total_all:,}")
    print(f"  Training:   {total_train:,} ({total_train/total_all*100:.1f}%)")
    print(f"  Test:       {total_test:,} ({total_test/total_all*100:.1f}%)")
    print()
    
    # Class imbalance
    emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    train_counts = [stats["train"].get(emotion, 0) for emotion in emotions]
    
    if total_train > 0:
        max_class = emotions[np.argmax(train_counts)]
        min_class = emotions[np.argmin(train_counts)]
        max_count = max(train_counts)
        min_count = min(train_counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print("Class balance:")
        print(f"  Most common:  {max_class} ({max_count} images)")
        print(f"  Least common: {min_class} ({min_count} images)")
        print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")
        print()
    
    print("Image specifications:")
    print("  Format: Grayscale (1 channel)")
    print("  Size: 48x48 pixels")
    print("  Type: JPG/PNG")
    print()


def main():
    """Main exploration routine."""
    print("=" * 60)
    print("FER-2013 Dataset Explorer")
    print("=" * 60)
    print()
    
    # Check if dataset exists
    if not check_dataset_exists():
        return False
    
    print("✓ Dataset found!")
    print()
    
    # Count images per class
    stats = count_images_per_class()
    
    # Print summary
    print_summary(stats)
    
    # Create visualizations
    print("=" * 60)
    print("Creating Visualizations...")
    print("=" * 60)
    print()
    
    try:
        plot_distribution(stats)
        plot_class_balance(stats)
        plot_sample_images()
        
        print()
        print("=" * 60)
        print("✓ Exploration Complete!")
        print("=" * 60)
        print("\nGenerated files:")
        print("  results/emotion_distribution.png")
        print("  results/class_balance.png")
        print("  results/sample_images.png")
        print("\nNext step: Run python scripts/train_stage1_warmup.py")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Visualization failed: {e}")
        print("\nMake sure matplotlib, seaborn, and PIL are installed:")
        print("  pip install matplotlib seaborn pillow")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
