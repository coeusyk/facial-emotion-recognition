"""
Visual Validation Script for Preprocessing Pipeline
===================================================

Generates side-by-side comparison images to validate preprocessing effectiveness:
- Original image (left) vs Preprocessed image (right)
- Statistical metrics for each image
- Visual checklist validation

Success criteria:
✓ Edges are sharper (eyebrows, mouth, nose more defined)
✓ Dark images are brightened without washing out
✓ Bright images are not overexposed
✓ No halos or artifacts around edges
✓ Facial features remain recognizable

Usage:
    python scripts/validate_preprocessing.py [--num-samples 20] [--output-dir artifacts/preprocessing_validation]
"""

import sys
import argparse
from pathlib import Path
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.preprocessing import preprocess_fer2013_image, get_preprocessing_stats
from config import Config


def create_comparison_image(img_before, img_after, stats, img_name, emotion):
    """
    Create a side-by-side comparison with statistics.
    
    Args:
        img_before (np.ndarray): Original image
        img_after (np.ndarray): Preprocessed image
        stats (dict): Statistical metrics
        img_name (str): Image filename
        emotion (str): Emotion label
    
    Returns:
        matplotlib.figure.Figure: Comparison figure
    """
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[1, 1])
    
    # Original image (left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_before, cmap='gray', vmin=0, vmax=255)
    ax1.set_title(f"Original Image\n{emotion.upper()}", fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Preprocessed image (right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(img_after, cmap='gray', vmin=0, vmax=255)
    ax2.set_title("Preprocessed Image\n(Unsharp Mask + CLAHE)", fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Statistics (left bottom)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis('off')
    
    stats_text = f"""
    PIXEL STATISTICS
    
    Mean:       {stats['before_mean']:.1f} → {stats['after_mean']:.1f}
    Std Dev:    {stats['before_std']:.1f} → {stats['after_std']:.1f} ({stats['std_increase_percent']:+.1f}%)
    Range:      [{stats['before_min']}, {stats['before_max']}] → [{stats['after_min']}, {stats['after_max']}]
    """
    ax3.text(0.1, 0.5, stats_text, fontsize=10, family='monospace', verticalalignment='center')
    
    # Edge statistics (right bottom)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    edge_text = f"""
    EDGE STATISTICS
    
    Edge Pixels: {stats['edge_pixels_before']} → {stats['edge_pixels_after']} ({stats['edge_increase_percent']:+.1f}%)
    
    VALIDATION CHECKS
    ✓ Contrast increased: {stats['std_increase_percent'] > 0}
    ✓ Edges enhanced:     {stats['edge_increase_percent'] > 0}
    """
    ax4.text(0.1, 0.5, edge_text, fontsize=10, family='monospace', verticalalignment='center')
    
    plt.suptitle(f"Preprocessing Validation: {img_name}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig


def validate_preprocessing(num_samples=20, output_dir=None, save_stats=True):
    """
    Validate preprocessing pipeline by generating side-by-side comparisons.
    
    Args:
        num_samples (int): Number of random samples to process
        output_dir (str, optional): Directory to save comparison images
        save_stats (bool): Whether to save aggregated statistics
    """
    print("=" * 80)
    print("PREPROCESSING VALIDATION SCRIPT")
    print("=" * 80)
    
    # Setup output directory
    if output_dir is None:
        output_dir = project_root / "artifacts" / "preprocessing_validation"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Get preprocessing config
    preprocess_config = Config.get_preprocessing_config()
    print("\nPreprocessing parameters:")
    print(f"  Unsharp Mask: radius={preprocess_config['unsharp_radius']}, "
          f"percent={preprocess_config['unsharp_percent']}, "
          f"threshold={preprocess_config['unsharp_threshold']}")
    print(f"  CLAHE: clip_limit={preprocess_config['clahe_clip_limit']}, "
          f"tile_grid_size={preprocess_config['clahe_tile_grid']}")
    
    # Find training images
    train_dir = Config.DATA_DIR / "train"
    if not train_dir.exists():
        print(f"\n✗ Training directory not found: {train_dir}")
        print("Please ensure FER-2013 dataset is downloaded.")
        return
    
    # Collect all image paths
    all_images = []
    emotion_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
    
    for emotion_dir in emotion_dirs:
        emotion_name = emotion_dir.name
        images = list(emotion_dir.glob("*.png")) + list(emotion_dir.glob("*.jpg"))
        all_images.extend([(img, emotion_name) for img in images])
    
    if not all_images:
        print(f"\n✗ No images found in {train_dir}")
        return
    
    print(f"\nFound {len(all_images)} total images across {len(emotion_dirs)} emotions")
    
    # Sample random images
    if len(all_images) < num_samples:
        print(f"⚠ Only {len(all_images)} images available, using all")
        samples = all_images
    else:
        samples = random.sample(all_images, num_samples)
    
    print(f"\nProcessing {len(samples)} samples...")
    print("=" * 80)
    
    # Track aggregated statistics
    all_stats = []
    validation_passed = 0
    validation_total = 0
    
    # Process each sample
    for i, (img_path, emotion) in enumerate(samples, 1):
        print(f"\n[{i}/{len(samples)}] Processing: {img_path.name} ({emotion})")
        
        # Load image
        img_before = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        if img_before is None:
            print(f"  ✗ Failed to load image")
            continue
        
        # Apply preprocessing
        img_after = preprocess_fer2013_image(img_before, config=preprocess_config)
        
        # Calculate statistics
        stats = get_preprocessing_stats(img_before, img_after)
        all_stats.append(stats)
        
        # Validation checks
        checks_passed = 0
        checks_total = 3
        
        if stats['std_increase_percent'] > 0:
            checks_passed += 1
        if stats['edge_increase_percent'] > 0:
            checks_passed += 1
        if img_after.dtype == np.uint8 and 0 <= img_after.min() <= 255 and 0 <= img_after.max() <= 255:
            checks_passed += 1
        
        validation_total += checks_total
        validation_passed += checks_passed
        
        print(f"  Contrast increase: {stats['std_increase_percent']:+.1f}%")
        print(f"  Edge increase:     {stats['edge_increase_percent']:+.1f}%")
        print(f"  Validation:        {checks_passed}/{checks_total} checks passed")
        
        # Create comparison image
        fig = create_comparison_image(img_before, img_after, stats, img_path.name, emotion)
        
        # Save comparison
        output_path = output_dir / f"comparison_{i:03d}_{emotion}_{img_path.stem}.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  ✓ Saved comparison: {output_path.name}")
    
    # Calculate aggregated statistics
    print("\n" + "=" * 80)
    print("AGGREGATED STATISTICS")
    print("=" * 80)
    
    if all_stats:
        avg_contrast_increase = np.mean([s['std_increase_percent'] for s in all_stats])
        avg_edge_increase = np.mean([s['edge_increase_percent'] for s in all_stats])
        
        print(f"\nAverage metrics across {len(all_stats)} samples:")
        print(f"  Contrast increase: {avg_contrast_increase:+.1f}%")
        print(f"  Edge increase:     {avg_edge_increase:+.1f}%")
        print(f"\nValidation checks: {validation_passed}/{validation_total} passed ({validation_passed/validation_total*100:.1f}%)")
        
        # Overall assessment
        print("\n" + "=" * 80)
        print("OVERALL ASSESSMENT")
        print("=" * 80)
        
        success_criteria = []
        
        # Check 1: Contrast improvement
        if avg_contrast_increase > 5:
            print("✓ Contrast significantly improved (>5%)")
            success_criteria.append(True)
        elif avg_contrast_increase > 0:
            print("⚠ Contrast moderately improved (0-5%)")
            success_criteria.append(True)
        else:
            print("✗ Contrast not improved")
            success_criteria.append(False)
        
        # Check 2: Edge enhancement
        if avg_edge_increase > 10:
            print("✓ Edges significantly enhanced (>10%)")
            success_criteria.append(True)
        elif avg_edge_increase > 0:
            print("⚠ Edges moderately enhanced (0-10%)")
            success_criteria.append(True)
        else:
            print("✗ Edges not enhanced")
            success_criteria.append(False)
        
        # Check 3: Validation pass rate
        pass_rate = validation_passed / validation_total
        if pass_rate >= 0.95:
            print(f"✓ High validation pass rate ({pass_rate*100:.1f}%)")
            success_criteria.append(True)
        elif pass_rate >= 0.80:
            print(f"⚠ Moderate validation pass rate ({pass_rate*100:.1f}%)")
            success_criteria.append(True)
        else:
            print(f"✗ Low validation pass rate ({pass_rate*100:.1f}%)")
            success_criteria.append(False)
        
        # Final verdict
        print("\n" + "=" * 80)
        if all(success_criteria):
            print("✓✓✓ PREPROCESSING VALIDATION PASSED ✓✓✓")
            print("\nPreprocessing is working as expected. Proceed with training.")
        elif sum(success_criteria) >= 2:
            print("⚠⚠⚠ PREPROCESSING VALIDATION ACCEPTABLE ⚠⚠⚠")
            print("\nPreprocessing shows improvements but may benefit from parameter tuning.")
        else:
            print("✗✗✗ PREPROCESSING VALIDATION FAILED ✗✗✗")
            print("\nPreprocessing not providing expected improvements. Check parameters.")
        print("=" * 80)
        
        # Save statistics to file
        if save_stats:
            stats_file = output_dir / "preprocessing_stats.txt"
            with open(stats_file, 'w') as f:
                f.write("PREPROCESSING VALIDATION STATISTICS\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Number of samples: {len(all_stats)}\n\n")
                f.write(f"Average contrast increase: {avg_contrast_increase:+.1f}%\n")
                f.write(f"Average edge increase:     {avg_edge_increase:+.1f}%\n\n")
                f.write(f"Validation checks passed:  {validation_passed}/{validation_total} ({pass_rate*100:.1f}%)\n\n")
                f.write("Per-sample statistics:\n")
                f.write("-" * 80 + "\n")
                for i, stats in enumerate(all_stats, 1):
                    f.write(f"\nSample {i}:\n")
                    f.write(f"  Contrast increase: {stats['std_increase_percent']:+.1f}%\n")
                    f.write(f"  Edge increase:     {stats['edge_increase_percent']:+.1f}%\n")
            
            print(f"\n✓ Statistics saved to: {stats_file}")
    
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print(f"\nComparison images saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Visually inspect comparison images for quality")
    print("2. If validation passed, enable preprocessing in config.py:")
    print("   Config.PREPROCESSING_ENABLED = True")
    print("3. Run Stage 1 training to validate performance gains:")
    print("   python scripts/train_stage1_warmup.py")
    print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate preprocessing pipeline with visual comparisons")
    parser.add_argument('--num-samples', type=int, default=20, help="Number of random samples to process (default: 20)")
    parser.add_argument('--output-dir', type=str, default=None, help="Output directory for comparisons (default: artifacts/preprocessing_validation)")
    parser.add_argument('--no-stats', action='store_true', help="Don't save aggregated statistics file")
    
    args = parser.parse_args()
    
    validate_preprocessing(
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        save_stats=not args.no_stats
    )


if __name__ == "__main__":
    main()
