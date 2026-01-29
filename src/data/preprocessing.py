"""
Image Preprocessing Pipeline for FER-2013 Dataset
=================================================

Professional preprocessing techniques to enhance image quality:
- Unsharp Masking: Recovers lost detail from 48x48 downsampling
- CLAHE: Normalizes contrast across varying lighting conditions

Expected performance gain: +4-5% accuracy (61.88% → 65-66%)

Based on SOTA papers:
- Khaireddin et al. (2021): "Facial Emotion Recognition: State of the Art Performance on FER2013"
- Li et al. (2020): "Deep Facial Expression Recognition: A Survey"
"""

import numpy as np
import cv2
from PIL import Image, ImageFilter
from typing import Optional, Dict, Tuple


def apply_unsharp_mask(
    image_array: np.ndarray,
    radius: float = 2.0,
    percent: int = 150,
    threshold: int = 3
) -> np.ndarray:
    """
    Apply Unsharp Masking to sharpen edges and recover detail lost in downsampling.
    
    Unsharp masking works by:
    1. Creating a blurred version of the image
    2. Subtracting it from the original
    3. Adding the difference back with amplification
    
    Formula: sharpened = original + amount × (original - gaussian_blur(original, radius))
    
    Why FER-2013 needs it:
    - 48×48 images are heavily downsampled from larger originals
    - Critical facial features (eyebrows, eyes, mouth) lose definition
    - Unsharp mask recovers edge detail without amplifying noise
    
    Args:
        image_array (np.ndarray): Input grayscale image (48×48, dtype=uint8, range [0, 255])
        radius (float): Blur kernel radius in pixels. Controls the size of edges to sharpen.
                       - Smaller (1-2): Sharp fine details
                       - Larger (3-5): Sharp coarse features
                       Recommended: 2.0 for 48×48 images
        percent (int): Sharpening strength as percentage (100 = 1x original, 150 = 1.5x)
                      - 100-120: Subtle sharpening
                      - 150-200: Moderate sharpening (RECOMMENDED)
                      - 200+: Aggressive (may cause halos)
                      Recommended: 150 for FER-2013
        threshold (int): Minimum brightness change to apply sharpening (0-255)
                        - 0: Sharpen everything (may amplify noise)
                        - 3: Skip small variations (RECOMMENDED)
                        - 10+: Only sharpen strong edges
                        Recommended: 3 to avoid noise amplification
    
    Returns:
        np.ndarray: Sharpened image (same shape and dtype as input)
    
    Example:
        >>> raw_image = cv2.imread('fer2013_sample.png', cv2.IMREAD_GRAYSCALE)
        >>> sharpened = apply_unsharp_mask(raw_image, radius=2.0, percent=150, threshold=3)
        >>> # Edges are now more defined, facial features clearer
    
    References:
        - PIL ImageFilter.UnsharpMask: https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html
        - Rosenfeld & Kak (1982): "Digital Picture Processing"
    """
    # Input validation
    assert image_array.dtype == np.uint8, f"Expected uint8, got {image_array.dtype}"
    assert image_array.ndim == 2, f"Expected grayscale (2D), got shape {image_array.shape}"
    assert 0 <= image_array.min() and image_array.max() <= 255, "Pixel values must be in [0, 255]"
    
    # Convert NumPy array to PIL Image
    pil_image = Image.fromarray(image_array, mode='L')  # 'L' = grayscale
    
    # Apply unsharp mask filter
    # PIL's UnsharpMask uses: radius (blur radius), percent (strength), threshold (minimum change)
    unsharp_filter = ImageFilter.UnsharpMask(
        radius=radius,
        percent=percent,
        threshold=threshold
    )
    sharpened_pil = pil_image.filter(unsharp_filter)
    
    # Convert back to NumPy array
    sharpened_array = np.array(sharpened_pil, dtype=np.uint8)
    
    return sharpened_array


def apply_clahe(
    image_array: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to normalize contrast.
    
    CLAHE enhances local contrast by:
    1. Dividing image into tiles (e.g., 8×8 grid = 64 tiles)
    2. Equalizing histogram within each tile independently
    3. Applying contrast limiting to prevent noise amplification
    4. Interpolating between tiles to avoid boundary artifacts
    
    Why FER-2013 needs it:
    - Dataset has inconsistent lighting (overexposed/underexposed faces)
    - Global histogram equalization loses detail in bright/dark regions
    - CLAHE adapts to local contrast, making all features visible
    
    Args:
        image_array (np.ndarray): Input grayscale image (48×48, dtype=uint8, range [0, 255])
        clip_limit (float): Contrast limiting threshold. Prevents over-amplification of noise.
                           - 1.0: Very conservative (minimal enhancement)
                           - 2.0: Moderate enhancement (RECOMMENDED)
                           - 3.0-4.0: Aggressive (may amplify noise)
                           Higher values = stronger contrast but more noise
                           Recommended: 2.0 for FER-2013
        tile_grid_size (Tuple[int, int]): Grid dimensions for adaptive equalization (rows, cols)
                                          - (4, 4): Coarse adaptation (16 tiles)
                                          - (8, 8): Standard adaptation (64 tiles, RECOMMENDED)
                                          - (10, 10): Fine adaptation (100 tiles)
                                          For 48×48 images, (8, 8) = 6×6 pixel tiles
                                          Recommended: (8, 8) for FER-2013
    
    Returns:
        np.ndarray: Contrast-enhanced image (same shape and dtype as input)
    
    Example:
        >>> dark_image = cv2.imread('underexposed_face.png', cv2.IMREAD_GRAYSCALE)
        >>> enhanced = apply_clahe(dark_image, clip_limit=2.0, tile_grid_size=(8, 8))
        >>> # Dark regions are brightened, bright regions dimmed, overall balanced
    
    References:
        - OpenCV cv.createCLAHE: https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html
        - Zuiderveld (1994): "Contrast Limited Adaptive Histogram Equalization"
        - Pizer et al. (1987): "Adaptive Histogram Equalization and Its Variations"
    """
    # Input validation
    assert image_array.dtype == np.uint8, f"Expected uint8, got {image_array.dtype}"
    assert image_array.ndim == 2, f"Expected grayscale (2D), got shape {image_array.shape}"
    assert 0 <= image_array.min() and image_array.max() <= 255, "Pixel values must be in [0, 255]"
    
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    # Apply CLAHE to image
    # OpenCV automatically handles uint8 input/output
    enhanced_array = clahe.apply(image_array)
    
    return enhanced_array


def preprocess_fer2013_image(
    image_array: np.ndarray,
    config: Optional[Dict] = None
) -> np.ndarray:
    """
    Complete preprocessing pipeline for FER-2013 images.
    
    Pipeline order (CRITICAL):
    1. Unsharp Mask: Sharpen edges (recover detail from downsampling)
    2. CLAHE: Normalize contrast (make sharpened features visible)
    
    Why this order?
    - Sharpen first: Recovers lost detail before contrast adjustment
    - CLAHE second: Makes sharpened features visible across all lighting
    - Reverse order can amplify noise from CLAHE
    
    Args:
        image_array (np.ndarray): Raw FER-2013 image (48×48 grayscale, uint8, [0, 255])
        config (Dict, optional): Custom preprocessing parameters. If None, uses defaults.
                                Keys:
                                - 'unsharp_radius': float (default: 2.0)
                                - 'unsharp_percent': int (default: 150)
                                - 'unsharp_threshold': int (default: 3)
                                - 'clahe_clip_limit': float (default: 2.0)
                                - 'clahe_tile_grid': Tuple[int, int] (default: (8, 8))
    
    Returns:
        np.ndarray: Preprocessed image ready for model training (48×48, uint8, [0, 255])
    
    Example:
        >>> # Default preprocessing
        >>> raw_img = load_fer2013_image('train/angry/001.png')
        >>> processed = preprocess_fer2013_image(raw_img)
        
        >>> # Custom preprocessing
        >>> custom_config = {
        ...     'unsharp_radius': 2.5,
        ...     'unsharp_percent': 180,
        ...     'unsharp_threshold': 3,
        ...     'clahe_clip_limit': 2.5,
        ...     'clahe_tile_grid': (10, 10)
        ... }
        >>> processed = preprocess_fer2013_image(raw_img, config=custom_config)
    
    Performance Impact:
        - Expected accuracy gain: +4-5% (61.88% → 65-66%)
        - Preprocessing time: ~2ms per image (negligible for training)
        - Stage 1 accuracy boost: +3-4% (35% → 38-39%)
    
    Success Criteria:
        - Visual: Edges sharper, contrast balanced, no halos/artifacts
        - Statistical: Higher pixel std dev, more detected edges
        - Training: Stage 1 improves by ≥2%, no training divergence
    """
    # Default configuration (research-backed optimal parameters for FER-2013)
    default_config = {
        'unsharp_radius': 2.0,      # Optimal for 48×48 images
        'unsharp_percent': 150,     # Moderate sharpening, no halos
        'unsharp_threshold': 3,     # Avoid noise amplification
        'clahe_clip_limit': 2.0,    # Balanced contrast enhancement
        'clahe_tile_grid': (8, 8)   # 64 tiles for 48×48 image
    }
    
    # Use custom config if provided, otherwise use defaults
    if config is None:
        config = default_config
    else:
        # Merge custom config with defaults (custom values override)
        config = {**default_config, **config}
    
    # Step 1: Apply Unsharp Mask (sharpen edges)
    sharpened = apply_unsharp_mask(
        image_array,
        radius=config['unsharp_radius'],
        percent=config['unsharp_percent'],
        threshold=config['unsharp_threshold']
    )
    
    # Step 2: Apply CLAHE (normalize contrast)
    enhanced = apply_clahe(
        sharpened,
        clip_limit=config['clahe_clip_limit'],
        tile_grid_size=config['clahe_tile_grid']
    )
    
    return enhanced


def get_preprocessing_stats(image_before: np.ndarray, image_after: np.ndarray) -> Dict:
    """
    Calculate statistical metrics to validate preprocessing effectiveness.
    
    Metrics:
    - Pixel intensity distribution (mean, std, range)
    - Contrast metrics (standard deviation increase)
    - Edge strength (Sobel edge detection)
    
    Args:
        image_before (np.ndarray): Original image before preprocessing
        image_after (np.ndarray): Image after preprocessing
    
    Returns:
        Dict: Statistical metrics comparing before/after
    
    Example:
        >>> raw = load_image('sample.png')
        >>> processed = preprocess_fer2013_image(raw)
        >>> stats = get_preprocessing_stats(raw, processed)
        >>> print(f"Contrast improved by {stats['std_increase']:.1f}%")
    """
    stats = {}
    
    # Pixel intensity statistics
    stats['before_mean'] = float(np.mean(image_before))
    stats['after_mean'] = float(np.mean(image_after))
    stats['before_std'] = float(np.std(image_before))
    stats['after_std'] = float(np.std(image_after))
    stats['before_min'] = int(np.min(image_before))
    stats['after_min'] = int(np.min(image_after))
    stats['before_max'] = int(np.max(image_before))
    stats['after_max'] = int(np.max(image_after))
    
    # Contrast improvement (higher std = more contrast)
    if stats['before_std'] > 0:
        stats['std_increase_percent'] = ((stats['after_std'] - stats['before_std']) / stats['before_std']) * 100
    else:
        stats['std_increase_percent'] = 0.0
    
    # Edge strength using Sobel operator
    sobel_x_before = cv2.Sobel(image_before, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y_before = cv2.Sobel(image_before, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude_before = np.sqrt(sobel_x_before**2 + sobel_y_before**2)
    
    sobel_x_after = cv2.Sobel(image_after, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y_after = cv2.Sobel(image_after, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude_after = np.sqrt(sobel_x_after**2 + sobel_y_after**2)
    
    # Count strong edges (magnitude > threshold)
    edge_threshold = 50
    stats['edge_pixels_before'] = int(np.sum(edge_magnitude_before > edge_threshold))
    stats['edge_pixels_after'] = int(np.sum(edge_magnitude_after > edge_threshold))
    
    if stats['edge_pixels_before'] > 0:
        stats['edge_increase_percent'] = ((stats['edge_pixels_after'] - stats['edge_pixels_before']) / stats['edge_pixels_before']) * 100
    else:
        stats['edge_increase_percent'] = 0.0
    
    return stats


def main():
    """
    Test preprocessing pipeline with sample images.
    Requires FER-2013 dataset to be downloaded.
    """
    import os
    from pathlib import Path
    
    print("=" * 80)
    print("PREPROCESSING PIPELINE TEST")
    print("=" * 80)
    
    # Sample image path (adjust based on your dataset location)
    project_root = Path(__file__).parent.parent.parent
    sample_dir = project_root / "data" / "raw" / "train"
    
    if not sample_dir.exists():
        print(f"\n⚠ Sample directory not found: {sample_dir}")
        print("Please ensure FER-2013 dataset is downloaded.")
        return
    
    # Find first available emotion directory
    emotion_dirs = [d for d in sample_dir.iterdir() if d.is_dir()]
    if not emotion_dirs:
        print(f"\n⚠ No emotion directories found in {sample_dir}")
        return
    
    sample_emotion_dir = emotion_dirs[0]
    sample_images = list(sample_emotion_dir.glob("*.png"))[:3]  # Load 3 samples
    
    if not sample_images:
        print(f"\n⚠ No images found in {sample_emotion_dir}")
        return
    
    print(f"\nTesting with {len(sample_images)} samples from: {sample_emotion_dir.name}")
    print("\nPreprocessing parameters (defaults):")
    print("  Unsharp Mask: radius=2.0, percent=150, threshold=3")
    print("  CLAHE: clip_limit=2.0, tile_grid_size=(8, 8)")
    
    for i, img_path in enumerate(sample_images, 1):
        print(f"\n{'-'*80}")
        print(f"Sample {i}: {img_path.name}")
        print(f"{'-'*80}")
        
        # Load image
        img_before = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        if img_before is None:
            print(f"Failed to load image: {img_path}")
            continue
        
        # Apply preprocessing
        img_after = preprocess_fer2013_image(img_before)
        
        # Calculate statistics
        stats = get_preprocessing_stats(img_before, img_after)
        
        print("\nStatistical Analysis:")
        print(f"  Pixel mean: {stats['before_mean']:.1f} → {stats['after_mean']:.1f}")
        print(f"  Pixel std dev: {stats['before_std']:.1f} → {stats['after_std']:.1f} ({stats['std_increase_percent']:+.1f}%)")
        print(f"  Pixel range: [{stats['before_min']}, {stats['before_max']}] → [{stats['after_min']}, {stats['after_max']}]")
        print(f"  Edge pixels: {stats['edge_pixels_before']} → {stats['edge_pixels_after']} ({stats['edge_increase_percent']:+.1f}%)")
        
        # Validation checks
        print("\nValidation Checks:")
        checks_passed = 0
        checks_total = 3
        
        # Check 1: Contrast should increase
        if stats['std_increase_percent'] > 0:
            print("  ✓ Contrast increased (std dev higher)")
            checks_passed += 1
        else:
            print("  ✗ Contrast did not increase")
        
        # Check 2: Edge strength should increase
        if stats['edge_increase_percent'] > 0:
            print("  ✓ Edge strength increased")
            checks_passed += 1
        else:
            print("  ✗ Edge strength did not increase")
        
        # Check 3: Output should be valid uint8
        if img_after.dtype == np.uint8 and 0 <= img_after.min() <= 255 and 0 <= img_after.max() <= 255:
            print("  ✓ Output format valid (uint8, [0, 255])")
            checks_passed += 1
        else:
            print("  ✗ Output format invalid")
        
        print(f"\n  Result: {checks_passed}/{checks_total} checks passed")
    
    print("\n" + "=" * 80)
    print("PREPROCESSING TEST COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Run validation script: python scripts/validate_preprocessing.py")
    print("2. Visual inspection of side-by-side comparisons")
    print("3. Test with Stage 1 training to validate performance gains")
    print("=" * 80)


if __name__ == "__main__":
    main()
