"""
Dataset download script for Facial Emotion Recognition project.
Downloads FER-2013 dataset from Kaggle using Kaggle API.
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path


def setup_kaggle_credentials():
    """Setup kaggle.json in the correct location for Kaggle API."""
    project_kaggle = Path.cwd() / "kaggle.json"
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    # Check if kaggle.json exists in either location
    if kaggle_json.exists():
        print(f"✓ Found kaggle.json at: {kaggle_json}")
        return True
    elif project_kaggle.exists():
        print(f"✓ Found kaggle.json at: {project_kaggle}")
        print(f"  Copying to: {kaggle_json}")
        
        try:
            # Create .kaggle directory if it doesn't exist
            kaggle_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy kaggle.json to ~/.kaggle/
            shutil.copy2(project_kaggle, kaggle_json)
            
            # Set permissions (read/write for owner only) - important for security
            if os.name != 'nt':  # Unix-like systems
                os.chmod(kaggle_json, 0o600)
            
            print(f"✓ Successfully copied kaggle.json to {kaggle_json}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to copy kaggle.json: {e}")
            return False
    else:
        print("✗ kaggle.json not found!")
        print("\nPlease download kaggle.json from:")
        print("  https://www.kaggle.com/settings/account")
        print("\nThen place it in:")
        print(f"  {project_kaggle}")
        print("  OR")
        print(f"  {kaggle_json}")
        return False


def setup_kaggle_api():
    """Setup Kaggle API authentication."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        # Initialize API
        api = KaggleApi()
        api.authenticate()
        print("✓ Kaggle API authenticated successfully")
        return api
    except ImportError:
        print("✗ Kaggle package not installed!")
        print("  Run: pip install kaggle")
        return None
    except Exception as e:
        print(f"✗ Kaggle authentication failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure kaggle.json exists")
        print("  2. Check kaggle.json format: {\"username\":\"...\",\"key\":\"...\"}")
        print("  3. Verify credentials at https://www.kaggle.com/settings/account")
        return None


def check_dataset_already_exists(base_path="data/raw"):
    """Check if dataset already exists and has content."""
    base_path = Path(base_path)
    
    # Check if raw directory exists
    if not base_path.exists():
        return False
    
    # Check if train and test directories exist
    train_path = base_path / "train"
    test_path = base_path / "test"
    
    if not train_path.exists() or not test_path.exists():
        return False
    
    # Check if directories have content
    emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    
    # Verify train set has emotion folders with images
    train_has_content = False
    for emotion in emotions:
        emotion_path = train_path / emotion
        if emotion_path.exists():
            images = list(emotion_path.glob("*.jpg")) + list(emotion_path.glob("*.png"))
            if len(images) > 0:
                train_has_content = True
                break
    
    # Verify test set has emotion folders with images
    test_has_content = False
    for emotion in emotions:
        emotion_path = test_path / emotion
        if emotion_path.exists():
            images = list(emotion_path.glob("*.jpg")) + list(emotion_path.glob("*.png"))
            if len(images) > 0:
                test_has_content = True
                break
    
    return train_has_content and test_has_content


def download_dataset(api, dataset_name="msambare/fer2013", download_path="data/raw"):
    """Download FER-2013 dataset from Kaggle."""
    try:
        download_path = Path(download_path)
        
        # Check if dataset already exists
        if check_dataset_already_exists(str(download_path)):
            print(f"\n{'='*60}")
            print("✓ Dataset already exists!")
            print(f"Location: {download_path.absolute()}")
            print(f"{'='*60}\n")
            print("Skipping download. Use --force to re-download.")
            return True
        
        # Create directory if it doesn't exist
        download_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Downloading dataset: {dataset_name}")
        print(f"Destination: {download_path.absolute()}")
        print(f"{'='*60}\n")
        
        # Download dataset
        print("Downloading... (this may take a few minutes)")
        api.dataset_download_files(
            dataset_name,
            path=download_path,
            unzip=True,
            quiet=False
        )
        
        print("\n✓ Dataset downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        print("\nPossible solutions:")
        print("  1. Check your internet connection")
        print("  2. Verify dataset name: msambare/fer2013")
        print("  3. Accept dataset terms on Kaggle website")
        return False


def verify_dataset_structure(base_path="data/raw"):
    """Verify the downloaded dataset has correct structure."""
    base_path = Path(base_path)
    
    expected_structure = {
        "train": ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
        "test": ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    }
    
    print(f"\n{'='*60}")
    print("Verifying dataset structure...")
    print(f"{'='*60}\n")
    
    all_valid = True
    total_images = {"train": 0, "test": 0}
    
    for split, emotions in expected_structure.items():
        split_path = base_path / split
        
        if not split_path.exists():
            print(f"✗ Missing directory: {split_path}")
            all_valid = False
            continue
        
        print(f"✓ Found {split}/ directory")
        
        for emotion in emotions:
            emotion_path = split_path / emotion
            
            if not emotion_path.exists():
                print(f"  ✗ Missing: {emotion}/")
                all_valid = False
                continue
            
            # Count images
            image_files = list(emotion_path.glob("*.jpg")) + \
                         list(emotion_path.glob("*.png"))
            count = len(image_files)
            total_images[split] += count
            
            if count > 0:
                print(f"  ✓ {emotion:10s}: {count:5d} images")
            else:
                print(f"  ✗ {emotion:10s}: 0 images (empty!)")
                all_valid = False
        
        print(f"  Total {split}: {total_images[split]} images\n")
    
    return all_valid, total_images


def cleanup_zip_files(base_path="data/raw"):
    """Remove downloaded zip files to save space."""
    base_path = Path(base_path)
    zip_files = list(base_path.glob("*.zip"))
    
    if zip_files:
        print("Cleaning up zip files...")
        for zip_file in zip_files:
            try:
                zip_file.unlink()
                print(f"  ✓ Removed {zip_file.name}")
            except Exception as e:
                print(f"  ✗ Could not remove {zip_file.name}: {e}")


def main():
    """Main download routine."""
    print("=" * 60)
    print("FER-2013 Dataset Downloader")
    print("=" * 60)
    print()
    
    # Step 1: Setup credentials (verify and copy if needed)
    if not setup_kaggle_credentials():
        return False
    print()
    
    # Step 2: Setup API
    api = setup_kaggle_api()
    if api is None:
        return False
    print()
    
    # Step 3: Download dataset
    if not download_dataset(api):
        return False
    
    # Step 4: Verify structure
    valid, totals = verify_dataset_structure()
    
    # Step 5: Cleanup
    cleanup_zip_files()
    
    # Summary
    print(f"\n{'='*60}")
    if valid:
        print("✓ DOWNLOAD COMPLETE!")
        print(f"  Training images: {totals['train']}")
        print(f"  Test images: {totals['test']}")
        print(f"  Total: {totals['train'] + totals['test']}")
        print("\nNext step: Run python scripts/setup/explore_dataset.py")
    else:
        print("✗ Dataset structure verification failed!")
        print("  Some directories or files may be missing.")
        print("  Try re-downloading or check the Kaggle dataset page.")
    print("=" * 60)
    
    return valid


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
