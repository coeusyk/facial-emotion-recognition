"""
GPU and environment verification script for Facial Emotion Recognition project.
Verifies PyTorch installation, CUDA availability, and required dependencies.
"""

import sys
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def check_pytorch():
    """Check PyTorch installation and version."""
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        return True, torch
    except ImportError:
        print("✗ PyTorch is NOT installed!")
        print("  Run: pip install torch torchvision torchaudio")
        return False, None


def check_cuda(torch_module):
    """Check CUDA availability and GPU details."""
    if torch_module is None:
        return False
    
    cuda_available = torch_module.cuda.is_available()
    
    if cuda_available:
        cuda_version = torch_module.version.cuda
        gpu_count = torch_module.cuda.device_count()
        
        print(f"✓ CUDA is available!")
        print(f"  CUDA version: {cuda_version}")
        print(f"  Number of GPUs: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_name = torch_module.cuda.get_device_name(i)
            gpu_memory = torch_module.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {gpu_name}")
            print(f"    Total memory: {gpu_memory:.2f} GB")
        
        return True
    else:
        print("✗ CUDA is NOT available!")
        print("  The system will run on CPU (slower training).")
        return False


def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'numpy',
        'pandas',
        'cv2',
        'matplotlib',
        'seaborn',
        'sklearn',
        'tqdm',
        'PIL',
    ]
    
    missing = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'sklearn':
                import sklearn
            elif package == 'PIL':
                from PIL import Image
            else:
                __import__(package)
        except ImportError:
            missing.append(package)
    
    if not missing:
        print("✓ All required libraries are installed!")
        return True
    else:
        print("✗ Missing libraries:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\n  Run: pip install -r requirements.txt")
        return False


def main():
    """Main verification routine."""
    print("=" * 60)
    print("GPU & Environment Verification for Facial Emotion Recognition")
    print("=" * 60)
    print()
    
    # Check PyTorch
    pytorch_ok, torch_module = check_pytorch()
    if not pytorch_ok:
        print("\n✗ Setup incomplete. Please install PyTorch first.")
        return False
    print()
    
    # Check CUDA
    cuda_ok = check_cuda(torch_module)
    print()
    
    # Check dependencies
    deps_ok = check_dependencies()
    print()
    
    # Summary
    print("=" * 60)
    if pytorch_ok and deps_ok:
        if cuda_ok:
            print("✓ GPU Setup: READY FOR TRAINING")
            print("  Expected training speed: FAST (GPU)")
        else:
            print("✓ Setup Complete: READY FOR TRAINING")
            print("  Expected training speed: SLOW (CPU only)")
            print("  Recommended: Install CUDA for better performance")
        print("=" * 60)
        return True
    else:
        print("✗ Setup Incomplete: MISSING DEPENDENCIES")
        print("  Please install all missing packages before training.")
        print("=" * 60)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
