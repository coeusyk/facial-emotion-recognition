#!/usr/bin/env python3
"""
NVIDIA Driver and CUDA Diagnostic Script
=========================================

Helps diagnose GPU/NVIDIA driver issues and verify the environment
is properly configured for GPU monitoring and training.

Usage:
    python scripts/diagnose_nvidia.py
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, description: str = None) -> tuple: # type: ignore
    """Run a system command and return (success, output)."""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True,
            timeout=5
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)


def main():
    print("="*80)
    print("NVIDIA DRIVER & CUDA DIAGNOSTIC TOOL")
    print("="*80)
    
    issues = []
    
    # 1. Check nvidia-smi
    print("\n1. Checking NVIDIA drivers (nvidia-smi)...")
    success, output = run_command('nvidia-smi')
    
    if success:
        print("   ✓ nvidia-smi found and working")
        # Extract GPU info
        for line in output.split('\n'):
            if 'NVIDIA' in line or 'GPU' in line or 'CUDA' in line:
                print(f"     {line.strip()}")
    else:
        print("   ✗ nvidia-smi not found or not working")
        print("     This means NVIDIA drivers are not properly installed")
        issues.append("NVIDIA drivers")
    
    # 2. Check CUDA
    print("\n2. Checking CUDA installation...")
    success, output = run_command('nvcc --version')
    
    if success:
        print("   ✓ CUDA compiler (nvcc) found")
        print(f"     {output.split(chr(10))[0]}")
    else:
        print("   ✗ CUDA compiler not found")
        issues.append("CUDA")
    
    # 3. Check Python CUDA support
    print("\n3. Checking Python CUDA/PyTorch support...")
    try:
        import torch
        print(f"   ✓ PyTorch installed: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"   ✓ CUDA available in PyTorch")
            print(f"     GPU: {torch.cuda.get_device_name(0)}")
            print(f"     CUDA: {torch.version.cuda}")
        else:
            print("   ✗ CUDA not available in PyTorch")
            issues.append("PyTorch CUDA support")
    except ImportError:
        print("   ✗ PyTorch not installed")
        issues.append("PyTorch")
    
    # 4. Check nvidia-ml-py3
    print("\n4. Checking nvidia-ml-py3 (for GPU monitoring)...")
    try:
        import pynvml
        print("   ✓ nvidia-ml-py3 imported successfully")
        
        try:
            pynvml.nvmlInit()
            print("   ✓ NVML initialized successfully")
            
            device_count = pynvml.nvmlDeviceGetCount()
            print(f"   ✓ Found {device_count} GPU(s)")
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                gpu_name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(gpu_name, bytes):
                    gpu_name = gpu_name.decode('utf-8')
                print(f"     GPU {i}: {gpu_name}")
            
            pynvml.nvmlShutdown()
            
        except Exception as e:
            print(f"   ✗ NVML initialization failed: {e}")
            issues.append("nvidia-ml-py3 package")
    
    except ImportError:
        print("   ✗ nvidia-ml-py3 not installed")
        issues.append("nvidia-ml-py3 package")
    
    # 5. Check environment variables
    print("\n5. Checking environment variables...")
    import os
    
    env_vars = [
        'CUDA_HOME',
        'CUDA_PATH',
        'PATH',
        'LD_LIBRARY_PATH'  # Linux
    ]
    
    for var in env_vars:
        if var in os.environ:
            value = os.environ[var]
            if len(value) > 80:
                value = value[:77] + "..."
            print(f"   ✓ {var}={value}")
    
    # 6. Check for nvidia-smi in PATH
    print("\n6. Checking nvidia-smi location...")
    possible_paths = [
        'C:\\Windows\\System32\\nvidia-smi.exe',
        'C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe',
        'C:\\Program Files (x86)\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe',
        '/usr/bin/nvidia-smi',
        '/usr/local/cuda/bin/nvidia-smi'
    ]
    
    found = False
    for path in possible_paths:
        if Path(path).exists():
            print(f"   ✓ Found: {path}")
            found = True
    
    if not found:
        print("   ⚠ nvidia-smi not found in common locations")
        issues.append("nvidia-smi in PATH")
    
    # Summary
    print("\n" + "="*80)
    if issues:
        print("ISSUES FOUND:")
        print("="*80)
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        
        print("\nRECOMMENDED FIXES:")
        print("="*80)
        
        if "NVIDIA drivers" in issues:
            print("  • Download and install NVIDIA drivers:")
            print("    https://www.nvidia.com/Download/driverDetails.aspx")
            print("  • After installation, restart your computer")
            print("  • Verify with: nvidia-smi")
        
        if "CUDA" in issues:
            print("  • Download and install CUDA Toolkit:")
            print("    https://developer.nvidia.com/cuda-downloads")
            print("  • Select your Windows version and download")
        
        if "PyTorch CUDA support" in issues:
            print("  • Reinstall PyTorch with CUDA support:")
            print("    https://pytorch.org/get-started/locally/")
            print("  • Select your CUDA version (check nvidia-smi for version)")
        
        if "nvidia-ml-py3 package" in issues:
            print("  • Install nvidia-ml-py3:")
            print("    pip install nvidia-ml-py3")
        
        if "nvidia-smi in PATH" in issues:
            print("  • Add NVIDIA driver directory to system PATH:")
            print("    Windows: Add 'C:\\Program Files\\NVIDIA Corporation\\NVSMI' to PATH")
            print("  • Or set environment variable:")
            print("    $env:NVIDIA_DRIVER_PATH = 'C:\\Program Files\\NVIDIA Corporation\\NVSMI'")
        
        print("\n" + "="*80)
        print("After fixing issues, run this script again to verify")
        return False
    else:
        print("✓ ALL CHECKS PASSED!")
        print("="*80)
        print("\nYour system is ready for GPU monitoring and training!")
        print("You can now run:")
        print("  • python scripts/gpu_monitor.py")
        print("  • python scripts/train_stage1_warmup.py")
        return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
