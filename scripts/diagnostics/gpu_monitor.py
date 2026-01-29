#!/usr/bin/env python3
"""
GPU Monitoring Script for Training Benchmarks
==============================================

Monitors NVIDIA GPU metrics continuously during model training.
Runs in a background daemon thread to avoid interfering with training.
Automatically generates visualization charts after monitoring completes.

Requirements:
    pip install nvidia-ml-py3 matplotlib

Usage:
    from gpu_monitor import GPUMonitor
    
    # Initialize and start monitoring
    # Metrics saved to gpu_metrics/gpu_metrics_rtx4060.csv
    # Charts auto-generated in gpu_metrics/ on stop()
    monitor = GPUMonitor(log_file='gpu_metrics_rtx4060.csv', interval=2, metrics_dir='gpu_metrics')
    monitor.start()
    
    # Run your training
    for epoch in range(10):
        train_model()
    
    # Stop and view summary + auto-generate charts
    monitor.stop()

Output:
    - gpu_metrics/gpu_metrics_RTX4060.csv (detailed metrics)
    - gpu_metrics/gpu_metrics_RTX4060_gpu_util.png (GPU utilization chart)
    - gpu_metrics/gpu_metrics_RTX4060_memory.png (memory usage chart)
    - gpu_metrics/gpu_metrics_RTX4060_temperature.png (temperature chart)
    - gpu_metrics/gpu_metrics_RTX4060_summary.png (4-panel summary)

Author: Facial Emotion Recognition Project
Date: 2026-01-29
"""

import time
import csv
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
import sys


class GPUMonitor:
    """
    Background GPU monitoring for training benchmarks.
    
    Captures GPU utilization, memory, temperature, and power metrics
    at regular intervals without blocking the main training process.
    """
    
    def __init__(
        self, 
        log_file: str = 'gpu_metrics.csv',
        interval: float = 2.0,
        gpu_index: int = 0,
        console_output: bool = False,
        metrics_dir: str = 'gpu_metrics'
    ):
        """
        Initialize GPU monitor.
        
        Args:
            log_file: CSV filename (without path, will be stored in metrics_dir)
            interval: Sampling interval in seconds (default: 2.0)
            gpu_index: GPU device index to monitor (default: 0)
            console_output: Print metrics to console in addition to CSV
            metrics_dir: Directory to store metrics (default: 'gpu_metrics')
        """
        # Create metrics directory if it doesn't exist
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.metrics_dir / log_file
        self.interval = interval
        self.gpu_index = gpu_index
        self.console_output = console_output
        
        # Thread control
        self._monitoring = False
        self._thread: Optional[threading.Thread] = None
        
        # Metrics storage for summary statistics
        self._metrics_history: List[Dict] = []
        
        # Initialize NVML
        self._nvml_initialized = False
        self._gpu_handle = None
        
        # Try to initialize NVML
        try:
            import pynvml
            self.pynvml = pynvml
            self._init_nvml()
        except ImportError:
            print("✗ ERROR: pynvml not installed!")
            print("  Install with: pip install nvidia-ml-py3")
            sys.exit(1)
        except Exception as e:
            print(f"✗ ERROR: Failed to initialize NVML: {e}")
            print("\nTroubleshooting steps:")
            print("  1. Verify NVIDIA drivers are installed:")
            print("     Windows: nvidia-smi (from command line)")
            print("     Linux: nvidia-smi")
            print("  2. If nvidia-smi works but this fails, reinstall pynvml:")
            print("     pip uninstall nvidia-ml-py3")
            print("     pip install nvidia-ml-py3")
            print("  3. Ensure NVIDIA driver paths are in system PATH")
            print("  4. On some systems, you may need to set:")
            print("     $env:NVIDIA_DRIVER_PATH = 'C:\\Program Files\\NVIDIA Corporation\\NVSMI'")
            print(f"\nOriginal error: {type(e).__name__}: {e}")
            sys.exit(1)
    
    def _init_nvml(self):
        """Initialize NVML and get GPU handle."""
        try:
            self.pynvml.nvmlInit()
            self._nvml_initialized = True
            self._gpu_handle = self.pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            
            # Get GPU name for confirmation
            gpu_name = self.pynvml.nvmlDeviceGetName(self._gpu_handle)
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode('utf-8')
            
            print(f"✓ NVML initialized successfully")
            print(f"  Monitoring GPU {self.gpu_index}: {gpu_name}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize NVML: {e}")
    
    def _get_gpu_metrics(self) -> Dict:
        """
        Capture current GPU metrics.
        
        Returns:
            Dictionary with timestamp and GPU metrics
        """
        try:
            # Get utilization
            util = self.pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
            gpu_util = util.gpu
            
            # Get memory info
            mem_info = self.pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
            mem_used_mb = int(mem_info.used) / (1024 ** 2)
            mem_total_mb = int(mem_info.total) / (1024 ** 2)
            mem_util = (int(mem_info.used) / int(mem_info.total)) * 100
            
            # Get temperature
            temp = self.pynvml.nvmlDeviceGetTemperature(
                self._gpu_handle, 
                self.pynvml.NVML_TEMPERATURE_GPU
            )
            
            # Get power draw (in milliwatts, convert to watts)
            try:
                power_mw = self.pynvml.nvmlDeviceGetPowerUsage(self._gpu_handle)
                power_w = power_mw / 1000.0
            except:
                # Some GPUs don't support power monitoring
                power_w = 0.0
            
            return {
                'timestamp': datetime.now().isoformat(),
                'unix_time': time.time(),
                'gpu_util_percent': gpu_util,
                'mem_used_mb': round(mem_used_mb, 2),
                'mem_total_mb': round(mem_total_mb, 2),
                'mem_util_percent': round(mem_util, 2),
                'temperature_c': temp,
                'power_draw_w': round(power_w, 2)
            }
            
        except Exception as e:
            print(f"⚠ Warning: Failed to read GPU metrics: {e}")
            return None # type: ignore
    
    def _monitoring_loop(self):
        """Background monitoring loop (runs in daemon thread)."""
        # Create CSV file with headers
        csv_exists = self.log_file.exists()
        
        try:
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'timestamp', 'unix_time', 'gpu_util_percent', 
                    'mem_used_mb', 'mem_total_mb', 'mem_util_percent',
                    'temperature_c', 'power_draw_w'
                ])
                
                # Write header if file is new
                if not csv_exists:
                    writer.writeheader()
                
                # Monitoring loop
                while self._monitoring:
                    metrics = self._get_gpu_metrics()
                    
                    if metrics:
                        # Write to CSV
                        writer.writerow(metrics)
                        f.flush()  # Ensure data is written immediately
                        
                        # Store for summary
                        self._metrics_history.append(metrics)
                        
                        # Console output if enabled
                        if self.console_output:
                            print(
                                f"[{metrics['timestamp']}] "
                                f"GPU: {metrics['gpu_util_percent']:3d}% | "
                                f"Mem: {metrics['mem_used_mb']:7.0f}/{metrics['mem_total_mb']:.0f} MB "
                                f"({metrics['mem_util_percent']:5.1f}%) | "
                                f"Temp: {metrics['temperature_c']:2d}°C | "
                                f"Power: {metrics['power_draw_w']:5.1f}W"
                            )
                    
                    # Sleep for interval
                    time.sleep(self.interval)
                    
        except Exception as e:
            print(f"✗ Error in monitoring loop: {e}")
    
    def start(self):
        """Start GPU monitoring in background thread."""
        if self._monitoring:
            print("⚠ Monitoring already running")
            return
        
        print(f"\n{'='*80}")
        print("GPU MONITORING STARTED")
        print(f"{'='*80}")
        print(f"  Log file: {self.log_file.absolute()}")
        print(f"  Interval: {self.interval}s")
        print(f"  Console output: {'Enabled' if self.console_output else 'Disabled'}")
        print(f"{'='*80}\n")
        
        self._monitoring = True
        self._metrics_history = []
        
        # Start monitoring thread (daemon=True so it terminates with main program)
        self._thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop GPU monitoring and print summary."""
        if not self._monitoring:
            print("⚠ Monitoring not running")
            return
        
        print(f"\n{'='*80}")
        print("STOPPING GPU MONITORING...")
        print(f"{'='*80}")
        
        self._monitoring = False
        
        # Wait for thread to finish (with timeout)
        if self._thread:
            self._thread.join(timeout=self.interval * 2)
        
        # Print summary statistics
        self._print_summary()
        
        print(f"\n✓ Monitoring stopped. Metrics saved to: {self.log_file.absolute()}")
        
        # Generate charts automatically
        try:
            self.generate_charts()
        except Exception as e:
            print(f"\n⚠ Could not generate charts: {e}")
            print(f"  Install matplotlib for visualizations: pip install matplotlib")
        
        print(f"{'='*80}\n")
    
    def _print_summary(self):
        """Print summary statistics from monitoring session."""
        if not self._metrics_history:
            print("\n⚠ No metrics collected")
            return
        
        # Calculate statistics
        gpu_utils = [m['gpu_util_percent'] for m in self._metrics_history]
        mem_utils = [m['mem_util_percent'] for m in self._metrics_history]
        temps = [m['temperature_c'] for m in self._metrics_history]
        powers = [m['power_draw_w'] for m in self._metrics_history if m['power_draw_w'] > 0]
        
        duration = self._metrics_history[-1]['unix_time'] - self._metrics_history[0]['unix_time']
        
        print(f"\n{'='*80}")
        print("GPU MONITORING SUMMARY")
        print(f"{'='*80}")
        print(f"  Duration: {duration:.1f}s ({len(self._metrics_history)} samples)")
        print(f"\n  GPU Utilization:")
        print(f"    Average: {sum(gpu_utils)/len(gpu_utils):.1f}%")
        print(f"    Maximum: {max(gpu_utils):.1f}%")
        print(f"    Minimum: {min(gpu_utils):.1f}%")
        
        print(f"\n  Memory Utilization:")
        print(f"    Average: {sum(mem_utils)/len(mem_utils):.1f}%")
        print(f"    Maximum: {max(mem_utils):.1f}%")
        print(f"    Peak Usage: {max(m['mem_used_mb'] for m in self._metrics_history):.0f} MB")
        
        print(f"\n  Temperature:")
        print(f"    Average: {sum(temps)/len(temps):.1f}°C")
        print(f"    Maximum: {max(temps):.1f}°C")
        
        if powers:
            print(f"\n  Power Draw:")
            print(f"    Average: {sum(powers)/len(powers):.1f}W")
            print(f"    Maximum: {max(powers):.1f}W")
        
        # Performance classification
        avg_gpu_util = sum(gpu_utils)/len(gpu_utils)
        print(f"\n  Performance Classification:")
        if avg_gpu_util >= 85:
            print(f"    ✓ GPU-Limited (excellent GPU utilization)")
        elif avg_gpu_util >= 60:
            print(f"    ⚠ Moderate GPU utilization")
        else:
            print(f"    ✗ CPU-Limited (GPU underutilized, bottleneck elsewhere)")
        
        print(f"{'='*80}")
    
    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics as dictionary.
        
        Returns:
            Dictionary with average, max, min statistics
        """
        if not self._metrics_history:
            return {}
        
        gpu_utils = [m['gpu_util_percent'] for m in self._metrics_history]
        mem_utils = [m['mem_util_percent'] for m in self._metrics_history]
        temps = [m['temperature_c'] for m in self._metrics_history]
        powers = [m['power_draw_w'] for m in self._metrics_history if m['power_draw_w'] > 0]
        
        return {
            'sample_count': len(self._metrics_history),
            'duration_seconds': self._metrics_history[-1]['unix_time'] - self._metrics_history[0]['unix_time'],
            'gpu_util_avg': sum(gpu_utils)/len(gpu_utils),
            'gpu_util_max': max(gpu_utils),
            'gpu_util_min': min(gpu_utils),
            'mem_util_avg': sum(mem_utils)/len(mem_utils),
            'mem_util_max': max(mem_utils),
            'mem_peak_mb': max(m['mem_used_mb'] for m in self._metrics_history),
            'temp_avg': sum(temps)/len(temps),
            'temp_max': max(temps),
            'power_avg': sum(powers)/len(powers) if powers else 0,
            'power_max': max(powers) if powers else 0
        }
    
    def generate_charts(self, output_dir: Optional[str] = None):
        """
        Generate visualization charts from collected GPU metrics.
        
        Creates 4 charts:
        - GPU Utilization over time
        - Memory Usage over time
        - Temperature over time
        - Combined Summary (4 subplots)
        
        Args:
            output_dir: Directory to save charts (default: metrics_dir)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("✗ matplotlib not installed. Install with: pip install matplotlib")
            return
        
        if not self._metrics_history:
            print("⚠ No metrics data available for charting")
            return
        
        # Use metrics_dir if output_dir not specified
        if output_dir is None:
            output_dir = str(self.metrics_dir)
        else:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Extract data for plotting
        times = [(m['unix_time'] - self._metrics_history[0]['unix_time']) / 60.0 
                 for m in self._metrics_history]  # Convert to minutes from start
        gpu_utils = [m['gpu_util_percent'] for m in self._metrics_history]
        mem_utils = [m['mem_util_percent'] for m in self._metrics_history]
        mem_used = [m['mem_used_mb'] for m in self._metrics_history]
        temps = [m['temperature_c'] for m in self._metrics_history]
        powers = [m['power_draw_w'] for m in self._metrics_history]
        
        gpu_name = Path(self.log_file).stem.replace('gpu_metrics_', '').replace('_', ' ')
        
        print(f"\n{'='*80}")
        print("GENERATING GPU METRICS CHARTS")
        print(f"{'='*80}")
        
        # Chart 1: GPU Utilization
        plt.figure(figsize=(12, 5))
        plt.plot(times, gpu_utils, linewidth=2, color='#1f77b4', marker='o', markersize=3, alpha=0.7)
        plt.axhline(y=80, color='red', linestyle='--', linewidth=1, alpha=0.5, label='High Utilization (80%)')
        plt.fill_between(times, gpu_utils, alpha=0.3, color='#1f77b4')
        plt.xlabel('Time (minutes)', fontsize=11)
        plt.ylabel('GPU Utilization (%)', fontsize=11)
        plt.title(f'GPU Utilization Over Time - {gpu_name}', fontsize=13, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(0, 105)
        chart_path = Path(output_dir) / f'{Path(self.log_file).stem}_gpu_util.png'
        plt.savefig(chart_path, dpi=100, bbox_inches='tight')
        print(f"  ✓ GPU Utilization: {chart_path}")
        plt.close()
        
        # Chart 2: Memory Usage
        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(times, mem_used, linewidth=2, color='#2ca02c', marker='s', markersize=3, alpha=0.7, label='Memory Used (MB)')
        ax1.set_xlabel('Time (minutes)', fontsize=11)
        ax1.set_ylabel('Memory Used (MB)', fontsize=11, color='#2ca02c')
        ax1.tick_params(axis='y', labelcolor='#2ca02c')
        ax1.grid(True, alpha=0.3)
        
        ax2 = ax1.twinx()
        ax2.plot(times, mem_utils, linewidth=2, color='#ff7f0e', marker='^', markersize=3, alpha=0.7, label='Memory Utilization (%)')
        ax2.set_ylabel('Memory Utilization (%)', fontsize=11, color='#ff7f0e')
        ax2.tick_params(axis='y', labelcolor='#ff7f0e')
        ax2.set_ylim(0, 105)
        
        plt.title(f'Memory Usage Over Time - {gpu_name}', fontsize=13, fontweight='bold')
        fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.95))
        chart_path = Path(output_dir) / f'{Path(self.log_file).stem}_memory.png'
        plt.savefig(chart_path, dpi=100, bbox_inches='tight')
        print(f"  ✓ Memory Usage: {chart_path}")
        plt.close()
        
        # Chart 3: Temperature
        plt.figure(figsize=(12, 5))
        plt.plot(times, temps, linewidth=2, color='#d62728', marker='D', markersize=3, alpha=0.7)
        plt.fill_between(times, temps, alpha=0.3, color='#d62728')
        plt.axhline(y=80, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Thermal Throttle Threshold')
        plt.xlabel('Time (minutes)', fontsize=11)
        plt.ylabel('Temperature (°C)', fontsize=11)
        plt.title(f'GPU Temperature Over Time - {gpu_name}', fontsize=13, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        chart_path = Path(output_dir) / f'{Path(self.log_file).stem}_temperature.png'
        plt.savefig(chart_path, dpi=100, bbox_inches='tight')
        print(f"  ✓ Temperature: {chart_path}")
        plt.close()
        
        # Chart 4: Combined Summary (4 subplots)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'GPU Metrics Summary - {gpu_name}', fontsize=14, fontweight='bold', y=1.00)
        
        # GPU Utilization
        axes[0, 0].plot(times, gpu_utils, linewidth=2, color='#1f77b4', alpha=0.7)
        axes[0, 0].fill_between(times, gpu_utils, alpha=0.3, color='#1f77b4')
        axes[0, 0].set_ylabel('GPU Util (%)', fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 105)
        axes[0, 0].set_title('GPU Utilization', fontweight='bold')
        
        # Memory Utilization
        axes[0, 1].plot(times, mem_utils, linewidth=2, color='#ff7f0e', alpha=0.7)
        axes[0, 1].fill_between(times, mem_utils, alpha=0.3, color='#ff7f0e')
        axes[0, 1].set_ylabel('Memory Util (%)', fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 105)
        axes[0, 1].set_title('Memory Utilization', fontweight='bold')
        
        # Temperature
        axes[1, 0].plot(times, temps, linewidth=2, color='#d62728', alpha=0.7)
        axes[1, 0].fill_between(times, temps, alpha=0.3, color='#d62728')
        axes[1, 0].set_xlabel('Time (minutes)', fontsize=10)
        axes[1, 0].set_ylabel('Temperature (°C)', fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_title('Temperature', fontweight='bold')
        
        # Power Draw
        axes[1, 1].plot(times, powers, linewidth=2, color='#9467bd', alpha=0.7)
        axes[1, 1].fill_between(times, powers, alpha=0.3, color='#9467bd')
        axes[1, 1].set_xlabel('Time (minutes)', fontsize=10)
        axes[1, 1].set_ylabel('Power Draw (W)', fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_title('Power Draw', fontweight='bold')
        
        plt.tight_layout()
        chart_path = Path(output_dir) / f'{Path(self.log_file).stem}_summary.png'
        plt.savefig(chart_path, dpi=100, bbox_inches='tight')
        print(f"  ✓ Summary Chart: {chart_path}")
        plt.close()
        
        print(f"{'='*80}")
        print(f"✓ Charts saved to: {Path(output_dir).absolute()}")
    
    def __del__(self):
        """Cleanup NVML on deletion."""
        if self._nvml_initialized:
            try:
                self.pynvml.nvmlShutdown()
            except:
                pass


def demo_training_loop():
    """Demo training loop to test GPU monitoring."""
    import torch
    import torch.nn as nn
    
    print("\n" + "="*80)
    print("DEMO TRAINING LOOP")
    print("="*80)
    print("Simulating model training with GPU operations...")
    print("This will run for ~30 seconds to generate GPU load\n")
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("✗ CUDA not available, using CPU")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(1024, 2048),
        nn.ReLU(),
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Linear(1024, 10)
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Simulate training
    for epoch in range(10):
        print(f"\nEpoch {epoch+1}/10")
        
        # Batch training
        for batch in range(20):
            # Generate random data
            inputs = torch.randn(128, 1024).to(device)
            labels = torch.randint(0, 10, (128,)).to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"  Loss: {loss.item():.4f}")
        time.sleep(0.5)  # Small pause between epochs
    
    print("\n✓ Demo training complete!")


if __name__ == '__main__':
    """
    Example usage: Run GPU monitoring during demo training.
    """
    print("\n" + "="*80)
    print("GPU MONITORING SCRIPT - DEMO MODE")
    print("="*80)
    print("\nThis will:")
    print("  1. Start GPU monitoring in background")
    print("  2. Run a demo training loop (~30 seconds)")
    print("  3. Stop monitoring and show summary")
    print("  4. Save metrics to CSV file")
    
    # Determine GPU name for log file
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode('utf-8')
        pynvml.nvmlShutdown()
        
        # Clean name for filename
        gpu_name_clean = gpu_name.replace(' ', '_').replace('NVIDIA', '').replace('GeForce', '').strip('_')
        log_file = f'gpu_metrics_{gpu_name_clean}.csv'
    except:
        log_file = 'gpu_metrics.csv'
    
    # Initialize monitor (metrics saved to gpu_metrics/ folder)
    monitor = GPUMonitor(
        log_file=log_file,
        interval=2.0,
        metrics_dir='gpu_metrics',
        console_output=False  # Set to True to see real-time metrics
    )
    
    try:
        # Start monitoring
        monitor.start()
        
        # Run demo training
        demo_training_loop()
        
    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Error during training: {e}")
    finally:
        # Always stop monitoring
        monitor.stop()
    
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)
    print(f"\nCheck the gpu_metrics/ folder for:")
    print(f"  • CSV file with detailed metrics: {log_file}")
    print(f"  • PNG charts showing GPU performance:")
    print(f"    - GPU Utilization over time")
    print(f"    - Memory Usage over time")
    print(f"    - Temperature over time")
    print(f"    - 4-panel Summary chart")
    print(f"\nYou can analyze this data in Excel, Python, or any CSV viewer.\n")
