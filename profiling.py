"""
Performance profiling utilities using NVIDIA tools
Time budget: 30 minutes
"""

import time
import subprocess
import psutil
import torch
import numpy as np
from contextlib import contextmanager
from typing import Dict, List
import json
import os
from datetime import datetime

class GPUMonitor:
    """Monitor GPU memory and utilization"""
    
    def __init__(self):
        self.snapshots = []
    
    def snapshot(self, label: str = ""):
        """Take a snapshot of GPU stats"""
        if not torch.cuda.is_available():
            return
        
        stats = {
            'timestamp': time.time(),
            'label': label,
            'memory_allocated_gb': torch.cuda.memory_allocated() / 1e9,
            'memory_reserved_gb': torch.cuda.memory_reserved() / 1e9,
            'max_memory_allocated_gb': torch.cuda.max_memory_allocated() / 1e9,
        }
        
        # Try to get nvidia-smi info
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                gpu_util, mem_used, mem_total, temp = result.stdout.strip().split(',')
                stats['gpu_utilization_%'] = float(gpu_util)
                stats['memory_used_mb'] = float(mem_used)
                stats['memory_total_mb'] = float(mem_total)
                stats['temperature_c'] = float(temp)
        except Exception as e:
            pass
        
        self.snapshots.append(stats)
        return stats
    
    def save_snapshots(self, filepath: str):
        """Save all snapshots to JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.snapshots, f, indent=2)
    
    def print_summary(self):
        """Print summary of GPU usage"""
        if not self.snapshots:
            print("No GPU snapshots recorded")
            return
        
        max_mem = max(s['memory_allocated_gb'] for s in self.snapshots)
        avg_mem = np.mean([s['memory_allocated_gb'] for s in self.snapshots])
        
        print(f"\nGPU Memory Usage Summary:")
        print(f"  Max Allocated: {max_mem:.2f} GB")
        print(f"  Avg Allocated: {avg_mem:.2f} GB")
        
        if 'gpu_utilization_%' in self.snapshots[0]:
            avg_util = np.mean([s['gpu_utilization_%'] for s in self.snapshots])
            max_temp = max(s.get('temperature_c', 0) for s in self.snapshots)
            print(f"  Avg GPU Utilization: {avg_util:.1f}%")
            print(f"  Max Temperature: {max_temp:.1f}°C")

@contextmanager
def profile_section(name: str, gpu_monitor: GPUMonitor = None):
    """
    Context manager to profile a code section
    
    Usage:
        with profile_section("Training Epoch", gpu_monitor):
            # code to profile
    """
    print(f"\n[Profile] Starting: {name}")
    start_time = time.time()
    start_cpu = psutil.Process().cpu_percent()
    
    if gpu_monitor:
        gpu_monitor.snapshot(f"{name}_start")
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed = time.time() - start_time
        end_cpu = psutil.Process().cpu_percent()
        
        if gpu_monitor:
            gpu_monitor.snapshot(f"{name}_end")
        
        print(f"[Profile] Completed: {name}")
        print(f"  Time: {elapsed:.2f}s ({elapsed/60:.2f} min)")
        print(f"  CPU: {(start_cpu + end_cpu)/2:.1f}%")
        
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"  Peak GPU Memory: {peak_mem:.2f} GB")

def run_nsys_profile(command: str, output_path: str, duration_sec: int = 30):
    """
    Run NVIDIA Nsight Systems profiling
    
    Args:
        command: Python command to profile (e.g., "python train.py")
        output_path: Path to save .nsys-rep file
        duration_sec: How long to profile
    
    Note: Requires nsys to be installed
    """
    print(f"\n{'='*60}")
    print("Running Nsight Systems Profiling")
    print(f"{'='*60}")
    print(f"Command: {command}")
    print(f"Output: {output_path}")
    print(f"Duration: {duration_sec}s")
    
    nsys_cmd = [
        'nsys', 'profile',
        '--trace=cuda,nvtx,osrt',
        f'--duration={duration_sec}',
        '--stats=true',
        f'--output={output_path}',
        '--force-overwrite=true',
    ] + command.split()
    
    try:
        print("\nStarting profiler...")
        result = subprocess.run(
            nsys_cmd,
            capture_output=True,
            text=True,
            timeout=duration_sec + 60
        )
        
        if result.returncode == 0:
            print(f"✓ Profiling completed successfully")
            print(f"  Report saved to: {output_path}.nsys-rep")
            
            # Try to generate a summary
            summary_cmd = ['nsys', 'stats', f'{output_path}.nsys-rep']
            summary_result = subprocess.run(summary_cmd, capture_output=True, text=True)
            if summary_result.returncode == 0:
                print("\nProfile Summary:")
                print(summary_result.stdout[:1000])  # Print first 1000 chars
        else:
            print(f"✗ Profiling failed with return code {result.returncode}")
            print(f"Error: {result.stderr}")
    
    except subprocess.TimeoutExpired:
        print("✗ Profiling timed out")
    except FileNotFoundError:
        print("✗ nsys not found. Install NVIDIA Nsight Systems to use profiling.")
        print("  Visit: https://developer.nvidia.com/nsight-systems")
    except Exception as e:
        print(f"✗ Profiling error: {e}")

class ThroughputMeter:
    """Measure throughput (items/sec)"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.start_time = time.time()
        self.count = 0
    
    def update(self, n: int = 1):
        self.count += n
    
    def get_throughput(self) -> float:
        elapsed = time.time() - self.start_time
        return self.count / elapsed if elapsed > 0 else 0
    
    def __str__(self):
        return f"{self.get_throughput():.2f} items/sec"

def benchmark_inference(model, dataloader, num_batches: int = 10, 
                       warmup_batches: int = 2, device='cuda') -> Dict:
    """
    Benchmark inference throughput and latency
    
    Args:
        model: PyTorch model
        dataloader: DataLoader
        num_batches: Number of batches to benchmark
        warmup_batches: Number of warmup batches
    
    Returns:
        Dict with throughput and latency statistics
    """
    model.eval()
    
    print(f"\n{'='*60}")
    print("Benchmarking Inference")
    print(f"{'='*60}")
    
    # Warmup
    print(f"Warming up ({warmup_batches} batches)...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= warmup_batches:
                break
            if isinstance(batch, dict):
                images = batch['image'].to(device)
            else:
                images = batch[0].to(device)
            _ = model(images)
    
    # Benchmark
    print(f"Benchmarking ({num_batches} batches)...")
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    batch_times = []
    total_items = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            if isinstance(batch, dict):
                images = batch['image'].to(device)
            else:
                images = batch[0].to(device)
            
            batch_size = images.size(0)
            
            start = time.time()
            _ = model(images)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed = time.time() - start
            
            batch_times.append(elapsed)
            total_items += batch_size
    
    # Compute statistics
    batch_times = np.array(batch_times)
    
    results = {
        'mean_batch_time_ms': float(batch_times.mean() * 1000),
        'std_batch_time_ms': float(batch_times.std() * 1000),
        'median_batch_time_ms': float(np.median(batch_times) * 1000),
        'throughput_items_per_sec': float(total_items / batch_times.sum()),
        'total_items': total_items,
        'num_batches': num_batches
    }
    
    print(f"\nInference Benchmark Results:")
    print(f"  Mean Batch Time: {results['mean_batch_time_ms']:.2f} ± {results['std_batch_time_ms']:.2f} ms")
    print(f"  Median Batch Time: {results['median_batch_time_ms']:.2f} ms")
    print(f"  Throughput: {results['throughput_items_per_sec']:.2f} items/sec")
    
    return results

def create_performance_report(gpu_snapshots: List[Dict],
                             throughput_data: Dict,
                             training_times: Dict,
                             config,
                             save_path: str):
    """
    Create comprehensive performance report
    
    Args:
        gpu_snapshots: List of GPU monitoring snapshots
        throughput_data: Throughput measurements
        training_times: Training time breakdowns
        config: Configuration object
        save_path: Path to save report
    """
    report = {
        'hardware': {
            'device_type': config.hardware.device_type,
            'cuda_available': torch.cuda.is_available(),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',
            'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0,
        },
        'configuration': {
            'clip_batch_size': config.hardware.get_clip_batch_size(),
            'linear_probe_batch_size': config.hardware.get_linear_probe_bs(),
            'captioner_batch_size': config.hardware.get_captioner_bs(),
            'use_amp': config.model.use_amp,
            'use_gradient_checkpointing': config.hardware.use_gradient_checkpointing(),
        },
        'gpu_usage': {
            'max_memory_allocated_gb': max(s['memory_allocated_gb'] for s in gpu_snapshots) if gpu_snapshots else 0,
            'avg_memory_allocated_gb': np.mean([s['memory_allocated_gb'] for s in gpu_snapshots]) if gpu_snapshots else 0,
        },
        'throughput': throughput_data,
        'training_times': training_times,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nSaved performance report to {save_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("Performance Report Summary")
    print(f"{'='*60}")
    print(f"Hardware: {report['hardware']['gpu_name']}")
    print(f"Max GPU Memory: {report['gpu_usage']['max_memory_allocated_gb']:.2f} GB")
    print(f"Total Training Time: {sum(training_times.values()):.2f}s ({sum(training_times.values())/60:.2f} min)")
    
    return report
