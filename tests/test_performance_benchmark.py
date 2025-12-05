"""
Simple CPU vs GPU Training Time Benchmark

Measures and compares training time on CPU vs GPU.
Runs multiple iterations to get reliable average results.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import torch
import time

from src.volforecast.models.deep_econo_net import DeepEconoNet, DeepEconoNetConfig


def benchmark_training(device: str, num_runs: int = 3, num_samples: int = 300):
    """
    Benchmark training time on a specific device.
    
    Args:
        device: "cpu" or "cuda"
        num_runs: Number of runs to average
        num_samples: Number of samples to train on
        
    Returns:
        Dictionary with timing statistics
    """
    times = []
    
    for run in range(num_runs):
        print(f"  Run {run + 1}/{num_runs}...", end=" ", flush=True)
        
        # Create fresh model for each run
        config = DeepEconoNetConfig(
            device=device,
            return_col="log_return"
        )
        model = DeepEconoNet(config)
        
        # Load data
        data_path = os.path.join(os.path.dirname(__file__), "..", "data/stock_info", "ADSK_dataset.csv")
        df = pd.read_csv(data_path).iloc[:num_samples].copy()
        
        # Synchronize if using GPU
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        
        # Measure training time
        start = time.time()
        model.fit_ticker(df, ticker=f"BENCH_{device}_{run}")
        elapsed = time.time() - start
        
        # Synchronize if using GPU
        if device == "cuda":
            torch.cuda.synchronize()
        
        times.append(elapsed)
        print(f"{elapsed:.2f}s")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    return {
        "avg": avg_time,
        "std": std_time,
        "min": min_time,
        "max": max_time,
        "times": times
    }


def print_results(cpu_results, gpu_results=None):
    """Pretty print benchmark results."""
    print("\n" + "=" * 70)
    print("TRAINING TIME BENCHMARK RESULTS")
    print("=" * 70)
    
    print("\nCPU Training:")
    print(f"  Average: {cpu_results['avg']:.3f}s")
    print(f"  Std Dev: ±{cpu_results['std']:.3f}s")
    print(f"  Range:   {cpu_results['min']:.3f}s - {cpu_results['max']:.3f}s")
    print(f"  Runs:    {cpu_results['times']}")
    
    if gpu_results:
        print("\nGPU Training:")
        print(f"  Average: {gpu_results['avg']:.3f}s")
        print(f"  Std Dev: ±{gpu_results['std']:.3f}s")
        print(f"  Range:   {gpu_results['min']:.3f}s - {gpu_results['max']:.3f}s")
        print(f"  Runs:    {gpu_results['times']}")
        
        speedup = cpu_results['avg'] / gpu_results['avg']
        improvement = (speedup - 1) * 100
        
        print("\n" + "-" * 70)
        print(f"GPU Speedup: {speedup:.2f}x faster")
        print(f"Improvement: {improvement:+.1f}%")
        print("=" * 70)
        
        return speedup
    
    return None


def propose_improvements():
    """Print optimization recommendations based on results."""
    print("\n" + "=" * 70)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("=" * 70)
    
    recommendations = [
        {
            "title": "Increase Batch Size",
            "description": "Larger batches improve GPU utilization and throughput",
            "implementation": "Set batch_size=64 or higher in config"
        },
        {
            "title": "Use Mixed Precision Training",
            "description": "Use float16 for faster compute on GPU while maintaining accuracy",
            "implementation": "Add automatic mixed precision (torch.autocast) to forward pass"
        },
        {
            "title": "Data Pipeline Optimization",
            "description": "Prefetch data and use multiple workers for parallel loading",
            "implementation": "Set num_workers>0 in DataLoader, pin_memory=True already enabled"
        },
        {
            "title": "Gradient Accumulation",
            "description": "Simulate larger batch sizes without OOM via gradient accumulation",
            "implementation": "Use config.gradient_accumulation_steps parameter (already in config)"
        },
        {
            "title": "Model Optimization",
            "description": "Profile and optimize bottleneck layers",
            "implementation": "Use torch.profiler to identify slow operations"
        }
    ]
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['title']}")
        print(f"   → {rec['description']}")
        print(f"   → {rec['implementation']}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CPU vs GPU TRAINING TIME BENCHMARK")
    print("=" * 70)
    
    # Configuration
    num_runs = 10
    num_samples = 300
    
    print(f"\nConfiguration:")
    print(f"  Samples: {num_samples}")
    print(f"  Epochs: 2")
    print(f"  Batch Size: 32")
    print(f"  Runs per device: {num_runs}")
    
    print(f"\nSystem Info:")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  CPU Cores: {os.cpu_count()}")
    
    # Benchmark CPU
    print(f"\n{'='*70}")
    print("Benchmarking CPU...")
    print(f"{'='*70}")
    cpu_results = benchmark_training("cpu", num_runs=num_runs, num_samples=num_samples)
    
    # Benchmark GPU if available
    gpu_results = None
    speedup = None
    
    if torch.cuda.is_available():
        print(f"\n{'='*70}")
        print("Benchmarking GPU...")
        print(f"{'='*70}")
        gpu_results = benchmark_training("cuda", num_runs=num_runs, num_samples=num_samples)
        speedup = print_results(cpu_results, gpu_results)
    else:
        print("\n⚠ GPU not available, skipping GPU benchmark")
        print_results(cpu_results)
    
    # Print recommendations
    propose_improvements()
    
    print("\n" + "=" * 70)
    
    if speedup and speedup > 1.0:
        print(f"✅ GPU acceleration is working! {speedup:.2f}x speedup achieved.")
    elif speedup:
        print(f"⚠ GPU slower than CPU for this workload ({speedup:.2f}x)")
        print("  This can happen with small datasets due to transfer overhead.")
    else:
        print("✅ CPU benchmark complete. GPU benchmark skipped.")
    
    print("=" * 70 + "\n")
    
    sys.exit(0)
