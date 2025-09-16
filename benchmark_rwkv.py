"""
Performance optimization script for RWKV model.
"""

import torch
import torch.nn as nn
import time
import numpy as np
from models.optimized_rwkv import OptimizedRWKV, create_optimized_rwkv
from models.rwkv import RWKV

def benchmark_model(model, input_tensor, num_runs=10, warmup_runs=3):
    """Benchmark a model's forward pass performance."""
    model.eval()
    device = next(model.parameters()).device
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)
    
    # Sync for accurate timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            output = model(input_tensor)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs
    
    return avg_time, output

def compare_models():
    """Compare original vs optimized RWKV performance."""
    print("🚀 RWKV Performance Comparison")
    print("=" * 50)
    
    # Test configuration
    batch_size = 8
    seq_len = 3750  # 30 seconds at 125Hz  
    input_size = 1
    hidden_size = 256
    num_layers = 6
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Input shape: ({batch_size}, {seq_len}, {input_size})")
    print(f"Model config: {num_layers} layers, {hidden_size} hidden size")
    print()
    
    # Test data
    x = torch.randn(batch_size, seq_len, input_size, device=device)
    
    # Original RWKV
    print("1️⃣ Original RWKV Model")
    original_model = RWKV(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.0
    ).to(device)
    
    original_params = sum(p.numel() for p in original_model.parameters())
    original_time, original_output = benchmark_model(original_model, x)
    
    print(f"   Parameters: {original_params:,}")
    print(f"   Forward time: {original_time:.4f}s")
    print(f"   Throughput: {batch_size/original_time:.2f} samples/sec")
    print(f"   Output shape: {original_output.shape}")
    
    if device.type == 'cuda':
        original_memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"   GPU memory: {original_memory:.2f}GB")
        torch.cuda.reset_peak_memory_stats()
    
    print()
    
    # Optimized RWKV
    print("2️⃣ Optimized RWKV Model")
    optimized_model = create_optimized_rwkv(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.0,
        compile_kernels=True
    ).to(device)
    
    optimized_params = sum(p.numel() for p in optimized_model.parameters())
    optimized_time, optimized_output = benchmark_model(optimized_model, x)
    
    print(f"   Parameters: {optimized_params:,}")
    print(f"   Forward time: {optimized_time:.4f}s")
    print(f"   Throughput: {batch_size/optimized_time:.2f} samples/sec")
    print(f"   Output shape: {optimized_output.shape}")
    
    if device.type == 'cuda':
        optimized_memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"   GPU memory: {optimized_memory:.2f}GB")
    
    print()
    
    # Performance comparison
    print("📊 Performance Comparison")
    print("-" * 30)
    speedup = original_time / optimized_time
    throughput_improvement = (batch_size/optimized_time) / (batch_size/original_time)
    
    print(f"Speedup: {speedup:.2f}x")
    print(f"Throughput improvement: {throughput_improvement:.2f}x")
    
    if device.type == 'cuda':
        memory_reduction = (original_memory - optimized_memory) / original_memory * 100
        print(f"Memory reduction: {memory_reduction:.1f}%")
    
    # Accuracy check
    output_diff = torch.abs(original_output - optimized_output).max().item()
    print(f"Max output difference: {output_diff:.2e}")
    
    if output_diff < 1e-4:
        print("✅ Outputs match (numerically stable)")
    else:
        print("⚠️  Outputs differ (check implementation)")

def profile_memory_usage():
    """Profile memory usage of optimized model."""
    if not torch.cuda.is_available():
        print("CUDA not available for memory profiling")
        return
    
    print("\n🧠 Memory Usage Profiling")
    print("=" * 30)
    
    device = torch.device('cuda')
    batch_sizes = [1, 2, 4, 8, 16]
    seq_len = 3750
    
    for batch_size in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        model = create_optimized_rwkv(
            input_size=1,
            hidden_size=256, 
            num_layers=6,
            dropout=0.0
        ).to(device)
        
        x = torch.randn(batch_size, seq_len, 1, device=device)
        
        with torch.no_grad():
            _ = model(x)
        
        memory_mb = torch.cuda.max_memory_allocated() / 1e6
        print(f"Batch size {batch_size:2d}: {memory_mb:6.1f} MB")

if __name__ == "__main__":
    # Run performance comparison
    compare_models()
    
    # Profile memory usage
    profile_memory_usage()
    
    print("\n🎯 Optimization Complete!")
    print("To use the optimized model in your training:")
    print("from models.optimized_rwkv import create_optimized_rwkv")
    print("model = create_optimized_rwkv(input_size=1, hidden_size=256, num_layers=6)")
