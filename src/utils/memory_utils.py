"""
Memory optimization utilities for RWKV training.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import gc

class MemoryOptimizedRWKV(nn.Module):
    """Memory-optimized wrapper for RWKV with gradient checkpointing."""
    
    def __init__(self, model, checkpoint_segments=2):
        super().__init__()
        self.model = model
        self.checkpoint_segments = checkpoint_segments
        
        # Split model into segments for checkpointing
        layers_per_segment = len(model.blocks) // checkpoint_segments
        self.segments = []
        
        for i in range(checkpoint_segments):
            start_idx = i * layers_per_segment
            end_idx = (i + 1) * layers_per_segment if i < checkpoint_segments - 1 else len(model.blocks)
            self.segments.append(nn.Sequential(*model.blocks[start_idx:end_idx]))
    
    def forward(self, x, state=None):
        """Forward with gradient checkpointing."""
        # Input projection
        x = self.model.input_proj(x)
        
        # Process segments with checkpointing
        states = []
        for i, segment in enumerate(self.segments):
            if self.training:
                # Use gradient checkpointing during training
                def segment_forward(x_seg, state_seg):
                    for j, block in enumerate(segment):
                        block_state = state_seg[j] if state_seg is not None else None
                        x_seg, new_state = block(x_seg, block_state)
                        states.append(new_state)
                    return x_seg
                
                x = checkpoint(segment_forward, x, state, use_reentrant=False)
            else:
                # Normal forward during inference
                for j, block in enumerate(segment):
                    block_state = state[i * len(segment) + j] if state is not None else None
                    x, new_state = block(x, block_state)
                    states.append(new_state)
        
        # Output processing
        x = self.model.ln_out(x)
        return self.model.head_norm(x[:, -1, :])

def optimize_memory_usage():
    """Set optimal memory settings for training."""
    if torch.cuda.is_available():
        # Enable memory efficient attention
        torch.backends.cuda.enable_flash_sdp(True)
        
        # Set memory fraction (use 90% of GPU memory)
        torch.cuda.set_per_process_memory_fraction(0.9)
        
        # Enable cudnn benchmark for consistent input sizes
        torch.backends.cudnn.benchmark = True
        
        # Disable deterministic operations for speed (if reproducibility not critical)
        # torch.backends.cudnn.deterministic = False
        
        print("✅ CUDA memory optimizations enabled")
    
    # Enable mixed precision training
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    print("✅ Mixed precision optimizations enabled")

def clear_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()

class MemoryMonitor:
    """Monitor memory usage during training."""
    
    def __init__(self):
        self.peak_memory = 0
        self.current_memory = 0
        
    def update(self):
        if torch.cuda.is_available():
            self.current_memory = torch.cuda.memory_allocated() / 1e9  # GB
            self.peak_memory = max(self.peak_memory, self.current_memory)
    
    def report(self):
        if torch.cuda.is_available():
            print(f"Current GPU memory: {self.current_memory:.2f}GB")
            print(f"Peak GPU memory: {self.peak_memory:.2f}GB")
            print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")

# Optimized training configuration
OPTIMIZED_TRAINING_CONFIG = {
    "use_amp": True,  # Automatic Mixed Precision
    "gradient_checkpointing": True,
    "max_grad_norm": 1.0,
    "accumulate_grad_batches": 2,  # Gradient accumulation
    "precision": 16,  # FP16 training
    "find_unused_parameters": False,  # For DDP
}

def create_memory_efficient_model(base_model, training_config=None):
    """Create memory-efficient version of model."""
    config = training_config or OPTIMIZED_TRAINING_CONFIG
    
    if config.get("gradient_checkpointing", False):
        model = MemoryOptimizedRWKV(base_model, checkpoint_segments=2)
    else:
        model = base_model
    
    # Enable gradient checkpointing on the model itself
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    return model

if __name__ == "__main__":
    print("🧠 Memory Optimization Utilities")
    print("=" * 40)
    
    # Apply optimizations
    optimize_memory_usage()
    
    # Create memory monitor
    monitor = MemoryMonitor()
    monitor.update()
    monitor.report()
    
    print("\n✅ Memory optimizations ready!")
    print("Use these in your training script:")
    print("from utils.memory_utils import optimize_memory_usage, MemoryMonitor")
    print("optimize_memory_usage()")
    print("monitor = MemoryMonitor()")
