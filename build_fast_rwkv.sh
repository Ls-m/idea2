#!/bin/bash

# Build script for RWKV CUDA kernels
echo "Building optimized RWKV CUDA kernels..."

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo "NVCC not found. Please install CUDA toolkit."
    exit 1
fi

# Set environment variables
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Get PyTorch version and CUDA version
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"

# Build the extension
cd src/models

python -c "
import torch
from torch.utils.cpp_extension import load
import os

print('Compiling RWKV CUDA kernels...')

# Compile the CUDA extension
rwkv_cuda = load(
    name='rwkv_cuda',
    sources=['wkv_kernels.cu'],
    extra_cflags=['-O3'],
    extra_cuda_cflags=[
        '-O3',
        '-use_fast_math',
        '--expt-relaxed-constexpr',
        '-Xcompiler', '-fPIC'
    ],
    verbose=True
)

print('✓ CUDA kernels compiled successfully!')
print('Testing kernels...')

# Quick test
import torch
B, T, C = 2, 100, 64
device = torch.device('cuda')

w = torch.randn(C, device=device)
u = torch.randn(C, device=device) 
k = torch.randn(B, T, C, device=device)
v = torch.randn(B, T, C, device=device)
state = torch.randn(B, C, 2, device=device)

try:
    y, new_state = rwkv_cuda.wkv_forward(w, u, k, v, state)
    print(f'✓ Forward pass successful: {y.shape}, {new_state.shape}')
except Exception as e:
    print(f'✗ Forward pass failed: {e}')
"

echo "Build complete!"