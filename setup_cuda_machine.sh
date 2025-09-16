#!/bin/bash

# Setup script for optimized RWKV on CUDA machine
echo "🚀 Setting up optimized RWKV training on CUDA machine..."

# 1. Check CUDA availability
echo "Checking CUDA setup..."
nvidia-smi
nvcc --version
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# 2. Install dependencies
echo "Installing dependencies..."
pip install torch>=2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pytorch-lightning>=2.0.0
pip install ninja  # For fast CUDA compilation
pip install numpy scipy scikit-learn pandas matplotlib seaborn
pip install tensorboard

# 3. Apply memory optimizations
echo "Applying CUDA optimizations..."
python -c "
from src.utils.memory_utils import optimize_memory_usage
optimize_memory_usage()
print('✅ Memory optimizations applied')
"

# 4. Build CUDA kernels
echo "Building CUDA kernels..."
cd src/models
python -c "
import torch
from torch.utils.cpp_extension import load
import os

if torch.cuda.is_available():
    print('Compiling RWKV CUDA kernels...')
    try:
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
        print('✅ CUDA kernels compiled successfully!')
        
        # Quick test
        B, T, C = 2, 100, 64
        device = torch.device('cuda')
        w = torch.randn(C, device=device)
        u = torch.randn(C, device=device) 
        k = torch.randn(B, T, C, device=device)
        v = torch.randn(B, T, C, device=device)
        state = torch.randn(B, C, 2, device=device)
        
        y, new_state = rwkv_cuda.wkv_forward(w, u, k, v, state)
        print(f'✅ Kernel test successful: {y.shape}, {new_state.shape}')
        
    except Exception as e:
        print(f'⚠️  CUDA kernel compilation failed: {e}')
        print('Falling back to optimized PyTorch implementation')
else:
    print('⚠️  CUDA not available, using CPU mode')
"

cd ../..

# 5. Test performance
echo "Testing optimized model performance..."
python benchmark_rwkv.py

echo "🎯 Setup complete! Ready for training."
echo ""
echo "To start training:"
echo "python src/main.py --config src/config/config.yaml --stage contrastive"
echo ""
echo "For full pipeline:"
echo "python src/main.py --config src/config/config.yaml --stage both"
