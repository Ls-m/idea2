// CUDA kernels for optimized RWKV operations
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cooperative_groups.h>

using namespace cooperative_groups;

// Optimized WKV forward kernel with memory coalescing
__global__ void wkv_forward_kernel(
    const float* __restrict__ w,      // (C,) time decay
    const float* __restrict__ u,      // (C,) time first  
    const float* __restrict__ k,      // (B, T, C) keys
    const float* __restrict__ v,      // (B, T, C) values
    const float* __restrict__ state,  // (B, C, 2) previous state or nullptr
    float* __restrict__ y,            // (B, T, C) output
    float* __restrict__ new_state,    // (B, C, 2) new state
    int B, int T, int C
) {
    // Each thread handles one (batch, channel) pair
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C) return;
    
    int b = idx / C;
    int c = idx % C;
    int bc_offset = b * C + c;
    
    // Load parameters
    float ww = w[c];
    float uu = u[c];
    
    // Initialize state
    float aa, bb;
    if (state) {
        aa = state[b * C * 2 + c * 2 + 0];
        bb = state[b * C * 2 + c * 2 + 1];
    } else {
        aa = 0.0f;
        bb = -1e38f;
    }
    
    // Process time steps
    for (int t = 0; t < T; t++) {
        int ktv_idx = b * T * C + t * C + c;
        float kk = k[ktv_idx];
        float vv = v[ktv_idx];
        
        // WKV computation with numerical stability
        float p = fmaxf(bb, uu + kk);
        float e1 = expf(bb - p);
        float e2 = expf(uu + kk - p);
        float inv_sum = 1.0f / (e1 + e2 + 1e-8f);
        
        // Output
        y[ktv_idx] = (e1 * aa + e2 * vv) * inv_sum;
        
        // State update
        float p2 = fmaxf(ww + bb, kk);
        float e3 = expf(ww + bb - p2);
        float e4 = expf(kk - p2);
        
        aa = e3 * aa + e4 * vv;
        bb = p2 + logf(e3 + e4 + 1e-8f);
    }
    
    // Store final state
    new_state[b * C * 2 + c * 2 + 0] = aa;
    new_state[b * C * 2 + c * 2 + 1] = bb;
}

// Backward kernel for WKV operation
__global__ void wkv_backward_kernel(
    const float* __restrict__ w,
    const float* __restrict__ u,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ state,
    const float* __restrict__ y,
    const float* __restrict__ grad_y,
    float* __restrict__ grad_w,
    float* __restrict__ grad_u,
    float* __restrict__ grad_k,
    float* __restrict__ grad_v,
    int B, int T, int C
) {
    // Simplified backward pass - can be optimized further
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * T * C) return;
    
    int b = idx / (T * C);
    int t = (idx % (T * C)) / C;
    int c = idx % C;
    
    // Simple gradient computation (placeholder)
    atomicAdd(&grad_k[idx], grad_y[idx]);
    atomicAdd(&grad_v[idx], grad_y[idx]);
}

// Host functions
std::tuple<torch::Tensor, torch::Tensor> wkv_forward_cuda(
    torch::Tensor w,     // (C,)
    torch::Tensor u,     // (C,)
    torch::Tensor k,     // (B, T, C)
    torch::Tensor v,     // (B, T, C)
    torch::Tensor state  // (B, C, 2) or empty
) {
    auto B = k.size(0);
    auto T = k.size(1);
    auto C = k.size(2);
    
    auto options = torch::TensorOptions()
        .dtype(k.dtype())
        .device(k.device());
    
    auto y = torch::empty({B, T, C}, options);
    auto new_state = torch::empty({B, C, 2}, options);
    
    const int threads = 256;
    const int blocks = (B * C + threads - 1) / threads;
    
    float* state_ptr = state.numel() > 0 ? state.data_ptr<float>() : nullptr;
    
    wkv_forward_kernel<<<blocks, threads>>>(
        w.data_ptr<float>(),
        u.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        state_ptr,
        y.data_ptr<float>(),
        new_state.data_ptr<float>(),
        B, T, C
    );
    
    cudaDeviceSynchronize();
    
    return std::make_tuple(y, new_state);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> wkv_backward_cuda(
    torch::Tensor w,
    torch::Tensor u, 
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor state,
    torch::Tensor y,
    torch::Tensor grad_y
) {
    auto B = k.size(0);
    auto T = k.size(1);
    auto C = k.size(2);
    
    auto grad_w = torch::zeros_like(w);
    auto grad_u = torch::zeros_like(u);
    auto grad_k = torch::zeros_like(k);
    auto grad_v = torch::zeros_like(v);
    
    const int threads = 256;
    const int blocks = (B * T * C + threads - 1) / threads;
    
    float* state_ptr = state.numel() > 0 ? state.data_ptr<float>() : nullptr;
    
    wkv_backward_kernel<<<blocks, threads>>>(
        w.data_ptr<float>(),
        u.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        state_ptr,
        y.data_ptr<float>(),
        grad_y.data_ptr<float>(),
        grad_w.data_ptr<float>(),
        grad_u.data_ptr<float>(),
        grad_k.data_ptr<float>(),
        grad_v.data_ptr<float>(),
        B, T, C
    );
    
    cudaDeviceSynchronize();
    
    return std::make_tuple(grad_w, grad_u, grad_k, grad_v);
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wkv_forward", &wkv_forward_cuda, "WKV forward (CUDA)");
    m.def("wkv_backward", &wkv_backward_cuda, "WKV backward (CUDA)");
}
