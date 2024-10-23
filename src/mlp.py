import triton
import triton.language as tl
import torch
import torch.nn as nn
import time 
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_linear_relu_kernel(
    X, W, Y,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_ym, stride_yn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = X + (offs_am[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    b_ptrs = W + (offs_k[:, None] * stride_wk + offs_bn[None, :] * stride_wn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_xk
        b_ptrs += BLOCK_SIZE_K * stride_wk
    
    c = accumulator.to(tl.float16)
    
    # Apply ReLU activation
    c = tl.where(c > 0, c, 0.0)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = Y + stride_ym * offs_cm[:, None] + stride_yn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

# Wrapper function to launch the kernel
def fused_linear_relu(X, W, Y):
    M, K = X.shape
    K, N = W.shape

    # Launch kernel
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    fused_linear_relu_kernel[grid](
        X, W, Y,
        M, N, K,
        X.stride(0), X.stride(1),
        W.stride(0), W.stride(1),
        Y.stride(0), Y.stride(1),
    )


class TritonLinearReLU(nn.Module):
    def __init__(self, in_features, out_features):
        super(TritonLinearReLU, self).__init__()
        self.weight = nn.Parameter(torch.nn.init.kaiming_normal_(
            torch.empty(in_features, out_features, device='cuda'),
            mode='fan_in',
            nonlinearity='relu'
        ))
        self.bias = nn.Parameter(torch.zeros(out_features, device='cuda'))

    def forward(self, x):
        Y = torch.empty(x.shape[0], self.weight.shape[1], device='cuda', dtype=torch.float16)
        fused_linear_relu(x, self.weight, Y)
        return Y + self.bias