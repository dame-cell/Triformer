import triton
import triton.language as tl
import torch
import torch.nn as nn
from torch.autograd import Function
import math
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import time


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_linear_relu_kernel(
    X, W, Y, B,
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

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = X + (offs_am[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    b_ptrs = W + (offs_k[:, None] * stride_wk + offs_bn[None, :] * stride_wn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0).to(tl.float16)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0).to(tl.float16)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_xk
        b_ptrs += BLOCK_SIZE_K * stride_wk

    c = accumulator.to(tl.float32)

    # Load and add bias
    bias = tl.load(B + offs_bn, mask=offs_bn < N, other=0.0).to(tl.float32)
    c += bias[None, :]

    # Apply ReLU activation
    c = tl.maximum(c, 0)

    # Convert to float16 after all computations
    c = c.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = Y + stride_ym * offs_cm[:, None] + stride_yn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def apply_relu_grad(grad_output, Y, num_elements, BLOCK_SIZE: tl.constexpr = 1024):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    x = tl.load(Y + offsets, mask=mask)
    grad = tl.load(grad_output + offsets, mask=mask)

    result = tl.where(x > 0, grad, 0.0)
    tl.store(grad_output + offsets, result, mask=mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32}),
        triton.Config({'BLOCK_SIZE_M': 32}),
        triton.Config({'BLOCK_SIZE_M': 32}),
    ],
    key=['M'],
)
@triton.jit
def backward_input_kernel(
    grad_output, weight, grad_input,
    M, N, K,
    stride_gom, stride_gon,
    stride_wk, stride_wn,
    stride_gim, stride_gik,
    BLOCK_SIZE_M: tl.constexpr,

):
    # Compute linear indices
    pid = tl.program_id(0)
    num_blocks_m = tl.cdiv(M, BLOCK_SIZE_M)
    
    # Block indices
    block_m = pid // K
    block_k = pid % K
    
    # Initialize offsets
    offs_m = block_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask = offs_m < M

    # Initialize accumulator
    acc = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    
    # Compute partial result
    for n in range(N):
        # Load grad_output
        g_ptr = grad_output + offs_m * stride_gom + n * stride_gon
        grad = tl.load(g_ptr, mask=mask, other=0.0)
        
        # Load weight
        w_ptr = weight + n * stride_wn + block_k * stride_wk
        w = tl.load(w_ptr)
        
        # Accumulate
        acc += grad * w

    # Write result
    out_ptr = grad_input + offs_m * stride_gim + block_k * stride_gik
    tl.store(out_ptr, acc, mask=mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 32,}),
        triton.Config({'BLOCK_SIZE_N': 32,}),
        triton.Config({'BLOCK_SIZE_N': 32,}),
    ],
    key=['N'],
)
@triton.jit
def backward_weight_kernel(
    grad_output, input, grad_weight,
    M, N, K,
    stride_gom, stride_gon,
    stride_im, stride_ik,
    stride_gwn, stride_gwk,

    BLOCK_SIZE_N: tl.constexpr,
):
    # Compute linear indices
    pid = tl.program_id(0)
    num_blocks_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Block indices
    block_n = pid // K
    block_k = pid % K
    
    # Initialize offsets
    offs_n = block_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = offs_n < N

    # Initialize accumulator
    acc = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    
    # Compute partial result
    for m in range(M):
        # Load grad_output
        g_ptr = grad_output + m * stride_gom + offs_n * stride_gon
        grad = tl.load(g_ptr, mask=mask, other=0.0)
        
        # Load input
        i_ptr = input + m * stride_im + block_k * stride_ik
        inp = tl.load(i_ptr)
        
        # Accumulate
        acc += grad * inp

    # Write result
    out_ptr = grad_weight + offs_n * stride_gwn + block_k * stride_gwk
    tl.store(out_ptr, acc, mask=mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32}),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32}),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32}),
    ],
    key=['M', 'N'],
)
@triton.jit
def fused_relu_bias_backward_kernel(
    grad_output, Y, grad_bias,
    M, N,
    stride_gom, stride_gon,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Get program ID and compute offsets
    pid = tl.program_id(0)
    num_blocks_n = tl.cdiv(N, BLOCK_SIZE_N)
    block_n = pid % num_blocks_n

    offs_n = block_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = offs_n < N

    # Initialize bias accumulator
    bias_acc = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)

    # Process each row (M dimension)
    for m in range(M):
        # Compute pointers for current row
        go_ptr = grad_output + m * stride_gom + offs_n * stride_gon
        y_ptr = Y + m * stride_gom + offs_n * stride_gon

        # Load values
        grad = tl.load(go_ptr, mask=mask, other=0.0)
        y = tl.load(y_ptr, mask=mask, other=0.0)

        grad = tl.where(y > 0, grad, 0.0)

        # Store modified gradient back
        tl.store(go_ptr, grad, mask=mask)

        # Accumulate for bias gradient
        bias_acc += grad

    # Store bias gradients
    tl.store(grad_bias + offs_n, bias_acc, mask=mask)



class TritonLinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        M, K = input.shape
        N, K = weight.shape

        # Output tensor
        Y = torch.empty((M, N), device=input.device, dtype=torch.float16)

        # Run kernel
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        )
        fused_linear_relu_kernel[grid](
            input, weight.t(), Y, bias,
            M, N, K,
            input.stride(0), input.stride(1),
            weight.stride(1), weight.stride(0),
            Y.stride(0), Y.stride(1),
        )

        ctx.save_for_backward(input, weight, bias)
        ctx.Y = Y
        return Y

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        Y = ctx.Y
        M, K = input.shape
        N, K = weight.shape

        # Compute bias gradients and apply ReLU gradient in-place
        grad_bias = torch.empty_like(bias)
        grid_fused = lambda META: (
            triton.cdiv(N, META['BLOCK_SIZE_N']),
        )
        fused_relu_bias_backward_kernel[grid_fused](
            grad_output, Y, grad_bias,
            M, N,
            grad_output.stride(0), grad_output.stride(1),
        )

        # Compute input gradients
        grad_input = torch.empty_like(input)
        grid_input = lambda META: (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * K,
        )
        backward_input_kernel[grid_input](
            grad_output, weight, grad_input,
            M, N, K,
            grad_output.stride(0), grad_output.stride(1),
            weight.stride(1), weight.stride(0),
            grad_input.stride(0), grad_input.stride(1),
        )

        # Compute weight gradients
        grad_weight = torch.empty_like(weight)
        grid_weight = lambda META: (
            triton.cdiv(N, META['BLOCK_SIZE_N']) * K,
        )
        backward_weight_kernel[grid_weight](
            grad_output, input, grad_weight,
            M, N, K,
            grad_output.stride(0), grad_output.stride(1),
            input.stride(0), input.stride(1),
            grad_weight.stride(0), grad_weight.stride(1),
        )

        return grad_input, grad_weight, grad_bias
        
class TritonLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device='cuda', dtype=torch.float16)
        )
        self.bias = nn.Parameter(
            torch.zeros(out_features, device='cuda', dtype=torch.float16)
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return TritonLinearFunction.apply(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}'
    
    