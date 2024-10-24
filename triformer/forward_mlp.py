import triton
import triton.language as tl
import torch
import torch.nn as nn
from torch.autograd import Function
import math 

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


class TritonLinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight, bias)
        M, K = input.shape
        N, K = weight.shape

        Y = torch.empty((M, N), device=input.device, dtype=torch.float16)

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

        torch.cuda.synchronize()

    

        ctx.Y = Y
        return Y

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, _ = ctx.saved_tensors
        Y = ctx.Y  # Retrieve saved output
        
        # Ensure consistent dtypes
        grad_output = grad_output.to(torch.float32)
        input = input.to(torch.float32)
        weight = weight.to(torch.float32)
        
        # Apply ReLU gradient
        relu_mask = (Y > 0).float()
        grad_output = grad_output * relu_mask
        
        # Compute gradients using PyTorch operations
        grad_input = grad_output.mm(weight)
        grad_weight = grad_output.t().mm(input)
        grad_bias = grad_output.sum(0)
        
        # Convert gradients back to float16
        grad_input = grad_input.to(torch.float16)
        grad_weight = grad_weight.to(torch.float16)
        grad_bias = grad_bias.to(torch.float16)
    
        return grad_input, grad_weight, grad_bias

class TritonLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(TritonLinear, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device='cuda', dtype=torch.float16))
        self.bias = nn.Parameter(torch.zeros(out_features, device='cuda', dtype=torch.float16))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return TritonLinearFunction.apply(x, self.weight, self.bias)

