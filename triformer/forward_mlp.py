import triton
import triton.language as tl
import torch
import torch.nn as nn
from torch.autograd import Function
import math

@triton.autotune(
    configs=[
        # Optimized for larger matrices
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        # Optimized for medium matrices
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # Optimized for smaller matrices
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def bmm_kernel(
    x_ptr, y_ptr, o_ptr,
    M, N, K,
    stride_al, stride_am, stride_ak,
    stride_bl, stride_bk, stride_bn,
    stride_ol, stride_om, stride_on,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    USE_RELU: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid = tl.program_id(1)

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
    x_ptrs = x_ptr + (offs_am[:, None]*stride_am + offs_k[None, :]*stride_ak + pid_batch*stride_al)
    y_ptrs = y_ptr + (offs_k[:, None]*stride_bk + offs_bn[None, :]*stride_bn + pid_batch*stride_bl)

    o = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        x = tl.load(x_ptrs)
        y = tl.load(y_ptrs)
        o += tl.dot(x, y)

        x_ptrs += BLOCK_SIZE_K * stride_ak
        y_ptrs += BLOCK_SIZE_K * stride_bk

    if USE_RELU:
        o = tl.maximum(o, 0.0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    o_ptrs = o_ptr + stride_om * offs_m[:, None] + stride_on * offs_n[None, :] + stride_ol * pid_batch
    tl.store(o_ptrs, o, mask=mask)

def triton_bmm(x, y, use_relu=False):
    B, M, K = x.shape

    if y.ndim == 2:
        y = y.unsqueeze(0).expand(B, -1, -1)

    _, K, N = y.shape
    assert (K % 32 == 0), "K must be divisible by 32"

    o = torch.empty((B, M, N), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (B, triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']))

    bmm_kernel[grid](
        x, y, o,
        M, N, K,
        x.stride(0), x.stride(1), x.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        USE_RELU=use_relu,
    )
    return o

class TritonLinearFunction(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, use_relu=True):
        # Add batch dimension if necessary
        if x.ndim == 2:
            x = x.unsqueeze(0)
            
        # Prepare weight with correct shape
        weight = weight.t().contiguous()
        
        # Apply the main computation
        output = triton_bmm(x, weight, use_relu=use_relu)
        
        # Add bias
        if bias is not None:
            output += bias.view(1, 1, -1)
            
        # Remove batch dimension if it was added
        if x.ndim == 3 and x.size(0) == 1:
            output = output.squeeze(0)
            
        if x.requires_grad:
            ctx.save_for_backward(x, weight, output)
            ctx.use_relu = use_relu
            
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, output = ctx.saved_tensors
        use_relu = ctx.use_relu

        if use_relu:
            grad_output = grad_output * (output > 0)

        # Add batch dimension if necessary
        if grad_output.ndim == 2:
            grad_output = grad_output.unsqueeze(0)
            
        # Compute gradients
        grad_input = triton_bmm(grad_output, weight)
        grad_weight = triton_bmm(x.transpose(-1, -2), grad_output).transpose(-1, -2)
        grad_bias = grad_output.sum((0, 1)) if grad_output.ndim == 3 else grad_output.sum(0)

        # Remove batch dimension if it was added
        if x.ndim == 2:
            grad_input = grad_input.squeeze(0)

        return grad_input, grad_weight, grad_bias, None

class TritonLinear(nn.Module):
    def __init__(self, in_features, out_features, use_relu=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_relu = use_relu
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
        return TritonLinearFunction.apply(x, self.weight, self.bias, self.use_relu)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}'
