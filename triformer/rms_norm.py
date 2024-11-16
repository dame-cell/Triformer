import torch
import triton
import triton.language as tl
from .utils import calculate_settings 

@triton.jit
def rmsnorm_forward(
    Y, Y_row_stride,
    X, X_row_stride,
    W,
    r, 
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y_ptr = Y + row_idx * Y_row_stride
    X_ptr = X + row_idx * X_row_stride
    r_ptr = r + row_idx

    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)

    X_squared = X_row * X_row
    mean_X_squared = tl.sum(X_squared, axis=0) / n_cols
    rms = tl.math.rsqrt(mean_X_squared + eps)
    tl.store(r_ptr, rms)
    output = X_row * rms * W_row
    tl.store(Y_ptr + col_offsets, output, mask=mask)

#autotuning for better backward performance
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128, 'NUM_WARPS': 4}),
        triton.Config({'BLOCK_SIZE': 256, 'NUM_WARPS': 8}),
        triton.Config({'BLOCK_SIZE': 512, 'NUM_WARPS': 16}),
        triton.Config({'BLOCK_SIZE': 1024, 'NUM_WARPS': 16}),
        triton.Config({'BLOCK_SIZE': 2048, 'NUM_WARPS': 32}),
        triton.Config({'BLOCK_SIZE': 4096, 'NUM_WARPS': 32}),
        triton.Config({'BLOCK_SIZE': 8192, 'NUM_WARPS': 48}),
    ],
    key=['n_cols']
)
@triton.jit
def _rms_layernorm_backward(
    dY, dY_row_stride,
    X,  X_row_stride,
    W,  W_row_stride,
    r,  r_row_stride,
    dX, dX_row_stride,
    dW,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
    NUM_WARPS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dY_ptr = dY + pid * dY_row_stride + col_offsets
    X_ptr = X + pid * X_row_stride + col_offsets
    dX_ptr = dX + pid * dX_row_stride + col_offsets

    dY_row = tl.load(dY_ptr, mask=mask, other=0).to(tl.float32)
    X_row = tl.load(X_ptr, mask=mask, other=0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)
    rms = tl.load(r + pid).to(tl.float32)

    X_norm = X_row * rms
    dY_W = dY_row * W_row
    sum_dY_X = tl.sum(dY_W * X_norm, axis=0)
    dX = rms * (dY_W - X_norm * (sum_dY_X / n_cols))
    dW_row = (dY_row * X_norm)
    tl.atomic_add(dW + col_offsets, dW_row, mask=mask)
    tl.store(dX_ptr, dX, mask=mask)


class Fast_RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, eps):
        shape = X.shape
        dim = shape[-1]
        X = X.contiguous().view(-1, dim)
        n_rows, n_cols = X.shape
        
        BLOCK_SIZE , num_warps = calculate_settings(n_cols)

        Y = torch.empty_like(X)
        r = torch.empty(n_rows, dtype=torch.float32, device=X.device)

        rmsnorm_forward[(n_rows,)](
            Y, Y.stride(0),
            X, X.stride(0),
            W,
            r,
            n_cols, eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        
        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.save_for_backward(X, W, r)
        
        return Y.view(*shape)


    @staticmethod
    def backward(ctx, dY):
        X, W, r = ctx.saved_tensors
        shape = dY.shape
        dim = shape[-1]
        dY = dY.contiguous().view(-1, dim)
        n_rows, n_cols = dY.shape

       
        
        # Allocate output buffers
        dX = torch.empty_like(dY)
        dW = torch.zeros_like(W)  # For atomic adds
        
        # Launch kernel with optimized grid
        grid = (n_rows,)
        
        _rms_layernorm_backward[grid](
            dY, dY.stride(0),
            X, X.stride(0),
            W, W.stride(0),
            r, r.stride(0),
            dX, dX.stride(0),
            dW,
            n_cols, ctx.eps,
            #BLOCK_SIZE=BLOCK_SIZE,
            #num_warps=num_warps,
        )

        return dX.view(*shape), dW, None

def fast_layernorm(rmsnorm, X):
    W = rmsnorm.weight
    eps = rmsnorm.variance_epsilon if hasattr(rmsnorm, "variance_epsilon") else rmsnorm.eps
    out = Fast_RMSNorm.apply(X, W, eps)
    return out

from transformers.models.llama.modeling_llama import LlamaRMSNorm
class TritonRMSNorm(LlamaRMSNorm):
    def forward(self, x):
        return fast_layernorm(self, x)
