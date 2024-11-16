import torch
import triton
import triton.language as tl
from .utils import calculate_settings

@triton.jit
def rmsnorm_forward(
    Y, Y_row_stride,
    X, X_row_stride,
    W,
    r,  # stores rms value
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Compute memory locations
    Y_ptr = Y + row_idx * Y_row_stride
    X_ptr = X + row_idx * X_row_stride
    r_ptr = r + row_idx

    # Load data
    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)

    # Calculate RMS (root mean square)
    X_squared = X_row * X_row
    mean_X_squared = tl.sum(X_squared, axis=0) / n_cols
    rms = tl.math.rsqrt(mean_X_squared + eps)
    
    # Store RMS for backward pass
    tl.store(r_ptr, rms)
    
    # Normalize and scale
    output = X_row * rms * W_row
    tl.store(Y_ptr + col_offsets, output, mask=mask)


@triton.jit
def _rms_layernorm_backward(
    dY, dY_row_stride,
    X,  X_row_stride,
    W,  W_row_stride,
    r,  r_row_stride,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load inputs
    dY_row = tl.load(dY + row_idx * dY_row_stride + col_offsets, mask=mask, other=0).to(tl.float32)
    X_row = tl.load(X + row_idx * X_row_stride + col_offsets, mask=mask, other=0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)
    rms = tl.load(r + row_idx).to(tl.float32)

    # Normalized input
    X_norm = X_row * rms

    # Compute gradient
    dX_norm = dY_row * W_row
    
    # Compute sum terms
    sum_dY_X = tl.sum(dX_norm * X_norm, axis=0)
    
    # Final gradient computation
    dX = rms * (dX_norm - X_norm * sum_dY_X / n_cols)
    
    # Store result
    tl.store(dY + row_idx * dY_row_stride + col_offsets, dX, mask=mask)

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

        dX = torch.empty_like(dY)
        
        _rms_layernorm_backward[(n_rows,)](
            dX, dX.stride(0),
            X, X.stride(0),
            W, W.stride(0),
            r, r.stride(0),
            n_cols, ctx.eps,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
            num_warps=ctx.num_warps,
        )

        # Weight gradients
        X_norm = X * r.view(-1, 1)
        dW = (dY * X_norm).sum(0)

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
