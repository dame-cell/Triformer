import triton
import triton.language as tl 
import torch 
from .utils import calculate_settings 

@triton.jit
def rms_norm_forward(
    Y, Y_row_stride,        
    X, X_row_stride,        
    W, W_row_stride,       
    R, R_row_stride,        
    n_cols,                 
    eps,                    
    BLOCK_SIZE: tl.constexpr,
    num_wraps:tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    Y += row_idx * Y_row_stride
    X += row_idx * X_row_stride
    R += row_idx * R_row_stride

    X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask=mask, other=0)

    # Compute RMS norm using rsqrt for better performance
    
    row_var = tl.sum(X_row * X_row, axis=0) / n_cols
    inv_rms = tl.math.rsqrt(row_var + eps)
    tl.store(R, inv_rms)
    normed = X_row * inv_rms
    normed = normed.to(W_row.dtype)  
    output = normed * W_row
    tl.store(Y + col_offsets, output, mask=mask)




class FastRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weight,eps):
      shape = X.shape 
      dim = shape[-1]
      X = X.view(-1,dim)
      n_rows ,n_cols = X.shape
      BLOCK_SIZE , num_wraps = calculate_settings(n_cols)
      
      Y  = torch.empty((n_rows, n_cols), dtype=X.dtype, device="cuda:0")
      scaler  = torch.empty(n_rows, dtype=torch.float32, device="cuda:0")
    
      rms_norm_forward[(n_rows,)](
        Y,Y.stride(0),
        X,X.stride(0),
        weight,weight.stride(0),
        scaler,scaler.stride(0),
        n_cols,eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_wraps=num_wraps
      )

      ctx.eps = eps
      ctx.BLOCK_SIZE = BLOCK_SIZE
      ctx.num_warps = num_wraps
      ctx.save_for_backward(X, weight, scaler)
      return Y.view(*shape)
    @staticmethod
    def backward(ctx, dY):
       pass 