import triton 
import triton.language as tl 
import torch 
from .utils import calculate_settings

@triton.jit
def rms_norm_forward(
    Y, Y_row_stride,        
    X, X_row_stride,        
    scaler, scaler_row_stride,       
    rms, rms_row_stride,        
    n_cols,                 
    eps,                    
    BLOCK_SIZE: tl.constexpr,
    num_warps:tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    Y += row_idx * Y_row_stride
    X += row_idx * X_row_stride
    rms += row_idx * rms_row_stride

    X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
    scaler_row = tl.load(scaler + col_offsets, mask=mask, other=0)

    # Compute RMS norm using rsqrt for better performance
    X_scaled = X_row * X_row 
    X_mean_squared = tl.sum(X_scaled, axis=0) / n_cols
    X_mean_squred_eps = X_mean_squared + eps
    inv_rms = tl.math.sqrt(X_mean_squred_eps)
    
    normed = X_row / inv_rms 
    tl.store(rms, inv_rms)
    normed = normed.to(scaler_row.dtype)  
    output = normed * scaler_row
    tl.store(Y + col_offsets, output, mask=mask)



@triton.jit
def rms_backward(
    dy, dy_stride,            
    x, x_stride,              
    scale, scale_stride,      
    rms, rms_stride,          
    dx, dx_stride,            
    n_cols,                   
    eps,                      
    BLOCK_SIZE: tl.constexpr, 
    num_warps: tl.constexpr   
):
    # Get row index and column offsets
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Calculate pointer offsets for current row
    dy_row_ptr = dy + row_idx * dy_stride
    x_row_ptr = x + row_idx * x_stride
    scale_row_ptr = scale + col_offsets
    dx_row_ptr = dx + row_idx * dx_stride
    rms_val = tl.load(rms + row_idx * rms_stride)

    # Load values for current row
    dy_row = tl.load(dy_row_ptr + col_offsets, mask=mask, other=0.).to(tl.float32)
    x_row = tl.load(x_row_ptr + col_offsets, mask=mask, other=0.).to(tl.float32)
    scale_row = tl.load(scale_row_ptr, mask=mask, other=0.).to(tl.float32)

    dx_normalized = dy_row * scale_row
    dx_from_norm = dx_normalized / rms_val
    dl_mean_sq_term = dx_normalized * (-x_row / (rms_val * rms_val))
    dl_mean_squared = tl.sum(dl_mean_sq_term, axis=0) / (2 * rms_val)
    
    dx_row = dx_from_norm + (dl_mean_squared * (2 * x_row / n_cols))
    tl.store(dx_row_ptr + col_offsets, dx_row, mask=mask)
  
    
    
class FastRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X: torch.Tensor, W: torch.Tensor, eps: float):
        shape = X.shape
        dim = shape[-1]
        X = X.view(-1, dim)
        n_rows, n_cols = X.shape
        
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)

        Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device="cuda:0")
        r = torch.empty(n_rows, dtype=torch.float32, device="cuda:0")

        rms_norm_forward[(n_rows,)](
            Y, Y.stride(0),
            X, X.stride(0),
            W, W.stride(0),
            r, r.stride(0),
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
    def backward(ctx, dY: torch.Tensor):
        shape = dY.shape
        dim = shape[-1]
        dY = dY.view(-1, dim)
        X, W, r = ctx.saved_tensors
        n_rows, n_cols = dY.shape

        BLOCK_SIZE, num_warps = ctx.BLOCK_SIZE, ctx.num_warps

        dX = torch.empty_like(X, device="cuda:0")

        rms_backward[(n_rows,)](
            dY, dY.stride(0),
            X, X.stride(0),
            W, W.stride(0),
            r, r.stride(0),
            dX, dX.stride(0),
            n_cols,
            ctx.eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps
        )

        return dX.view(*shape), None, None

class TritonRMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = torch.nn.Parameter(torch.ones(dim))
        self.dim = dim

    def forward(self, x):
        return FastRMSNorm.apply(x, self.scale, self.eps)