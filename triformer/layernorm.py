import triton 
import triton.language as tl
import torch 
from .utils import calculate_settings
@triton.jit
def layernorm_forward(
    Y, Y_row_stride,
    X, X_row_stride,
    weight,
    bias,
    var,
    mean,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y += row_idx * Y_row_stride
    X += row_idx * X_row_stride
    var += row_idx
    mean += row_idx

    X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
    weight_row = tl.load(weight + col_offsets, mask=mask, other=0).to(tl.float32)
    bias_row = tl.load(bias + col_offsets, mask=mask, other=0).to(tl.float32)

    mean_X  = tl.sum(X_row, axis=0) / n_cols
    XX      = X_row - mean_X
    row_var  = tl.sum(XX * XX, axis=0) / n_cols
    inv_var = tl.math.rsqrt(row_var + eps)
    tl.store(var, inv_var)
    tl.store(mean, mean_X)
    output = (XX * inv_var) * weight_row + bias_row
    tl.store(Y + col_offsets, output, mask=mask)

@triton.jit
def layernorm_backward_verbose(
    dY, dY_row_stride,
    X, X_row_stride,
    gamma, beta,
    var, mean,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dY += row_idx * dY_row_stride
    X += row_idx * X_row_stride
    var += row_idx
    mean += row_idx

    # Load data
    dout = tl.load(dY + col_offsets, mask=mask, other=0).to(tl.float32)
    x = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
    gamma_row = tl.load(gamma + col_offsets, mask=mask, other=0).to(tl.float32)
    beta_row = tl.load(beta + col_offsets, mask=mask, other=0).to(tl.float32)

    mean_val = tl.load(mean).to(tl.float32)
    var_val = tl.load(var).to(tl.float32)
    std_dev = tl.math.sqrt(var_val + eps)

    # Forward pass components (for reference)
    x_centered = x - mean_val
    x_normalized = x_centered / std_dev

    # Gradients for gamma and beta
    dgamma = tl.sum(dout * x_normalized, axis=0)
    dbeta = tl.sum(dout, axis=0)

    # Gradient w.r.t. x_normalized
    dx_normalized = dout * gamma_row

    # Gradient w.r.t. std_dev
    dstd = tl.sum(dx_normalized * (-x_centered / (std_dev * std_dev)), axis=0)

    # Gradient w.r.t. variance
    dvar = dstd * (0.5 / std_dev)

    # Gradient w.r.t. mean
    dx_centered = dx_normalized / std_dev
    dmean_through_var = dvar * -2 * (tl.sum(x_centered, axis=0) / n_cols)  
    dmean = tl.sum(-dx_centered, axis=0) + dmean_through_var

    # Gradient w.r.t. x
    dx = dx_centered
    dx += dmean / n_cols
    dx += 2 * dvar * x_centered / n_cols

    # Store the computed gradient for inputs back to dY
    tl.store(dY + col_offsets, dx, mask=mask)



class Fast_Layernorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weight, bias, eps):
        shape = X.shape
        dim = shape[-1]
        X = X.view(-1, dim)
        n_rows, n_cols = X.shape
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)

        Y  = torch.empty((n_rows, n_cols), dtype=X.dtype, device="cuda:0")
        var  = torch.empty(n_rows, dtype=torch.float32, device="cuda:0")
        mean = torch.empty(n_rows, dtype=torch.float32, device="cuda:0")

        layernorm_forward[(n_rows,)](
            Y, Y.stride(0),
            X, X.stride(0),
            weight,
            bias,
            var,
            mean,
            n_cols, eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.save_for_backward(X, weight, bias, var, mean)
        return Y.view(*shape)

    @staticmethod
    def backward(ctx, dY):
        shape = dY.shape
        dim = shape[-1]
        dY = dY.view(-1, dim)
        X, weight, bias, var, mean = ctx.saved_tensors
        n_rows, n_cols = dY.shape

        layernorm_backward_verbose[(n_rows,)](
            dY, dY.stride(0),
            X,  X.stride(0),
            weight,
            bias,
            var,
            mean,
            n_cols, ctx.eps,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
            num_warps=ctx.num_warps,
        )
        dX = dY.view(*shape)
        return dX, None, None, None, None




class TritonLayerNorm(torch.nn.LayerNorm):
    def forward(self, x):
        return Fast_Layernorm.apply(
            x,
            self.weight,
            self.bias,
            self.eps
        )
    
