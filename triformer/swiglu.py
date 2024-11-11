import triton 
import triton.language as tl
import torch 
from .utils import calculate_settings


@triton.jit
def swiglu_forward_optimized(
    e_ptr, g_ptr, output_ptr,
    e_stride, g_stride, output_stride,
    BLOCK_SIZE: tl.constexpr,
    n_cols,
    num_warps: tl.constexpr
):
    row_idx = tl.program_id(axis=0)
    col_offset = tl.arange(0, BLOCK_SIZE)
    mask = col_offset < n_cols

    e_ptr += row_idx * e_stride
    g_ptr += row_idx * g_stride
    output_ptr += row_idx * output_stride

    e_row = tl.load(e_ptr + col_offset, mask=mask).to(tl.float32)
    g_row = tl.load(g_ptr + col_offset, mask=mask).to(tl.float32)
    
    sigmoid_e_row = tl.sigmoid(e_row)
    f_row = e_row * sigmoid_e_row
    output_row = f_row * g_row

    tl.store(output_ptr + col_offset, output_row, mask=mask)

@triton.jit
def swiglu_backward_optimized(
    dy_ptr, e_ptr, g_ptr,
    grad_e_ptr, grad_g_ptr,
    dy_stride, e_stride, g_stride,
    grad_e_stride, grad_g_stride,
    BLOCK_SIZE: tl.constexpr,
    n_cols,
    num_warps: tl.constexpr
):
    row_idx = tl.program_id(axis=0)
    col_offset = tl.arange(0, BLOCK_SIZE)
    mask = col_offset < n_cols

    # Calculate pointers
    dy_ptr += row_idx * dy_stride
    e_ptr += row_idx * e_stride
    g_ptr += row_idx * g_stride
    grad_e_ptr += row_idx * grad_e_stride
    grad_g_ptr += row_idx * grad_g_stride

    # Load inputs
    dy = tl.load(dy_ptr + col_offset, mask=mask).to(tl.float32)
    e = tl.load(e_ptr + col_offset, mask=mask).to(tl.float32)
    g = tl.load(g_ptr + col_offset, mask=mask).to(tl.float32)

    # Compute intermediates on the fly
    sigmoid_e = tl.sigmoid(e)
    f = e * sigmoid_e

    # Compute gradients
    grad_g = dy * f
    grad_e = dy * g * sigmoid_e * (1.0 + e * (1.0 - sigmoid_e))

    # Store results
    tl.store(grad_e_ptr + col_offset, grad_e, mask=mask)
    tl.store(grad_g_ptr + col_offset, grad_g, mask=mask)

class FastSwigluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, e, g):
        e = e.contiguous()
        g = g.contiguous()
        assert e.is_contiguous() and g.is_contiguous(), "Input tensors must be contiguous"
        assert e.shape == g.shape, "Input tensors must have the same shape"
        
        shape = e.shape
        dim = shape[-1]
        e_reshaped = e.view(-1, dim)
        g_reshaped = g.view(-1, dim)
        n_rows, n_cols = e_reshaped.shape

        BLOCK_SIZE, num_warps = calculate_settings(n_cols)
        
        output = torch.empty_like(e)
        output_reshaped = output.view(-1, dim)

        grid = (n_rows,)
        swiglu_forward_optimized[grid](
            e_reshaped, g_reshaped, output_reshaped,
            e_reshaped.stride(0), g_reshaped.stride(0), output_reshaped.stride(0),
            BLOCK_SIZE,
            n_cols,
            num_warps
        )

        ctx.save_for_backward(e, g)
        ctx.shape = shape
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps

        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        e, g = ctx.saved_tensors
        shape = ctx.shape
        BLOCK_SIZE = ctx.BLOCK_SIZE
        num_warps = ctx.num_warps

        dim = shape[-1]
        grad_output_reshaped = grad_output.view(-1, dim)
        e_reshaped = e.view(-1, dim)
        g_reshaped = g.view(-1, dim)

        grad_e = torch.empty_like(e)
        grad_g = torch.empty_like(g)
        grad_e_reshaped = grad_e.view(-1, dim)
        grad_g_reshaped = grad_g.view(-1, dim)

        n_rows, n_cols = e_reshaped.shape

        grid = (n_rows,)
        swiglu_backward_optimized[grid](
            grad_output_reshaped, e_reshaped, g_reshaped,
            grad_e_reshaped, grad_g_reshaped,
            grad_output_reshaped.stride(0), 
            e_reshaped.stride(0), 
            g_reshaped.stride(0),
            grad_e_reshaped.stride(0), 
            grad_g_reshaped.stride(0),
            BLOCK_SIZE,
            n_cols,
            num_warps
        )

        return grad_e, grad_g

class TritonSwiglu(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, e, g):
        return FastSwigluFunction.apply(e, g)