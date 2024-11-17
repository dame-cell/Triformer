import triton 
import triton.language as tl
import torch 

@triton.jit
def swiglu_forward_optimized(
    e_ptr, g_ptr, output_ptr,
    sigmoid_ptr, f_ptr, 
    e_stride, g_stride, output_stride,
    sigmoid_stride, f_stride, 
    BLOCK_SIZE: tl.constexpr,
    n_cols,
):
    row_idx = tl.program_id(axis=0)
    col_offset = tl.arange(0, BLOCK_SIZE)  
    mask = col_offset < n_cols

    e_ptr += row_idx * e_stride
    g_ptr += row_idx * g_stride
    output_ptr += row_idx * output_stride
    sigmoid_ptr += row_idx * sigmoid_stride
    f_ptr += row_idx * f_stride

    e_row = tl.load(e_ptr + col_offset, mask=mask).to(tl.float32)
    g_row = tl.load(g_ptr + col_offset, mask=mask).to(tl.float32)

    sigmoid_e_row = tl.sigmoid(e_row)
    f_row = e_row * sigmoid_e_row
    
    tl.store(sigmoid_ptr + col_offset, sigmoid_e_row, mask=mask)
    tl.store(f_ptr + col_offset, f_row, mask=mask)
    
    output_row = f_row * g_row
    tl.store(output_ptr + col_offset, output_row, mask=mask)


@triton.jit 
def swiglu_backward(
    grad_output_ptr, grad_e_ptr, grad_g_ptr, 
    e_ptr, g_ptr, n_cols, sigmoid_ptr, f_ptr,
    grad_output_stride, grad_e_stride, grad_g_stride,
    e_stride, g_stride, sigmoid_stride, f_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    col_offset = tl.arange(0, BLOCK_SIZE)
    mask = col_offset < n_cols

    grad_output_row = tl.load(grad_output_ptr + pid * grad_output_stride + col_offset, mask=mask)
    e_row = tl.load(e_ptr + pid * e_stride + col_offset, mask=mask)
    g_row = tl.load(g_ptr + pid * g_stride + col_offset, mask=mask)
    sigmoid_row = tl.load(sigmoid_ptr + pid * sigmoid_stride + col_offset, mask=mask)
    f_row = tl.load(f_ptr + pid * f_stride + col_offset, mask=mask)

    grad_g_row = grad_output_row * f_row
    
    grad_e_row = grad_output_row * g_row * sigmoid_row * (1.0 + e_row * (1.0 - sigmoid_row))

    tl.store(grad_e_ptr + pid * grad_e_stride + col_offset, grad_e_row, mask=mask)
    tl.store(grad_g_ptr + pid * grad_g_stride + col_offset, grad_g_row, mask=mask)

class FastSwigluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, e, g):
        n_cols = g.shape[-1]
        sigmoid_ptr = torch.empty_like(e)
        f_ptr = torch.empty_like(e)
        output = torch.empty_like(e)

        # Calculate BLOCK_SIZE
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        BLOCK_SIZE = min(max(BLOCK_SIZE, 32), 1024)

        grid = (e.shape[0] * e.shape[1],)
        swiglu_forward_optimized[grid](
            e, g, output,
            sigmoid_ptr, f_ptr,
            e.stride(1), g.stride(1), output.stride(1),
            sigmoid_ptr.stride(1), f_ptr.stride(1),
            BLOCK_SIZE=BLOCK_SIZE,
            n_cols=n_cols,
        )

        ctx.save_for_backward(e, g, sigmoid_ptr, f_ptr)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.n_cols = n_cols
        return output

    @staticmethod
    def backward(ctx, grad_output):
        e, g, sigmoid_ptr, f_ptr = ctx.saved_tensors
        BLOCK_SIZE = ctx.BLOCK_SIZE
        n_cols = ctx.n_cols

        grad_e = torch.empty_like(e)
        grad_g = torch.empty_like(g)

        grid = (e.shape[0] * e.shape[1],)
        swiglu_backward[grid](
            grad_output.contiguous(), grad_e, grad_g,
            e, g, n_cols, sigmoid_ptr, f_ptr,
            grad_output.stride(1), grad_e.stride(1), grad_g.stride(1),
            e.stride(1), g.stride(1), sigmoid_ptr.stride(1), f_ptr.stride(1),
            BLOCK_SIZE=BLOCK_SIZE
        )
        return grad_e, grad_g

class TritonSwiglu(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, e, g):
        return FastSwigluFunction.apply(e, g)
