import torch 
import triton 
import triton.language as tl 

@triton.autotune(
    configs=[
        # Smaller sizes
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        # Medium sizes
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        # Large sizes
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
        # Very large sizes
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=32),
    ],
    key=['n_cols'],
)
@triton.jit
def fwd_softmax_kernel(
    input_ptr, 
    output_ptr,
    stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr,
):
    row_idx = tl.program_id(axis=0)
    row_start_ptr = input_ptr + (row_idx * stride)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    row_mask = col_offsets < n_cols
    
    # Load and find max
    row = tl.load(input_ptrs, mask=row_mask, other=float('-inf'))
    row_max = tl.max(row, axis=0)
    
    # Compute exponentials and sum
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    
    # Normalize and store
    output = numerator / denominator
    output_row_ptr = output_ptr + (row_idx * stride)
    output_ptrs = output_row_ptr + col_offsets
    tl.store(output_ptrs, output, mask=row_mask)

@triton.autotune(
    configs=[
        # Smaller sizes
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        # Medium sizes
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        # Large sizes
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
        # Very large sizes
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=32),
    ],
    key=['n_cols'],
)

@triton.jit
def softmax_kernel_backward(
    output_ptr,
    input_ptr,
    grad_ptr,
    grad_row_stride,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr,
):

    row_idx = tl.program_id(0)

    row_start_ptr = input_ptr + row_idx * input_row_stride
    grad_row_start_ptr = grad_ptr + row_idx * grad_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    grad_ptrs = grad_row_start_ptr + col_offsets

    mask = col_offsets < n_cols

    probs_row = tl.load(input_ptrs, mask = mask, other = 0.)
    grad_row = tl.load(grad_ptrs, mask = mask, other = 0.)

    dxhat = probs_row * grad_row
    softmax_grad_output = dxhat - probs_row * tl.sum(dxhat, axis = 0)

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_grad_output, mask = mask)
    
class TritonsoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        rows, cols = x.shape
        output = torch.empty_like(x)
        grid = (rows,)
        fwd_softmax_kernel[grid](
            x,
            output,
            x.stride(0),
            cols,
        )
        ctx.save_for_backward(output)  # Save output for backward pass
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        rows, cols = grad_output.shape
        grad_input = torch.empty_like(grad_output)
        grid = (rows,)
        
        softmax_kernel_backward[grid](
            grad_input,  # output gradient
            output,      # saved forward output
            grad_output, # incoming gradient
            grad_output.stride(0),
            output.stride(0),
            grad_input.stride(0),
            cols,
        )
        return grad_input

triton_softmax = TritonsoftmaxFunction.apply

def TritonSoftmax(x):
    
    return triton_softmax(x)
