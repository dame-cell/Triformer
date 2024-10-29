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
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Get row index
    row_idx = tl.program_id(0)
    
    # Setup pointers and offsets
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    
    # Load with masking
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    
    # Compute softmax with numerical stability
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    
    # Store result
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)


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
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load softmax output and gradient
    row_start_ptr = input_ptr + row_idx * input_row_stride
    grad_start_ptr = grad_ptr + row_idx * grad_row_stride
    
    probs = tl.load(row_start_ptr + col_offsets, mask=mask, other=0.0)
    grad = tl.load(grad_start_ptr + col_offsets, mask=mask, other=0.0)

    # Compute gradient (fixed formula)
    sum_grad_times_output = tl.sum(grad * probs, axis=0)
    grad_output = probs * (grad - sum_grad_times_output)

    # Store result
    output_start_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_start_ptr + col_offsets, grad_output, mask=mask)
    
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
            BLOCK_SIZE=min(triton.next_power_of_2(cols), 4096),
        )
        ctx.save_for_backward(output) 

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
            BLOCK_SIZE=min(triton.next_power_of_2(cols), 4096),
            num_warps=2,
        )
        return grad_input

triton_softmax = TritonsoftmaxFunction.apply
# TO DO : implement causal softmax
class TritonSoftmax(torch.nn.Module):
    def __init__(self, dim=-1, causal=False):
        super().__init__()
        self.dim = dim
        self.causal = causal
    
    def forward(self, x):
        return triton_softmax(x, self.causal)