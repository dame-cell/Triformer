import torch 
import triton 
import triton.language as tl 


# this is specifically for T4 gpu 
@triton.autotune(
    configs=[
        
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['n_cols'],
    warmup=100,
    rep=100, 
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

class TritonsoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        rows, cols = x.shape
        assert x.dim() == 2, "only supports 2D tensors"
        output = torch.empty_like(x)
        grid = (rows,)
        fwd_softmax_kernel[grid](
            x,
            output,
            x.stride(0),
            cols,
        )
        return output
      
      
