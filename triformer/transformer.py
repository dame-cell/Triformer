@triton.jit
def _seeded_dropout_strided(
    x_ptr,
    output_ptr,
    seeds_ptr,  # Vector of seeds, one per row
    stride_x_row,  # Stride between rows
    stride_x_col,  # Stride between columns
    stride_out_row,
    stride_out_col,
    n_rows,
    n_cols,
    p,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Program ID for parallelism
    pid = tl.program_id(axis=0)
    row_idx = pid // tl.cdiv(n_cols, BLOCK_SIZE_N)
    col_idx = pid % tl.cdiv(n_cols, BLOCK_SIZE_N)

    # Compute offsets for this block
    row_start = row_idx * BLOCK_SIZE_M
    col_start = col_idx * BLOCK_SIZE_N
    
    # Create ranges for the block
    rows = row_start + tl.arange(0, BLOCK_SIZE_M)
    cols = col_start + tl.arange(0, BLOCK_SIZE_N)
    
    # Compute mask for valid elements
    row_mask = rows < n_rows
    col_mask = cols < n_cols
    mask = row_mask[:, None] & col_mask[None, :]

    # Load the seed for this row
    seed = tl.load(seeds_ptr + rows, mask=row_mask)
    
    # Compute memory locations
    x_ptrs = x_ptr + rows[:, None] * stride_x_row + cols[None, :] * stride_x_col
    out_ptrs = output_ptr + rows[:, None] * stride_out_row + cols[None, :] * stride_out_col
    
    # Load input data
    x = tl.load(x_ptrs, mask=mask)
    
    # Generate random numbers - note we use both row seed and col position
    offsets = cols[None, :] + rows[:, None] * n_cols  # Unique offset per element
    random = tl.rand(seed[:, None], offsets)
    x_keep = random > p
    
    # Apply dropout
    output = tl.where(x_keep, x / (1 - p), 0.0)
    
    # Store results
    tl.store(out_ptrs, output, mask=mask)

def strided_dropout(x, p, seeds=None):
    if not x.is_cuda:
        raise ValueError("Input tensor must be on CUDA device")
    
    # Generate seeds if not provided (one per row)
    if seeds is None:
        seeds = torch.randint(0, 2**31-1, (x.size(0),), device=x.device, dtype=torch.int32)
    
    output = torch.empty_like(x)
    
    # Handle non-contiguous tensors
    if not x.is_contiguous():
        x = x.contiguous()
    
    n_rows, n_cols = x.size()
    
    # Grid and block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 64
    grid = (triton.cdiv(n_rows, BLOCK_SIZE_M) * triton.cdiv(n_cols, BLOCK_SIZE_N),)
    
    _seeded_dropout_strided[grid](
        x_ptr=x,  
        output_ptr=output,
        seeds_ptr=seeds,
        stride_x_row=x.stride(0),
        stride_x_col=x.stride(1),
        stride_out_row=output.stride(0),
        stride_out_col=output.stride(1),
        n_rows=n_rows,
        n_cols=n_cols,
        p=p,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return output

class TritonDropout(autograd.Function):
  @classmethod
  def forward(self, ctx, x, p, seeds=None):
    output = strided_dropout(x, p, seeds)
    ctx.save_for_backward(output,p,seeds)
    return output
  
  @classmethod
  def backward(self, ctx, dy):
    output,p,seeds= ctx.saved_tensors
    return strided_dropout(dy, p, seeds), None, None
