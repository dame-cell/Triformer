import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from utils import calc_num_warps
@triton.jit
def softmax_kernel_forward(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    causal:bool,
    BLOCK_SIZE: tl.constexpr,  # Add constexpr here
    num_warps: tl.constexpr    # Add constexpr here
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols

    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

    if causal:
        causal_mask = col_offsets > (row_idx % n_cols)
        row = row + tl.where(causal_mask, -float('inf'), 0.)

    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)

@triton.jit
def softmax_kernel_backward(
    output_ptr,
    input_ptr,
    grad_ptr,
    grad_row_stride,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,  # Add constexpr here
    num_warps: tl.constexpr    # Add constexpr here
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    grad_row_start_ptr = grad_ptr + row_idx * grad_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    grad_ptrs = grad_row_start_ptr + col_offsets

    mask = col_offsets < n_cols

    probs_row = tl.load(input_ptrs, mask=mask, other=0.)
    grad_row = tl.load(grad_ptrs, mask=mask, other=0.)

    dxhat = probs_row * grad_row
    softmax_grad_output = dxhat - probs_row * tl.sum(dxhat, axis=0)

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_grad_output, mask=mask)

class SoftmaxFunction(torch.autograd.Function):
    @classmethod
    def forward(ctx, x, causal):
        shape = x.shape
        x = x.view(-1, shape[-1])
        n_rows, n_cols = x.shape

        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        num_warps = calc_num_warps(BLOCK_SIZE)

        y = torch.empty_like(x)

        softmax_kernel_forward[(n_rows,)](
            y,
            x,
            x.stride(0),
            y.stride(0),
            n_cols,
            causal,
            BLOCK_SIZE,
            num_warps
        )

        if x.requires_grad:
            ctx.save_for_backward(y)
        return y.view(*shape)

    @classmethod
    def backward(ctx, grad_probs):
        shape = grad_probs.shape
        probs, = ctx.saved_tensors

        grad_probs = grad_probs.view(-1, grad_probs.shape[-1])
        n_rows, n_cols = grad_probs.shape

        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        num_warps = calc_num_warps(BLOCK_SIZE)

        dx = torch.empty_like(probs)

        softmax_kernel_backward[(n_rows,)](
            dx,
            probs,
            grad_probs,
            grad_probs.stride(0),
            probs.stride(0),
            dx.stride(0),
            n_cols,
            BLOCK_SIZE,
            num_warps
        )

        return dx.view(*shape), None



class TritonSoftmax(torch.nn.Module):
    def __init__(self, x, causal):
        super().__init__()
        self.causal = causal
  
        
    def forward(self, x):
        return SoftmaxFunction.apply(x=x, causal=self.causal)