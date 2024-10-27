import triton
import triton.language as tl
import torch 
import torch.nn as nn 
from torch.autograd import Function
import time 
import matplotlib.pyplot as plt



# a very simple forward layernorm kernel 
@triton.jit
def fwd_layernorm_kernel(
  input_ptr,
  input_stride_0,
  output_ptr,
  output_stride_0,
  gamma_ptr,
  beta_ptr,
  cols,
  eps,
  BLOCK_SIZE: tl.constexpr):
  
  row_idx = tl.program_id(axis=0)
  row_start_ptr = input_ptr + (row_idx * input_stride_0)
  cols_offseet = tl.arange(0, BLOCK_SIZE)
  input_ptrs = row_start_ptr + cols_offseet
  row_mask = cols_offseet < cols 
  
  # load the data from HBM to SRAM 
  rows = tl.load(input_ptrs, mask=row_mask, other=0)
  gamma = tl.load(gamma_ptr + cols_offseet, mask=row_mask)
  beta = tl.load(beta_ptr + cols_offseet, mask=row_mask)
  
  # start layer norm computation
  mean = tl.sum(rows) / cols
  var = tl.sum((rows - mean) * (rows - mean)) / cols
  std = tl.sqrt(var + eps)
  y = (rows - mean) / std
  out = gamma * y + beta
  
  # write back to HBM 
  output_row_ptrs = output_ptr + (row_idx * output_stride_0)
  output_ptrs = output_row_ptrs + cols_offseet
  tl.store(output_ptrs, out, mask=row_mask)


# a very simple backward layernorm kernel 
@triton.jit 
def bwd_layernorm_kernel(
  input_ptr,
  input_stride_0,
  grad_output_ptr,
  grad_output_stride_0,
  gamma_ptr, 
  beta_ptr,
  cols,
  eps,
  BLOCK_SIZE: tl.constexpr):
  
  row_idx = tl.program_id(axis=0)
  row_start_ptr = input_ptr + (row_idx * input_stride_0)
  col_offset = tl.arange(0, BLOCK_SIZE)
  input_ptrs = row_start_ptr + col_offset
  row_mask = col_offset < cols 
  
  # Load data for backward pass
  x = tl.load(input_ptrs, mask=row_mask, other=0)
  grad_output = tl.load(grad_output_ptr + (row_idx * grad_output_stride_0) + col_offset, mask=row_mask, other=0)
  gamma = tl.load(gamma_ptr + col_offset, mask=row_mask)
  
  # Calculate mean and variance (more numerically stable)
  mean = tl.sum(x) / cols
  centered_x = x - mean
  var = tl.sum(centered_x * centered_x) / cols
  std = tl.sqrt(var + eps)
  inv_std = 1.0 / std
  normalized_x = centered_x * inv_std
  
  # Compute gradients with better numerical stability
  grad_output_scaled = grad_output * gamma
  sum_grad_output = tl.sum(grad_output_scaled)
  sum_grad_output_x = tl.sum(grad_output_scaled * normalized_x)
  
  # Gradient for input with improved stability
  grad_input = gamma * inv_std * (
      grad_output - (normalized_x * sum_grad_output_x + sum_grad_output) / cols
  )
  
  # Gradients for gamma and beta
  grad_gamma = grad_output * normalized_x
  grad_beta = grad_output
  
  # Store results
  grad_input_ptrs = grad_output_ptr + (row_idx * grad_output_stride_0) + col_offset
  tl.store(grad_input_ptrs, grad_input, mask=row_mask)
  
  # Accumulate gradients for gamma and beta across rows
  tl.atomic_add(gamma_ptr + col_offset, grad_gamma, mask=row_mask)
  tl.atomic_add(beta_ptr + col_offset, grad_beta, mask=row_mask)



class TritonLayerNormFunction(Function):
    @staticmethod
    def forward(ctx, input, gamma, beta, eps):  
        rows, cols = input.shape 
        assert input.dim() == 2, "We are working with only 2D tensors for now" 
        block_size = triton.next_power_of_2(cols)
        num_warps = 4 
        if block_size == 2047:
            num_warps = 8 
        if block_size == 4095:
            num_warps = 16 

        sm_out = torch.empty_like(input)
        ctx.save_for_backward(input, gamma, beta)
        ctx.eps = eps
        
        grid = (rows,)
        fwd_layernorm_kernel[grid](
            input,
            input.stride(0),
            sm_out,
            sm_out.stride(0),
            gamma,
            beta,
            cols,
            eps,
            BLOCK_SIZE=block_size
        )

        return sm_out  

    @staticmethod
    def backward(ctx, grad_output):
        rows, cols = grad_output.shape  
        assert grad_output.dim() == 2, "We are working with only 2D tensors for now" 
        block_size = triton.next_power_of_2(cols)
        num_warps = 4 
        if block_size == 2047:
            num_warps = 8 
        if block_size == 4095:
            num_warps = 16 

        input, gamma, beta = ctx.saved_tensors
        eps = ctx.eps

        # Create tensors to store gradients - initialize to zero!
        grad_input = torch.empty_like(grad_output)
        grad_gamma = torch.zeros_like(gamma)  # Changed to zeros_like
        grad_beta = torch.zeros_like(beta)    # Changed to zeros_like

        grid = (rows,)
        bwd_layernorm_kernel[grid](
            input,
            input.stride(0),
            grad_output,
            grad_output.stride(0),
            grad_gamma,  # Changed from gamma to grad_gamma
            grad_beta,   # Changed from beta to grad_beta
            cols,
            eps,
            BLOCK_SIZE=block_size
        )

        # Return gradients for all inputs (input, gamma, beta, eps)
        # eps is a scalar, so its gradient is None
        return grad_input, grad_gamma, grad_beta, None



