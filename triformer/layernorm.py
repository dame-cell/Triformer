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
  mean_centered_ptr,
  normed_ptr,
  cols,
  eps,
  stable,
  BLOCK_SIZE: tl.constexpr):
  
  row_idx = tl.program_id(axis=0)
  col_offset = tl.arange(0, BLOCK_SIZE)
  row_mask = col_offset < cols 
  
  # Load inputs
  x = tl.load(input_ptr + row_idx * input_stride_0 + col_offset, mask=row_mask, other=0.).to(tl.float32)
  gamma = tl.load(gamma_ptr + col_offset, mask=row_mask).to(tl.float32)
  beta = tl.load(beta_ptr + col_offset, mask=row_mask).to(tl.float32)
  
  # Numerical stability improvement
  if stable:
      row_max = tl.max(tl.where(row_mask, x, float('-inf')), axis=0)
      x = tl.where(row_mask, x / row_max, 0.)
  
  # Layer norm computation
  mean = tl.sum(x, axis=0) / cols
  x_centered = tl.where(row_mask, x - mean, 0.)
  var = tl.sum(x_centered * x_centered, axis=0) / cols
  inv_std = 1. / tl.sqrt(var + eps)
  normed = x_centered * inv_std
  
  # Output computation
  output = normed * gamma + beta
  
  # Store results
  output_ptr = output_ptr + row_idx * output_stride_0 + col_offset
  tl.store(output_ptr, output, mask=row_mask)
  
  # Store intermediate values for backward
  mean_centered_ptr = mean_centered_ptr + row_idx * input_stride_0 + col_offset
  normed_ptr = normed_ptr + row_idx * input_stride_0 + col_offset
  tl.store(mean_centered_ptr, x_centered, mask=row_mask)
  tl.store(normed_ptr, normed, mask=row_mask)


# a very simple backward layernorm kernel 
@triton.jit 
def bwd_layernorm_kernel(
  grad_output_ptr,
  grad_output_stride_0,
  mean_centered_ptr,
  normed_ptr,
  gamma_ptr,
  grad_input_ptr,
  cols,
  eps,
  BLOCK_SIZE: tl.constexpr):
  
  row_idx = tl.program_id(axis=0)
  col_offset = tl.arange(0, BLOCK_SIZE)
  row_mask = col_offset < cols 
  
  # Load saved values from forward pass
  dy = tl.load(grad_output_ptr + row_idx * grad_output_stride_0 + col_offset, mask=row_mask, other=0.).to(tl.float32)
  x_centered = tl.load(mean_centered_ptr + row_idx * grad_output_stride_0 + col_offset, mask=row_mask, other=0.).to(tl.float32)
  normed = tl.load(normed_ptr + row_idx * grad_output_stride_0 + col_offset, mask=row_mask, other=0.).to(tl.float32)
  gamma = tl.load(gamma_ptr + col_offset, mask=row_mask).to(tl.float32)
  
  # Compute gradients
  dy_gamma = dy * gamma
  var = tl.sum(x_centered * x_centered, axis=0) / cols
  inv_std = 1. / tl.sqrt(var + eps)
  
  grad_input = 1. / cols * inv_std * (
      cols * dy_gamma - 
      tl.sum(dy_gamma, axis=0) - 
      normed * tl.sum(dy_gamma * normed, axis=0)
  )
  
  # Store gradients
  grad_input_ptr = grad_input_ptr + row_idx * grad_output_stride_0 + col_offset
  tl.store(grad_input_ptr, grad_input, mask=row_mask)



class TritonLayerNormFunction(Function):
    @staticmethod
    def forward(ctx, input, gamma, beta, eps=1e-5, stable=True):
        rows, cols = input.shape
        assert input.dim() == 2, "We are working with only 2D tensors for now"
        
        # Allocate output and intermediate buffers
        output = torch.empty_like(input)
        mean_centered = torch.empty_like(input)
        normed = torch.empty_like(input)
        
        block_size = triton.next_power_of_2(cols)
        grid = (rows,)
        
        fwd_layernorm_kernel[grid](
            input, input.stride(0),
            output, output.stride(0),
            gamma, beta,
            mean_centered, normed,
            cols, eps, stable,
            BLOCK_SIZE=block_size
        )
        
        ctx.save_for_backward(mean_centered, normed, gamma)
        ctx.eps = eps
        return output

    @staticmethod
    def backward(ctx, grad_output):
        mean_centered, normed, gamma = ctx.saved_tensors
        rows, cols = grad_output.shape
        
        grad_input = torch.empty_like(grad_output)
        grad_gamma = torch.zeros_like(gamma)
        grad_beta = torch.zeros_like(gamma)
        
        block_size = triton.next_power_of_2(cols)
        grid = (rows,)
        
        bwd_layernorm_kernel[grid](
            grad_output, grad_output.stride(0),
            mean_centered, normed,
            gamma, grad_input,
            cols, ctx.eps,
            BLOCK_SIZE=block_size
        )
        
        # Compute gradients for gamma and beta
        grad_gamma = (grad_output * normed).sum(0)
        grad_beta = grad_output.sum(0)
        
        return grad_input, grad_gamma, grad_beta, None, None