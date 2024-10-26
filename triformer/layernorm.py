import triton
import triton.language as tl
import torch 
from torch.autograd import autograd

@triton.jit
def fwd_layernorm_kernel():
  pass 

@triton.jit
def bwd_layernorm_kernel():
  pass 


class TritonLayerNormFunction(autograd.Function):
  @staticmethod
  def forward(ctx, x, weight, bias):
    pass 

  @staticmethod
  def backward(ctx, grad_output):
    pass 


class TritonLayerNorm(nn.Module):
  pass 
