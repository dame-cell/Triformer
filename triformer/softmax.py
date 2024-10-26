import triton
import triton.language as tl
import torch 
from torch.autograd import autograd

@triton.jit
def fwd_softmax_kernel():
  pass 

@triton.jit
def bwd_softmax_kernel():
  pass 


class TritonsoftmaxFunction(autograd.Function):
  @staticmethod
  def forward(ctx, x, weight, bias):
    pass 

  @staticmethod
  def backward(ctx, grad_output):
    pass 


class Tritonsoftmax(nn.Module):
  pass 
