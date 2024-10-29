# huge thanks to ludicrains for the reference https://github.com/lucidrains/triton-transformer/blob/main/triton_transformer/softmax.py
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from .utils import calc_num_warps

@triton.jit
def softmax_kernel_forward(
    out_ptr,
    inp_ptr,
    inp_stride,
    out_stride, 
    seq_len,
    is_causal,
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr
):
    batch_idx = tl.program_id(0)
    batch_start_ptr = inp_ptr + batch_idx * inp_stride
    pos_offsets = tl.arange(0, BLOCK_SIZE)
    batch_ptrs = batch_start_ptr + pos_offsets
    valid_mask = pos_offsets < seq_len

    logits = tl.load(batch_ptrs, mask=valid_mask, other=-float('inf'))

    if is_causal:
        attn_mask = pos_offsets > (batch_idx % seq_len)
        logits = logits + tl.where(attn_mask, -float('inf'), 0.)

    shifted_logits = logits - tl.max(logits, axis=0)
    exp_logits = tl.exp(shifted_logits)
    sum_exp = tl.sum(exp_logits, axis=0)
    probs = exp_logits / sum_exp

    out_batch_ptr = out_ptr + batch_idx * out_stride
    out_ptrs = out_batch_ptr + pos_offsets
    tl.store(out_ptrs, probs, mask=valid_mask)

@triton.jit
def softmax_kernel_backward(
    grad_out_ptr,
    probs_ptr,
    grad_in_ptr,
    grad_stride,
    probs_stride,
    out_stride,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr
):
    batch_idx = tl.program_id(0)
    probs_start_ptr = probs_ptr + batch_idx * probs_stride
    grad_start_ptr = grad_in_ptr + batch_idx * grad_stride

    pos_offsets = tl.arange(0, BLOCK_SIZE)
    probs_ptrs = probs_start_ptr + pos_offsets
    grad_ptrs = grad_start_ptr + pos_offsets

    valid_mask = pos_offsets < seq_len

    probs_vals = tl.load(probs_ptrs, mask=valid_mask, other=0.)
    grad_vals = tl.load(grad_ptrs, mask=valid_mask, other=0.)

    grad_times_probs = probs_vals * grad_vals
    final_grad = grad_times_probs - probs_vals * tl.sum(grad_times_probs, axis=0)

    out_start_ptr = grad_out_ptr + batch_idx * out_stride
    out_ptrs = out_start_ptr + pos_offsets
    tl.store(out_ptrs, final_grad, mask=valid_mask)

class SoftmaxFunction(torch.autograd.Function):
    @classmethod
    def forward(self, ctx, inputs, is_causal):
        orig_shape = inputs.shape
        flattened = inputs.view(-1, orig_shape[-1])
        batch_size, seq_len = flattened.shape

        block_dim = triton.next_power_of_2(seq_len)
        n_warps = calc_num_warps(block_dim)

        outputs = torch.empty_like(flattened)

        softmax_kernel_forward[(batch_size,)](
            outputs,
            flattened,
            flattened.stride(0),
            outputs.stride(0),
            seq_len,
            is_causal,
            block_dim,
            n_warps
        )

        if inputs.requires_grad:
            ctx.save_for_backward(outputs)
        return outputs.view(*orig_shape)

    @classmethod
    def backward(self, ctx, grad_outputs):
        orig_shape = grad_outputs.shape
        probs, = ctx.saved_tensors

        flat_grads = grad_outputs.view(-1, grad_outputs.shape[-1])
        batch_size, seq_len = flat_grads.shape

        block_dim = triton.next_power_of_2(seq_len)
        n_warps = calc_num_warps(block_dim)

        grad_inputs = torch.empty_like(probs)

        softmax_kernel_backward[(batch_size,)](
            grad_inputs,
            probs,
            flat_grads,
            flat_grads.stride(0),
            probs.stride(0),
            grad_inputs.stride(0),
            seq_len,
            block_dim,
            n_warps
        )

        return grad_inputs.view(*orig_shape), None

class TritonSoftmax(torch.nn.Module):
    def __init__(self, is_causal):
        super().__init__()
        self.is_causal = is_causal
        
    def forward(self, inputs):
        return SoftmaxFunction.apply(inputs, self.is_causal)
