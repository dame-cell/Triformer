import triton
import triton.language as tl
import torch

@triton.jit
def cross_entropy_fwd_bwd_kernel(
    output_loss_ptr,
    output_logit_grad_ptr,
    input_logit_ptr,
    input_targ_ptr,
    input_divisor_ptr,
    output_loss_stride,
    output_logit_grad_stride,
    input_logit_stride,
    input_targ_stride,
    n_cols,
    ignore_index,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    logit_grad_row_start_ptr = output_logit_grad_ptr + row_idx * output_logit_grad_stride
    logit_row_start_ptr = input_logit_ptr + row_idx * input_logit_stride
    targ_ptr = input_targ_ptr + row_idx * input_targ_stride
    loss_ptr = output_loss_ptr + row_idx * output_loss_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    logit_row_ptrs = logit_row_start_ptr + col_offsets
    logit_grad_row_ptrs = logit_grad_row_start_ptr + col_offsets

    logit_row = tl.load(logit_row_ptrs, mask=col_offsets < n_cols, other=float("-Inf"))
    targ = tl.load(targ_ptr)
    divisor = tl.load(input_divisor_ptr)

    logit_row = logit_row - tl.max(logit_row, axis=0)
    exp_logit_row = tl.exp(logit_row)
    sum_exp_logit_row = tl.sum(exp_logit_row, axis=0)

    log_sum_exp_logit_row = tl.log(sum_exp_logit_row)
    logit_gt_logit = tl.sum(tl.where(targ == col_offsets, logit_row, 0.0))
    loss = log_sum_exp_logit_row - logit_gt_logit
    loss = loss / divisor
    loss = tl.where(targ == ignore_index, 0.0, loss)
    tl.store(loss_ptr, loss)

    targ_one_hot = tl.where(targ == col_offsets, 1.0, 0.0)
    grad = (exp_logit_row / sum_exp_logit_row - targ_one_hot)
    grad = grad / divisor
    grad = tl.where(targ == ignore_index, 0.0, grad)
    tl.store(logit_grad_row_ptrs, grad, mask=col_offsets < n_cols)


class CrossEntropyLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor, targ: torch.Tensor, ignore_index: int, reduction: str, n_chunks: int = 1):
        n_tokens = logits.shape[0]
        n_classes = logits.shape[1]

        assert logits.ndim == 2, logits.ndim
        assert targ.ndim == 1, targ.shape
        assert logits.shape[0] == targ.shape[0], f"Number of tokens in logits and targ is not equal: {(logits.shape, targ.shape) = }"
        assert reduction in ("mean", "sum"), reduction
        assert n_chunks > 0 and n_tokens % n_chunks == 0, f"n_tokens must be divisible by n_chunks: {n_tokens}, {n_chunks}"

        NUM_WARPS = 16
        BLOCK_SIZE = triton.next_power_of_2(n_classes)
        chunk_size = n_tokens // n_chunks

        loss = torch.empty(n_tokens, dtype=logits.dtype, device=logits.device)
        grad_logits = logits  # In-place gradient computation

        divisor = (targ != ignore_index).sum() if reduction == "mean" else torch.ones(1, dtype=logits.dtype, device=logits.device)
        divisor = torch.maximum(divisor, torch.ones_like(divisor))

        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size

            logits_chunk = grad_logits[start_idx:end_idx]
            targ_chunk = targ[start_idx:end_idx]
            loss_chunk = loss[start_idx:end_idx]

            cross_entropy_fwd_bwd_kernel[(chunk_size,)](
                loss_chunk,
                logits_chunk,  # Reuse logits tensor for gradients
                logits_chunk,
                targ_chunk,
                divisor,
                loss_chunk.stride(0),
                logits_chunk.stride(0),
                logits_chunk.stride(0),
                targ_chunk.stride(0),
                n_classes,
                ignore_index,
                num_warps=NUM_WARPS,
                BLOCK_SIZE=BLOCK_SIZE,
            )

            torch.cuda.empty_cache()

        loss = loss.sum()
        
        if logits.requires_grad:
            ctx.save_for_backward(grad_logits)
        
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        grad_logits, = ctx.saved_tensors
        grad_logits *= grad_output
        return grad_logits, None, None, None, None


class TritonCrossEntropyLoss(torch.nn.Module):
    def __init__(self, pad_token_id: int, reduction: str, n_chunks: int = 1):
        super().__init__()
        self.ignore_index = pad_token_id
        self.reduction = reduction
        self.n_chunks = n_chunks
        
    def forward(self, logits, targets):
        logits = logits.view(-1, logits.size(-1))  
        targets = targets.view(-1)  
        return CrossEntropyLossFunction.apply(
            logits, 
            targets, 
            self.ignore_index,
            self.reduction,
            self.n_chunks
        )
      
      
 