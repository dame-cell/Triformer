import torch
import triton
import triton.language as tl

@triton.jit
def _seeded_dropout(x_ptr, output_ptr, n_elements, p, seed, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE * 4
    
    for i in range(4):
        offset = block_start + BLOCK_SIZE * i + tl.arange(0, BLOCK_SIZE)
        mask = offset < n_elements
        x = tl.load(x_ptr + offset, mask=mask)
        r = tl.randint(seed, offset)
        keep = r > p
        output = tl.where(keep, x / (1 - p), 0.0)
        tl.store(output_ptr + offset, output, mask=mask)
        
def seeded_dropout(x, p, seed):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    
    # Make sure seed is on GPU
    if not seed.is_cuda:
        seed = seed.cuda()
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE * 4),)
    
    _seeded_dropout[grid](
        x_ptr=x,
        output_ptr=output,
        n_elements=n_elements,
        p=p,
        seed=seed,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output

class TritonDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, seed=None):
        if seed is None:
            seed = torch.randint(0, 2**31-1, (), device=x.device)
        elif not isinstance(seed, torch.Tensor):
            seed = torch.tensor(seed, device=x.device)
            
        output = seeded_dropout(x, p, seed)
        mask = (output != 0).to(x.dtype)
        ctx.save_for_backward(mask)
        ctx.p = p
        return output

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        grad_input = grad_output * mask
        return grad_input, None, None

