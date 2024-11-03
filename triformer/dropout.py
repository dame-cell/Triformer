import torch
import triton
import triton.language as tl


@triton.jit
def _seeded_dropout(x_ptr, output_ptr, n_elements, p, seed, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE * 4
    
    offset = block_start + tl.arange(0, BLOCK_SIZE)
    r0, r1, r2, r3 = tl.random.rand4x(seed, offset)
    
    scale = 1.0 / (1.0 - p)  # Calculate scale factor once
    
    for i in tl.static_range(4):
        curr_offset = offset + BLOCK_SIZE * i
        mask = curr_offset < n_elements
        x = tl.load(x_ptr + curr_offset, mask=mask)
        r = tl.where(i == 0, r0, 
            tl.where(i == 1, r1,
                tl.where(i == 2, r2, r3)))
        
        keep = r > p
        output = tl.where(keep, x * scale, 0.0)  
        tl.store(output_ptr + curr_offset, output, mask=mask)

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
        # Save the actual output for backward pass
        ctx.save_for_backward(output)
        ctx.p = p
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        # The mask is implicitly encoded in the output
        # Where output is 0, grad should be 0
        # Where output is nonzero, grad should be scaled by 1/(1-p)
        grad_input = torch.where(output != 0, grad_output, 0.0)
        return grad_input, None, None

