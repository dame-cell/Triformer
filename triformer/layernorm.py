import triton
import triton.language as tl
import torch 
import torch.nn as nn 
from torch.autograd import Function
import time 
import matplotlib.pyplot as plt


    
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
  


class TritonLayerNorm(Function):
    @staticmethod
    def forward(ctx, input, gamma, beta, eps):  # Add `ctx` as the first argument
        rows, cols = input.shape 
        assert input.dim() == 2, "We are working with only 2D tensors for now" 
        block_size = triton.next_power_of_2(cols)
        num_warps = 4 
        if block_size == 2047:
            num_warps = 8 
        if block_size == 4095:
            num_warps = 16 

        sm_out = torch.empty_like(input)

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





def benchmark_layer_norm_with_plot(dtype=torch.float16, num_runs=100):  # Increase runs to 2000 for longer runtime
    M = 4096   # Fixed M dimension
    N_values = [512 * i for i in range(2, 24)]  # Testing only 6 points instead of 30

    torch_throughputs = []
    triton_throughputs = []

    for N in N_values:
        print(f"\nBenchmarking N={N}")
        # Create data
        x_shape = (M, N)
        w_shape = (x_shape[-1],)
        gamma = torch.ones(w_shape, dtype=dtype, device='cuda')
        beta = torch.zeros(w_shape, dtype=dtype, device='cuda')
        x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device='cuda')

        # PyTorch LayerNorm
        torch_ln = torch.nn.LayerNorm(N).to('cuda').to(dtype)

        # Warmup
        for _ in range(10):
            torch_ln(x)
            TritonLayerNorm.apply(x, gamma, beta, 1e-5)

        # Benchmark functions
        def bench_torch():
            torch_ln(x)

        def bench_triton():
            TritonLayerNorm.apply(x, gamma, beta, 1e-5)

        # Benchmark PyTorch
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(num_runs):
            bench_torch()
            torch.cuda.synchronize()
        torch_time = (time.time() - start_time) / num_runs

        # Benchmark Triton
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(num_runs):
            bench_triton()
            torch.cuda.synchronize()
        triton_time = (time.time() - start_time) / num_runs

        # Calculate throughput (GB/s)
        bytes_processed = 2 * x.numel() * x.element_size()  # input + output
        torch_throughput = bytes_processed / torch_time / 1e9
        triton_throughput = bytes_processed / triton_time / 1e9

        torch_throughputs.append(torch_throughput)
        triton_throughputs.append(triton_throughput)

        # Print current iteration results
        print(f"N={N}:")
        print(f"PyTorch: {torch_time*1000:.3f} ms, {torch_throughput:.2f} GB/s")
        print(f"Triton:  {triton_time*1000:.3f} ms, {triton_throughput:.2f} GB/s")
        print(f"Speedup: {torch_time/triton_time:.2f}x")

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(N_values, torch_throughputs, 'g-', label='Torch', marker='o')
    plt.plot(N_values, triton_throughputs, 'b-', label='Triton', marker='o')

    plt.xlabel('N')
    plt.ylabel('GB/s')
    plt.title(f'LayerNorm Performance (M={M})')
    plt.legend()
    plt.grid(True)

    # Add throughput values as text
    for i, N in enumerate(N_values):
        plt.text(N, torch_throughputs[i], f'{torch_throughputs[i]:.0f}',
                 verticalalignment='bottom')
        plt.text(N, triton_throughputs[i], f'{triton_throughputs[i]:.0f}',
                 verticalalignment='top')

    # Display the plot
    plt.savefig('layer_norm_performance2.png')
    plt.show()

    # Print summary
    print("\nSummary:")
    print(f"{'N':<8} {'Torch (GB/s)':<15} {'Triton (GB/s)':<15} {'Speedup':<10}")
    print("-" * 48)
    for i, N in enumerate(N_values):
        speedup = triton_throughputs[i] / torch_throughputs[i]
        print(f"{N:<8} {torch_throughputs[i]:<15.2f} {triton_throughputs[i]:<15.2f} {speedup:<10.2f}x")

    return N_values, torch_throughputs, triton_throughputs


if __name__ == "__main__":
    # Run benchmark and create plot
    N_values, torch_throughputs, triton_throughputs = benchmark_layer_norm_with_plot()


