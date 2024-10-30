import triton 
import triton.language as tl
import torch 


# copied from https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/utils.py
next_power_of_2 = triton.next_power_of_2
MAX_FUSED_SIZE = 65536

def calculate_settings(n):
    BLOCK_SIZE = next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(f"Cannot launch Triton kernel since n = {n} exceeds "\
                           f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}.")
    num_warps = 4
    if   BLOCK_SIZE >= 32768: num_warps = 32
    elif BLOCK_SIZE >=  8192: num_warps = 16
    elif BLOCK_SIZE >=  2048: num_warps = 8
    return BLOCK_SIZE, num_warps
pass