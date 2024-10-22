import triton
import triton.language as tl
import torch
import torch.nn as nn


# -------------------------------
# Triton Kernels for Linear Layer
# -------------------------------

@triton.jit
def linear_kernel(X, W, Y, M, N, K, BLOCK_SIZE: tl.constexpr):
    m = tl.program_id(0)  # Batch dimension
    n = tl.program_id(1)  # Output feature dimension

    acc = 0.0  # Initialize as a scalar

    for k_start in range(0, K, BLOCK_SIZE):
        k = k_start + tl.arange(0, BLOCK_SIZE)
        mask = k < K

        # Load X[m, k]
        x_ptrs = X + m * K + k
        x = tl.load(x_ptrs, mask=mask, other=0.0)

        # Load W[k, n]
        w_ptrs = W + k * N + n
        w = tl.load(w_ptrs, mask=mask, other=0.0)

        acc += tl.sum(x * w, axis=0)  

    # Store the accumulated scalar result
    tl.store(Y + m * N + n, acc)

@triton.jit
def linear_backward_x_kernel(dY, W, dX, M, N, K, BLOCK_SIZE: tl.constexpr):
    m = tl.program_id(0)  # Batch dimension
    k = tl.program_id(1)  # Input feature dimension

    acc = 0.0  # Initialize as a scalar

    for n_start in range(0, N, BLOCK_SIZE):
        n = n_start + tl.arange(0, BLOCK_SIZE)
        mask = n < N

        # Load W[k, n]
        w_ptrs = W + k * N + n
        w = tl.load(w_ptrs, mask=mask, other=0.0)

        # Load dY[m, n]
        dy_ptrs = dY + m * N + n
        dy = tl.load(dy_ptrs, mask=mask, other=0.0)

        acc += tl.sum(dy * w, axis=0)  # Ensure scalar accumulation

    # Store the accumulated scalar gradient
    tl.store(dX + m * K + k, acc)

@triton.jit
def linear_backward_w_kernel(dY, X, dW, M, N, K, BLOCK_SIZE: tl.constexpr):
    k = tl.program_id(0)  # Input feature dimension
    n = tl.program_id(1)  # Output feature dimension

    acc = 0.0  # Initialize as a scalar

    for m_start in range(0, M, BLOCK_SIZE):
        m = m_start + tl.arange(0, BLOCK_SIZE)
        mask = m < M

        # Load X[m, k]
        x_ptrs = X + m * K + k
        x = tl.load(x_ptrs, mask=mask, other=0.0)

        # Load dY[m, n]
        dy_ptrs = dY + m * N + n
        dy = tl.load(dy_ptrs, mask=mask, other=0.0)

        acc += tl.sum(x * dy, axis=0)  # Ensure scalar accumulation

    # Store the accumulated scalar gradient
    tl.store(dW + k * N + n, acc)

# -----------------------------------
# Custom Autograd Function for Linear
# -----------------------------------

class TritonLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, bias, BLOCK_SIZE=128):
        M, K = X.shape
        K_w, N = W.shape

        Y = torch.empty((M, N), device=X.device, dtype=X.dtype)

        grid = (M, N)
        linear_kernel[grid](
            X, W, Y,
            M, N, K,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        Y += bias

        ctx.save_for_backward(X, W)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        return Y

    @staticmethod
    def backward(ctx, dY):
        X, W = ctx.saved_tensors
        BLOCK_SIZE = ctx.BLOCK_SIZE
        M, K = X.shape
        K_w, N = W.shape

        dX = torch.empty_like(X)
        dW = torch.empty_like(W)
        dbias = torch.empty(N, device=X.device, dtype=X.dtype)

        # Compute dX using the corrected kernel
        grid_dX = (M, K)
        linear_backward_x_kernel[grid_dX](
            dY, W, dX,
            M, N, K,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # Compute dW using the corrected kernel
        grid_dW = (K, N)
        linear_backward_w_kernel[grid_dW](
            dY, X, dW,
            M, N, K,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # Compute dbias
        dbias = dY.sum(0)

        return dX, dW, dbias, None  # None for BLOCK_SIZE

# ------------------------------
# TritonLinear PyTorch Module
# ------------------------------

class TritonLinear(nn.Module):
    def __init__(self, in_features, out_features, BLOCK_SIZE=128):
        super(TritonLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features, device='cuda') * (2. / in_features) ** 0.5)
        self.bias = nn.Parameter(torch.zeros(out_features, device='cuda'))
        self.BLOCK_SIZE = BLOCK_SIZE

    def forward(self, x):
        return TritonLinearFunction.apply(x, self.weight, self.bias, self.BLOCK_SIZE)
