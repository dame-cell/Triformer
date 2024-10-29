import torch
import pytest
import numpy as np
from torch.nn.functional import softmax
from triformer.softmax import TritonSoftmax

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_softmax_forward():
    # Test cases with different shapes
    test_cases = [
        (2, 4),  # Simple 2D case
        (2, 8, 4),  # 3D case
        (1, 1, 1024),  # Large dimension case
    ]
    
    for shape in test_cases:
        # Generate random input
        x = torch.randn(*shape, dtype=torch.float32, device='cuda')
        
        # Compute reference result using PyTorch
        ref_output = softmax(x, dim=-1)
            
        # Compute result using our implementation
        triton_layer = TritonSoftmax(causal=False)
        test_output = triton_layer(x)
        
        # Check results match within tolerance
        assert torch.allclose(ref_output, test_output, rtol=1e-4, atol=1e-4), \
            f"Forward pass failed for shape {shape}"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_softmax_backward():
    # Test gradient computation
    shapes = [(2, 4), (2, 8, 4), (1, 1, 1024)]
    
    for shape in shapes:
        x = torch.randn(*shape, dtype=torch.float32, device='cuda', requires_grad=True)
        grad_output = torch.randn(*shape, dtype=torch.float32, device='cuda')
        
        # PyTorch reference
        ref_output = softmax(x, dim=-1)
        ref_output.backward(grad_output)
        ref_grad = x.grad.clone()
        
        # Reset gradients
        x.grad = None
        
        # Our implementation
        triton_layer = TritonSoftmax(causal=False)
        test_output = triton_layer(x)
        test_output.backward(grad_output)
        test_grad = x.grad
        
        # Check gradients match within tolerance
        assert torch.allclose(ref_grad, test_grad, rtol=1e-4, atol=1e-4), \
            f"Backward pass failed for shape {shape}"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_numerical_stability():
    # Test numerical stability with extreme values
    x = torch.tensor([[-1e10, 0, 1e10]], device='cuda')
    triton_layer = TritonSoftmax(causal=False)
    output = triton_layer(x)
    
    # Check that we don't have any NaN or inf values
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
    
    # Sum should be close to 1
    assert torch.allclose(output.sum(dim=-1), torch.ones_like(output.sum(dim=-1)))

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_memory_efficiency():
    # Test memory efficiency with large inputs
    shape = (32, 32, 1024)  # Typical transformer sequence length
    x = torch.randn(*shape, device='cuda')
    
    def measure_memory(func):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        func()
        return torch.cuda.max_memory_allocated()
    
    # Measure PyTorch memory usage
    pytorch_mem = measure_memory(lambda: softmax(x, dim=-1))
    
    # Measure our implementation memory usage
    triton_layer = TritonSoftmax(causal=False)
    triton_mem = measure_memory(lambda: triton_layer(x))
    
    # Our implementation should not use significantly more memory
    assert triton_mem <= pytorch_mem * 1.1  # Allow 10% overhead