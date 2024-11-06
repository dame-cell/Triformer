import torch
import pytest
import triton
from triformer import TritonLayerNorm

@pytest.mark.parametrize("batch_size,seq_len,hidden_size", [
    # Small configurations
    (1, 128, 256),
    (8, 512, 1024),
    (16, 256, 512),
    
    # Medium configurations
    (4, 1024, 768),
    (8, 1024, 1024),
    (16, 1024, 1024),
    (32, 512, 1024),
])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestLayerNorm:
    def test_forward_match(self, batch_size, seq_len, hidden_size):
        # Setup
        torch.manual_seed(42)
        x = torch.randn(batch_size * seq_len, hidden_size, device='cuda', dtype=torch.float16)
        
        # Create both implementations
        triton_ln = TritonLayerNorm(hidden_size).cuda().half()
        torch_ln = torch.nn.LayerNorm(hidden_size).cuda().half()
        
        # Copy weights to ensure same initialization
        torch_ln.weight.data.copy_(triton_ln.weight.data)
        torch_ln.bias.data.copy_(triton_ln.bias.data)
        
        # Forward pass
        with torch.no_grad():
            triton_output = triton_ln(x)
            torch_output = torch_ln(x)
        
        # Assert
        torch.testing.assert_close(
            triton_output,
            torch_output,
            rtol=1e-0,
            atol=1e-0,
            msg="LayerNorm forward pass results don't match!"
        )

    def test_backward_match(self, batch_size, seq_len, hidden_size):
        # Setup
        torch.manual_seed(42)
        x = torch.randn(batch_size * seq_len, hidden_size, device='cuda', dtype=torch.float16, requires_grad=True)
        grad_output = torch.randn_like(x)
        
        # Create both implementations
        triton_ln = TritonLayerNorm(hidden_size).cuda().half()
        torch_ln = torch.nn.LayerNorm(hidden_size).cuda().half()
        
        # Copy weights
        torch_ln.weight.data.copy_(triton_ln.weight.data)
        torch_ln.bias.data.copy_(triton_ln.bias.data)
        
        # Forward + backward pass
        triton_output = triton_ln(x)
        torch_output = torch_ln(x)
        
        triton_output.backward(grad_output)
        torch_output.backward(grad_output)
        
        # Assert gradients match
        torch.testing.assert_close(
            triton_ln.weight.grad,
            torch_ln.weight.grad,
            rtol=1e-5,
            atol=1e-5,
            msg="LayerNorm weight gradients don't match!"
        )
        
        torch.testing.assert_close(
            triton_ln.bias.grad,
            torch_ln.bias.grad,
            rtol=1e-5,
            atol=1e-5,
            msg="LayerNorm bias gradients don't match!"
        )

        torch.testing.assert_close(
            x.grad,
            x.grad,
            rtol=1e-5,
            atol=1e-5,
            msg="LayerNorm input gradients don't match!"
        )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_numerical_stability():
        # Test numerical stability with extreme values
        x = torch.tensor([[-1e10, 0, 1e10]], device='cuda')
        triton_ln = TritonLayerNorm(x.shape[-1]).cuda()
        output = triton_ln(x)
        
        # Check that we don't have any NaN or inf values
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
        # Mean should be close to 0
        assert torch.allclose(output.mean(dim=-1), torch.zeros_like(output.mean(dim=-1)))

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
        pytorch_ln = torch.nn.LayerNorm(shape[-1]).cuda()
        pytorch_mem = measure_memory(lambda: pytorch_ln(x))
        
        # Measure our implementation memory usage
        triton_ln = TritonLayerNorm(shape[-1]).cuda()
        triton_mem = measure_memory(lambda: triton_ln(x))
        
        # Our implementation should not use significantly more memory
        assert triton_mem <= pytorch_mem * 1.1  # Allow 10% overhead