import torch
import pytest
from triformer import TritonDropout

@pytest.mark.parametrize("batch_size,seq_len,hidden_size,p", [
    (1, 128, 256, 0.1),
    (8, 512, 1024, 0.2),
    (16, 256, 512, 0.3),
    (4, 1024, 768, 0.1),


])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestDropout:
    def test_dropout_rate(self, batch_size, seq_len, hidden_size, p):
        # Setup
        torch.manual_seed(42)
        x = torch.ones(batch_size * seq_len, hidden_size, device='cuda', dtype=torch.float32)
        seed = torch.tensor(42, device='cuda')
        
        # Forward pass
        output = TritonDropout.apply(x, p, seed)
        
        # Check dropout rate
        zeros = (output == 0).float().mean()
        assert torch.abs(zeros - p) < 0.1, f"Dropout rate {zeros.item():.3f} differs significantly from target {p}"
        
        # Check scaling of non-zero elements
        nonzero_vals = output[output != 0]
        expected_scale = 1.0 / (1.0 - p)
        assert torch.allclose(nonzero_vals, torch.full_like(nonzero_vals, expected_scale), rtol=1e-5)

    def test_dropout_deterministic(self, batch_size, seq_len, hidden_size, p):
        # Setup
        x = torch.ones(batch_size * seq_len, hidden_size, device='cuda', dtype=torch.float32)
        seed = torch.tensor(42, device='cuda')
        
        # Two forward passes with same seed
        output1 = TritonDropout.apply(x, p, seed)
        output2 = TritonDropout.apply(x, p, seed)
        
        # Check outputs are identical
        assert torch.all(output1 == output2), "Dropout pattern not deterministic with same seed"
        
        # Different seed should give different pattern
        output3 = TritonDropout.apply(x, p, torch.tensor(43, device='cuda'))
        assert not torch.all(output1 == output3), "Different seeds produced same dropout pattern"

    def test_dropout_backward(self, batch_size, seq_len, hidden_size, p):
        # Setup
        x = torch.ones(batch_size * seq_len, hidden_size, device='cuda', dtype=torch.float32, requires_grad=True)
        seed = torch.tensor(42, device='cuda')
        grad_output = torch.randn_like(x)
        
        # Forward + backward pass
        output = TritonDropout.apply(x, p, seed)
        output.backward(grad_output)
        
        # Check gradient properties
        mask = (output != 0).float()
        expected_grad = grad_output * mask / (1.0 - p)
        assert torch.allclose(x.grad, expected_grad, rtol=1e-5)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_numerical_stability():
    # Test numerical stability with extreme values
    x = torch.tensor([[-1e10, 0, 1e10]], device='cuda')
    output = TritonDropout.apply(x, 0.5, torch.tensor(42, device='cuda'))
    
    # Check that we don't have any NaN or inf values
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
    
    # Check that zeros are exactly zero
    mask = output == 0
    assert torch.all(output[mask] == 0)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_memory_efficiency():
    # Test memory efficiency with large inputs
    shape = (32, 32, 1024)  # Typical transformer sequence length
    x = torch.randn(*shape, device='cuda')
    p = 0.1
    
    def measure_memory(func):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        func()
        return torch.cuda.max_memory_allocated()
    
    # Measure PyTorch memory usage
    pytorch_mem = measure_memory(lambda: torch.nn.functional.dropout(x, p, training=True))
    
    # Measure our implementation memory usage
    triton_mem = measure_memory(lambda: TritonDropout.apply(x, p, torch.tensor(42, device='cuda')))
    
    # Our implementation should not use significantly more memory
    assert triton_mem <= pytorch_mem * 1.1  # Allow 10% overhead
