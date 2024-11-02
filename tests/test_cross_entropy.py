import torch
import pytest
from triformer import TritonCrossEntropyLoss

@pytest.mark.parametrize("batch_size,seq_len,vocab_size,pad_ratio", [
    # Small configurations
    (1, 32, 100, 0.0),
    (8, 64, 1000, 0.1),
    (16, 128, 5000, 0.2),
    
    # Medium configurations
    (4, 256, 10000, 0.1),
    (8, 512, 30000, 0.15),
    (32, 128, 50000, 0.2),
])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCrossEntropy:
    def test_forward_match(self, batch_size, seq_len, vocab_size, pad_ratio):
        # Setup
        torch.manual_seed(42)
        logits = torch.randn(batch_size, seq_len, vocab_size, device='cuda')
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device='cuda')
        
        # Add padding tokens
        pad_mask = torch.rand(batch_size, seq_len, device='cuda') < pad_ratio
        targets[pad_mask] = -100
        
        # Create both implementations
        triton_ce = TritonCrossEntropyLoss(pad_token_id=-100, reduction="mean",n_chunks=2)
        torch_ce = torch.nn.CrossEntropyLoss(ignore_index=-100)
        
        # Forward pass
        with torch.no_grad():
            triton_loss = triton_ce(logits, targets)
            torch_loss = torch_ce(logits.view(-1, vocab_size), targets.view(-1))
        
        # Assert
        torch.testing.assert_close(
            triton_loss,
            torch_loss,
            rtol=1e-5,
            atol=1e-5,
            msg="CrossEntropy forward pass results don't match!"
        )

    def test_backward_match(self, batch_size, seq_len, vocab_size, pad_ratio):
        # Setup
        torch.manual_seed(42)
        logits = torch.randn(batch_size, seq_len, vocab_size, device='cuda', requires_grad=True)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device='cuda')
        
        # Add padding tokens
        pad_mask = torch.rand(batch_size, seq_len, device='cuda') < pad_ratio
        targets[pad_mask] = -100
        
        # Create both implementations
        triton_ce = TritonCrossEntropyLoss(pad_token_id=-100, reduction="mean",n_chunks=2)
        torch_ce = torch.nn.CrossEntropyLoss(ignore_index=-100)
        
        # Forward + backward pass
        triton_loss = triton_ce(logits, targets)
        triton_loss.backward()
        triton_grad = logits.grad.clone()
        
        logits.grad = None
        
        torch_loss = torch_ce(logits.view(-1, vocab_size), targets.view(-1))
        torch_loss.backward()
        torch_grad = logits.grad
        
        # Assert gradients match
        torch.testing.assert_close(
            triton_grad,
            torch_grad,
            rtol=1e-5,
            atol=1e-5,
            msg="CrossEntropy gradients don't match!"
        )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_all_padding():
    # Test case where all tokens are padding
    logits = torch.randn(2, 4, 100, device='cuda', requires_grad=True)
    targets = torch.full((2, 4), -100, device='cuda')
    
    triton_ce = TritonCrossEntropyLoss(pad_token_id=-100, reduction="mean",n_chunks=2)
    torch_ce = torch.nn.CrossEntropyLoss(ignore_index=-100)
    
    # Forward pass
    triton_loss = triton_ce(logits, targets)
    torch_loss = torch_ce(logits.view(-1, 100), targets.view(-1))
    
    assert triton_loss.item() == 0.0, "All-padding case should return 0 loss"
    
    # Backward pass
    triton_loss.backward()
    assert torch.all(logits.grad == 0), "All-padding case should have zero gradients"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_numerical_stability():
    # Test numerical stability with extreme values
    logits = torch.tensor([[1e10, -1e10]], device='cuda', requires_grad=True)
    targets = torch.tensor([0], device='cuda')
    
    triton_ce = TritonCrossEntropyLoss(pad_token_id=-100, reduction="mean",n_chunks=2)
    output = triton_ce(logits, targets)
    
    # Check that we don't have any NaN or inf values
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
    
    # Test backward pass stability
    output.backward()
    assert not torch.isnan(logits.grad).any()
    assert not torch.isinf(logits.grad).any()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_memory_efficiency():
    # Test memory efficiency with large inputs
    shape = (32, 512, 50000)  # Typical transformer logits shape
    logits = torch.randn(*shape, device='cuda')
    targets = torch.randint(0, shape[-1], (shape[0], shape[1]), device='cuda')
    
    def measure_memory(func):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        func()
        return torch.cuda.max_memory_allocated()
    
    # Measure PyTorch memory usage
    torch_ce = torch.nn.CrossEntropyLoss(ignore_index=-100)
    pytorch_mem = measure_memory(
        lambda: torch_ce(logits.view(-1, shape[-1]), targets.view(-1))
    )
    
    # Measure our implementation memory usage
    triton_ce = TritonCrossEntropyLoss(pad_token_id=-100, reduction="mean",n_chunks=2)
    triton_mem = measure_memory(
        lambda: triton_ce(logits, targets)
    )
    
    # Our implementation should not use significantly more memory
    assert triton_mem <= pytorch_mem * 1.1  # Allow 10% overhead
