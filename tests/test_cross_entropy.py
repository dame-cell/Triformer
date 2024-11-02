import torch
import pytest
from triformer import TritonCrossEntropyLoss

@pytest.mark.parametrize("batch_size,seq_len,vocab_size,n_chunks", [
    # Small configurations
    (2, 8, 100, 1),
    (4, 32, 1000, 2),
    (8, 64, 2000, 2),
    
    # Medium configurations
    (16, 128, 5000, 4),
    (32, 256, 10000, 4),
])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCrossEntropy:
    def test_forward_match(self, batch_size, seq_len, vocab_size, n_chunks):
        # Setup
        torch.manual_seed(42)
        logits = torch.randn(batch_size, seq_len, vocab_size, device='cuda', dtype=torch.float32)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device='cuda')
        
        # Add padding tokens
        pad_mask = torch.rand(batch_size, seq_len, device='cuda') < 0.1
        targets[pad_mask] = -100
        
        # Create both implementations
        triton_ce = TritonCrossEntropyLoss(pad_token_id=-100, reduction="mean", n_chunks=n_chunks)
        torch_ce = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
        
        # Forward pass
        with torch.no_grad():
            triton_logits = logits.clone()  # Create separate copies
            torch_logits = logits.clone()
            
            triton_loss = triton_ce(triton_logits, targets)
            torch_loss = torch_ce(torch_logits.view(-1, vocab_size), targets.view(-1))
        
        # Assert
        torch.testing.assert_close(
            triton_loss,
            torch_loss,
            rtol=1e-3,
            atol=1e-3,
            msg=f"CrossEntropy forward pass results don't match for shape {(batch_size, seq_len, vocab_size)}!"
        )

    def test_backward_match(self, batch_size, seq_len, vocab_size, n_chunks):
        # Setup
        torch.manual_seed(42)
        logits = torch.randn(batch_size, seq_len, vocab_size, device='cuda', dtype=torch.float32)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device='cuda')
        pad_mask = torch.rand(batch_size, seq_len, device='cuda') < 0.1
        targets[pad_mask] = -100
        
        # Create both implementations
        triton_ce = TritonCrossEntropyLoss(pad_token_id=-100, reduction="mean", n_chunks=n_chunks)
        torch_ce = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
        
        # Forward + backward pass
        triton_logits = logits.clone().requires_grad_(True)
        torch_logits = logits.clone().requires_grad_(True)
        
        triton_loss = triton_ce(triton_logits, targets)
        torch_loss = torch_ce(torch_logits.view(-1, vocab_size), targets.view(-1))
        
        triton_loss.backward()
        torch_loss.backward()
        
        # Assert gradients match
        torch.testing.assert_close(
            triton_logits.grad,
            torch_logits.grad,
            rtol=1e-3,
            atol=1e-3,
            msg=f"CrossEntropy gradients don't match for shape {(batch_size, seq_len, vocab_size)}!"
        )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_numerical_stability():
    # Test numerical stability with extreme values
    logits = torch.tensor([[[-1e10, 0, 1e10]]], device='cuda')
    targets = torch.tensor([[1]], device='cuda')
    
    triton_ce = TritonCrossEntropyLoss(pad_token_id=-100, reduction="mean")
    output = triton_ce(logits, targets)
    
    # Check that we don't have any NaN or inf values
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_memory_efficiency():
    # Test memory efficiency with medium-sized inputs
    batch_size, seq_len, vocab_size = 16, 128, 5000  # Reduced size
    logits = torch.randn(batch_size, seq_len, vocab_size, device='cuda')
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device='cuda')
    
    def measure_memory(func):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        func()
        return torch.cuda.max_memory_allocated()
    
    # Measure PyTorch memory usage
    torch_ce = torch.nn.CrossEntropyLoss(ignore_index=-100)
    pytorch_mem = measure_memory(lambda: torch_ce(logits.view(-1, vocab_size), targets.view(-1)))
    
    # Measure our implementation memory usage with different chunk sizes
    for n_chunks in [1, 2, 4]:
        triton_ce = TritonCrossEntropyLoss(pad_token_id=-100, reduction="mean", n_chunks=n_chunks)
        triton_mem = measure_memory(lambda: triton_ce(logits, targets))
        
        # Print memory usage
        print(f"\nChunks={n_chunks}:")
        print(f"PyTorch Memory: {pytorch_mem/1024**2:.1f} MB")
        print(f"Triton Memory: {triton_mem/1024**2:.1f} MB")
        print(f"Memory Reduction: {pytorch_mem/triton_mem:.2f}x")
        
        # Our implementation should use less memory
        assert triton_mem < pytorch_mem, f"Triton implementation with {n_chunks} chunks uses more memory than PyTorch"

