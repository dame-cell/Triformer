import torch
import triton
import pytest
from triformer import TritonSoftmax 

class TestSoftmax:
    @pytest.mark.parametrize(
        "batch_size,num_heads,seq_len",
        [
            (1, 8, 128),    # Small batch
            (4, 12, 512),   # Medium batch
            (16, 16, 1024), # Large batch
        ]
    )
    def test_attention_softmax(self, batch_size, num_heads, seq_len):
        # Setup
        torch.manual_seed(42)
        
        # Create attention scores (Q @ K^T)
        scores = torch.randn(
            batch_size, 
            num_heads, 
            seq_len, 
            seq_len, 
            device='cuda'
        )
        
        # Create implementations
        triton_softmax = TritonSoftmax(dim=-1).cuda()
        
        # Forward pass with causal masking
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        
        with torch.no_grad():
            triton_output = triton_softmax(scores)
            torch_output = torch.nn.functional.softmax(scores, dim=-1)
        
        # Assertions
        triton.testing.assert_close(
            triton_output,
            torch_output,
            rtol=1e-3,
            atol=1e-3,
        )
        
        # Check row sums = 1
        row_sums = triton_output.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums))
        
        # Check causality
        assert torch.all(triton_output.triu(diagonal=1) == 0)

    @pytest.mark.parametrize(
        "batch_size,seq_len,vocab_size",
        [
            (1, 128, 32000),    # Small batch
            (4, 512, 32000),    # Medium batch
            (16, 1024, 32000),  # Large batch
        ]
    )
    def test_vocab_softmax(self, batch_size, seq_len, vocab_size):
        # Setup
        torch.manual_seed(42)
        
        # Create logits
        logits = torch.randn(
            batch_size,
            seq_len,
            vocab_size,
            device='cuda'
        )
        
        # Create implementations
        triton_softmax = TritonSoftmax(dim=-1).cuda()
        
        # Forward pass
        with torch.no_grad():
            triton_output = triton_softmax(logits)
            torch_output = torch.nn.functional.softmax(logits, dim=-1)
        
        # Assertions
        triton.testing.assert_close(
            triton_output,
            torch_output,
            rtol=1e-3,
            atol=1e-3,
        )
        
        # Check row sums = 1
        row_sums = triton_output.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums))

    def test_numerical_stability(self):
        # Test with very large values
        x = torch.tensor([[1000., 0., -1000.]], device='cuda')
        triton_output = TritonSoftmax()(x)
        torch_output = torch.nn.functional.softmax(x, dim=-1)
        
        triton.testing.assert_close(
            triton_output,
            torch_output,
            rtol=1e-3,
            atol=1e-3,
        )
        
        # Test with very small values
        x = torch.tensor([[-1000., -1000., -1000.]], device='cuda')
        triton_output = TritonSoftmax()(x)
        torch_output = torch.nn.functional.softmax(x, dim=-1)
        
        triton.testing.assert_close(
            triton_output,
            torch_output,
            rtol=1e-3,
            atol=1e-3,
        )