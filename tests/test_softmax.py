import torch
import triton
import pytest
from triformer import TritonSoftmax

class TestSoftmax:
    @pytest.mark.parametrize(
        "batch_size,seq_len,hidden_size,dtype",
        [
            (1, 128, 256, torch.float32),
            (8, 512, 1024, torch.float32),
            (16, 256, 512, torch.float32),
            (1, 128, 256, torch.float16),
            (8, 512, 1024, torch.float16),
            (16, 256, 512, torch.float16),
        ]
    )
    def test_forward_match(self, batch_size, seq_len, hidden_size, dtype):
        # Setup
        torch.manual_seed(42)
        x = torch.randn(batch_size * seq_len, hidden_size, device='cuda', dtype=dtype)
        
        # Test both regular and causal softmax
        for causal in [False, True]:
            # Create implementations
            triton_softmax = TritonSoftmax(causal=causal).cuda()
            
            # Forward pass
            with torch.no_grad():
                triton_output = triton_softmax(x)
                # For causal comparison, we need to apply mask to torch softmax
                if causal:  
                    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().cuda()
                    x_masked = x.masked_fill(mask, float('-inf'))
                    torch_output = torch.nn.functional.softmax(x_masked, dim=-1)
                else:
                    torch_output = torch.nn.functional.softmax(x, dim=-1)
            
            # Adjust tolerances based on dtype
            rtol = 1e-3 if dtype == torch.float32 else 1e-2
            atol = 1e-3 if dtype == torch.float32 else 1e-2
            
            # Assert
            triton.testing.assert_close(
                triton_output,
                torch_output,
                rtol=rtol,
                atol=atol,
                err_msg=f"Softmax forward pass results don't match for dtype={dtype}, causal={causal}!"
            )
            
            # Additional assertions for numerical stability
            assert torch.all(triton_output >= 0), "Softmax output should be non-negative"
            assert torch.all(triton_output <= 1), "Softmax output should be less than or equal to 1"
            assert torch.allclose(triton_output.sum(dim=-1), torch.ones(batch_size, device='cuda'), 
                                   rtol=1e-5, atol=1e-5), "Softmax outputs should sum to 1"

    def test_numerical_stability(self):
        # Test with extreme values
        test_cases = [
            torch.tensor([[1000., 0., -1000.]], device='cuda'),
            torch.tensor([[-1000., -1000., -1000.]], device='cuda'),
            torch.tensor([[0., 0., 0.]], device='cuda'),
        ]
        
        triton_softmax = TritonSoftmax().cuda()
        
        for x in test_cases:
            triton_output = triton_softmax(x)
            torch_output = torch.nn.functional.softmax(x, dim=-1)
            
            # Check results
            triton.testing.assert_close(
                triton_output,
                torch_output,
                rtol=1e-3,
                atol=1e-3,
            )
            
            # Check sum = 1
            assert torch.allclose(triton_output.sum(dim=-1), 
                                torch.ones_like(triton_output.sum(dim=-1)))

    def test_backward_pass(self):
        # Test gradient computation
        x = torch.randn(2, 3, requires_grad=True, device='cuda')
        triton_softmax = TritonSoftmax().cuda()
        
        out = triton_softmax(x)
        loss = out.sum()
        loss.backward()
        
        # Should not have NaN gradients
        assert not torch.isnan(x.grad).any()

    def test_backward_match(self, batch_size, seq_len, hidden_size, dtype):
        # Setup
        torch.manual_seed(42)
        x = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=dtype, requires_grad=True)
        grad_output = torch.randn_like(x)
        
        # Create implementations
        triton_softmax = TritonSoftmax().cuda()
        torch_softmax = torch.nn.functional.softmax
        
        # Forward pass
        triton_output = triton_softmax(x)
        torch_output = torch_softmax(x, dim=-1)
        
        # Backward pass
        triton_output.backward(grad_output)
        torch_output.backward(grad_output)
        
        # Assert gradients match
        triton.testing.assert_close(
            triton_softmax.weight.grad,
            torch_softmax.weight.grad,
            rtol=1e-0,
            atol=1e-0,
            err_msg="Softmax weight gradients don't match!"
        )
        
        # Check for NaN gradients
        assert not torch.isnan(x.grad).any(), "Gradients should not contain NaN values"