import torch
import pytest
import triton
from triformer import TritonLayerNorm

@pytest.mark.parametrize("batch_size,seq_len,hidden_size", [
    (1, 128, 256),
    (8, 512, 1024),
    (16, 256, 512),
])
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
        triton.testing.assert_close(
            triton_output,
            torch_output,
            rtol=1e-1,
            atol=1e-1,
            err_msg="LayerNorm forward pass results don't match!"
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
        
        
        print("triton_ln.weight.grad", triton_ln.weight.grad[0:3])
        print("torch_ln.weight.grad", torch_ln.weight.grad[0:3])
        # Assert gradients match
        triton.testing.assert_close(
            triton_ln.weight.grad,
            torch_ln.weight.grad,
            rtol=1e-1,
            atol=1e-1,
            err_msg="LayerNorm weight gradients don't match!"
        )
        
        print("triton_ln.bias.grad", triton_ln.bias.grad[0:3])
        print("torch_ln.bias.grad", torch_ln.bias.grad[0:3])
        triton.testing.assert_close(
            triton_ln.bias.grad,
            torch_ln.bias.grad,
            rtol=1e-1,
            atol=1e-1,
            err_msg="LayerNorm bias gradients don't match!"
        )

