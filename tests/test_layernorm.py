import torch
import pytest
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
        assert torch.allclose(triton_output, torch_output, rtol=1e-3, atol=1e-3)

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
        assert torch.allclose(triton_ln.weight.grad, torch_ln.weight.grad, rtol=1e-3, atol=1e-3)
        assert torch.allclose(triton_ln.bias.grad, torch_ln.bias.grad, rtol=1e-3, atol=1e-3)

def test_layernorm_training():
    # Setup
    torch.manual_seed(42)
    hidden_size = 256
    batch_size = 32
    seq_len = 128
    
    # Create model with LayerNorm
    class SimpleModel(torch.nn.Module):
        def __init__(self, use_triton=True):
            super().__init__()
            if use_triton:
                self.ln = TritonLayerNorm(hidden_size)
                self.linear = torch.nn.Linear(hidden_size, hidden_size)
            else:
                self.ln = torch.nn.LayerNorm(hidden_size)
                self.linear = torch.nn.Linear(hidden_size, hidden_size)

        def forward(self, x):
            return self.linear(self.ln(x))
    
    # Create models
    triton_model = SimpleModel(use_triton=True).cuda().half()
    torch_model = SimpleModel(use_triton=False).cuda().half()
    
    # Training setup
    criterion = torch.nn.MSELoss()
    triton_optim = torch.optim.Adam(triton_model.parameters(), lr=0.001)
    torch_optim = torch.optim.Adam(torch_model.parameters(), lr=0.001)
    
    # Training loop
    n_steps = 100
    triton_losses = []
    torch_losses = []
    
    for _ in range(n_steps):
        x = torch.randn(batch_size * seq_len, hidden_size, device='cuda', dtype=torch.float16)
        target = torch.randn(batch_size * seq_len, hidden_size, device='cuda', dtype=torch.float16)
        
        # Triton model step
        triton_optim.zero_grad()
        triton_output = triton_model(x)
        triton_loss = criterion(triton_output, target)
        triton_loss.backward()
        triton_optim.step()
        triton_losses.append(triton_loss.item())
        
        # PyTorch model step
        torch_optim.zero_grad()
        torch_output = torch_model(x)
        torch_loss = criterion(torch_output, target)
        torch_loss.backward()
        torch_optim.step()
        torch_losses.append(torch_loss.item())
    
    # Assert both models are learning
    assert triton_losses[-1] < triton_losses[0]
    assert torch_losses[-1] < torch_losses[0]
    
    # Assert similar convergence
    assert abs(triton_losses[-1] - torch_losses[-1]) < 0.1 