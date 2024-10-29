import torch
import pytest
from triformer import TritonSoftmax
import triton

@pytest.mark.parametrize("batch_size,seq_len,hidden_size,dtype", [
    (1, 128, 256, torch.float32),
    (8, 512, 1024, torch.float32),
    (16, 256, 512, torch.float32),
    (1, 128, 256, torch.float16),
    (8, 512, 1024, torch.float16),
    (16, 256, 512, torch.float16),
])
class TestSoftmax:
    def test_forward_match(self, batch_size, seq_len, hidden_size, dtype):
        # Setup
        torch.manual_seed(42)
        x = torch.randn(batch_size * seq_len, hidden_size, device='cuda', dtype=dtype)
        
        # Create implementations
        triton_softmax = TritonSoftmax().cuda()
        
        # Forward pass
        with torch.no_grad():
            triton_output = triton_softmax(x)
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
            err_msg=f"Softmax forward pass results don't match for dtype={dtype}!"
        )

    def test_backward_match(self, batch_size, seq_len, hidden_size, dtype):
        # Setup
        torch.manual_seed(42)
        x = torch.randn(batch_size * seq_len, hidden_size, device='cuda', dtype=dtype, requires_grad=True)
        x_clone = x.clone().detach().requires_grad_(True)
        grad_output = torch.randn_like(x)
        
        # Forward + backward pass
        triton_output = TritonSoftmax()(x)
        torch_output = torch.nn.functional.softmax(x_clone, dim=-1)
        
        triton_output.backward(grad_output)
        torch_output.backward(grad_output)
        
        # Assert gradients match
        triton.testing.assert_close(
            x.grad,
            x_clone.grad,
            rtol=1e-3,
            atol=1e-3,
            err_msg="Softmax backward pass gradients don't match!"
        )

def test_softmax_training():
    # Setup
    torch.manual_seed(42)
    hidden_size = 256
    batch_size = 32
    seq_len = 128
    
    # Create model with Softmax
    class SimpleModel(torch.nn.Module):
        def __init__(self, use_triton=True):
            super().__init__()
            if use_triton:
                self.softmax = TritonSoftmax()
                self.linear = torch.nn.Linear(hidden_size, hidden_size)
            else:
                self.softmax = torch.nn.Softmax(dim=-1)
                self.linear = torch.nn.Linear(hidden_size, hidden_size)

        def forward(self, x):
            return self.linear(self.softmax(x))
    
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
    assert triton_losses[-1] < triton_losses[0], "Triton model is not learning"
    assert torch_losses[-1] < torch_losses[0], "PyTorch model is not learning"
    
   