from triformer.layernorm import TritonLayerNormFunction
import torch 
import torch.nn as nn

def test_layernorm_implementation():
    # Test with different sizes
    batch_sizes = [1, 8, 32]
    hidden_sizes = [256, 512, 1024]
    
    for batch_size in batch_sizes:
        for hidden_size in hidden_sizes:
            print(f"\nTesting with batch_size={batch_size}, hidden_size={hidden_size}")
            
            eps = 1e-5
            torch.manual_seed(0)
            
            # Create random input tensor
            x = torch.randn(batch_size, hidden_size, device='cuda', requires_grad=True)
            gamma = torch.randn(hidden_size, device='cuda', requires_grad=True)
            beta = torch.randn(hidden_size, device='cuda', requires_grad=True)
            
            # PyTorch implementation
            torch_ln = nn.LayerNorm(hidden_size, eps=eps).cuda()
            torch_ln.weight = nn.Parameter(gamma.clone())
            torch_ln.bias = nn.Parameter(beta.clone())
            
            # Forward pass
            triton_out = TritonLayerNormFunction.apply(x.clone(), gamma.clone(), beta.clone(), eps)
            torch_out = torch_ln(x.clone())
            
            # Check forward pass
            max_diff = torch.max(torch.abs(triton_out - torch_out))
            print(f"Forward pass max diff: {max_diff:.8f}")
            
            # Backward pass
            grad_output = torch.randn_like(x)
            triton_out.backward(grad_output)
            torch_out.backward(grad_output)
            
            # Compare gradients
            grad_input_diff = torch.max(torch.abs(x.grad - x.grad))
            grad_gamma_diff = torch.max(torch.abs(gamma.grad - torch_ln.weight.grad))
            grad_beta_diff = torch.max(torch.abs(beta.grad - torch_ln.bias.grad))
            
            print(f"Gradient diffs - Input: {grad_input_diff:.8f}, Gamma: {grad_gamma_diff:.8f}, Beta: {grad_beta_diff:.8f}")


if __name__ == "__main__":
    test_layernorm_implementation()


