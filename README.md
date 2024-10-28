# triton-transformers

This will be an implementation of  transformers using triton, 
- This is my first introduction to low-level GPU coding neurel networks i guess. 
- I will try to Also train the model not sure yet but maybe 
- As of right now I am still learning Triton 

### Installation 
- First install triformer 
```bash
pip install triformer
```
- Then you can use the components 
- please keep in mind that the TritonLinear is a fused with relu
- As of right now the TritonLinear is very slow compared to the Pytorch Linear layer, I'm asssuming its because I divided the kernel into 3 parts and the overhead of switching between different kernels is causing the slowdown I'm still looking into it. I might fused the kernels to see if that helps. 


```python
from triformer import TritonLinear
class TritonMLP(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=768):
        super(TritonMLP, self).__init__()
        self.fc1 = TritonLinear(input_size, hidden_size)
        self.fc2 = TritonLinear(hidden_size, hidden_size*2)
        self.fc3 = TritonLinear(hidden_size*2, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
```
You can now try out the layer norm and softmax 

```python
import torch
import triton
from triformer import TritonLayerNorm, TritonSoftmax

# Set random seed for reproducibility
torch.manual_seed(0)

def test_layer_norm_and_softmax():
    # Test parameters
    batch_size = 4
    num_features = 8
    
    # Create input data
    x = torch.randn(batch_size, num_features).cuda()
    
    # Test LayerNorm
    print("Testing LayerNorm...")
    triton_ln = TritonLayerNorm(num_features).cuda()
    torch_ln = torch.nn.LayerNorm(num_features).cuda()
    
    # Copy parameters from torch to triton layer norm to ensure fair comparison
    with torch.no_grad():
        triton_ln.weight.copy_(torch_ln.weight)
        triton_ln.bias.copy_(torch_ln.bias)
    
    triton_ln_out = triton_ln(x)
    torch_ln_out = torch_ln(x)
    
    try:
        triton.testing.assert_close(
            triton_ln_out,
            torch_ln_out,
            rtol=1e-3,
            atol=1e-3,
            err_msg="LayerNorm results don't match!"
        )
        print("✓ LayerNorm test passed!")
    except AssertionError as e:
        print("✗ LayerNorm test failed:", str(e))
        print("Max absolute difference:", torch.max(torch.abs(triton_ln_out - torch_ln_out)))
    
    # Test Softmax
    print("\nTesting Softmax...")
    triton_softmax = TritonSoftmax().cuda()
    torch_softmax = torch.nn.Softmax(dim=-1).cuda()
    
    triton_softmax_out = triton_softmax(x)
    torch_softmax_out = torch_softmax(x)
    
    try:
        triton.testing.assert_close(
            triton_softmax_out,
            torch_softmax_out,
            rtol=1e-3,
            atol=1e-3,
            err_msg="Softmax results don't match!"
        )
        print("✓ Softmax test passed!")
    except AssertionError as e:
        print("✗ Softmax test failed:", str(e))
        print("Max absolute difference:", torch.max(torch.abs(triton_softmax_out - torch_softmax_out)))



if __name__ == "__main__":
    test_layer_norm_and_softmax()

## output:
#Testing LayerNorm...
#✓ LayerNorm test passed!

#Testing Softmax...
#✓ Softmax test passed!
```

### Try it out!

You can try out the TritonMLP on CIFAR10 dataset using this Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tupdi2hgIEY9zSZ9N47LmdUmbn3IE9pO?usp=sharing)



### Layer Normalization

The layer normalization in the triton docs is very fast but when i tested it had some problems first of all the vlaues were not stable so I had to use my own benchmark to test it. 

here are the results:

| Shape | Torch GB/s | Triton GB/s |
|-------|------------|-------------|
| (32, 512, 1024) | 57.70 GB/s | 89.58 GB/s |
| (32, 1024, 1024) | 73.04 GB/s | 115.53 GB/s |
| (64, 512, 1024) | 91.47 GB/s | 115.85 GB/s |
| (32, 1024, 2048) | 102.51 GB/s | 116.50 GB/s |
| (64, 1024, 2048) | 97.83 GB/s | 116.77 GB/s |
| (128, 1024, 2048) | 97.52 GB/s | 116.99 GB/s |
| (32, 2048, 4096) | 82.71 GB/s | 115.87 GB/s |
| (64, 2048, 4096) | 82.36 GB/s | 116.02 GB/s |
| (128, 2048, 4096) | 82.15 GB/s | 116.20 GB/s |

![LayerNorm Performance](triformer/images/layernorm.png)

### Softmax
The softmax kernel is also implemented in Triton and it is blazing fast. it was actually more easier than the layer normalization to implement in triton.


![Softmax Performance](triformer/images/softmax.png)


## Future Plans - To Do
- [ ] Create a library specifically for transformers in vision and language
- [x] Implement the layernorm in Triton 
- [x] Implement the softmax in Triton 
- [ ] add test for each and every component
- [ ] Make the TritonLinear more flexible to either use relu or not
- [ ] Fuse the kernels of TritonLinear to see if it speeds up the training process 

