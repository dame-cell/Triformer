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

### Try it out!

You can try out the TritonMLP on CIFAR10 dataset using this Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tupdi2hgIEY9zSZ9N47LmdUmbn3IE9pO?usp=sharing)



### Layer Normalization

The layer normalization in the triton docs is very fast but when i tested it had some problems first of all the vlaues were not stable so I had to use my own benchmark to test it. 

here are the results:

| Shape | Size | Torch Time | Torch GB/s | Torch Reserved | Torch Allocated | Triton Time | Triton GB/s | Triton Reserved | Triton Allocated | Speedup |
|-------|------|------------|------------|----------------|-----------------|-------------|-------------|-----------------|------------------|----------|
| (32, 512, 1024) | 0.03 GB | 0.54 ms | 57.70 GB/s | 0.06 GB | 0.06 GB | 0.35 ms | 89.58 GB/s | 0.10 GB | 0.06 GB | 1.55x |
| (32, 1024, 1024) | 0.06 GB | 0.86 ms | 73.04 GB/s | 0.13 GB | 0.13 GB | 0.54 ms | 115.53 GB/s | 0.19 GB | 0.13 GB | 1.58x |
| (64, 512, 1024) | 0.06 GB | 0.68 ms | 91.47 GB/s | 0.13 GB | 0.13 GB | 0.54 ms | 115.85 GB/s | 0.19 GB | 0.13 GB | 1.27x |
| (32, 1024, 2048) | 0.12 GB | 1.22 ms | 102.51 GB/s | 0.25 GB | 0.25 GB | 1.07 ms | 116.50 GB/s | 0.38 GB | 0.25 GB | 1.14x |
| (64, 1024, 2048) | 0.25 GB | 2.56 ms | 97.83 GB/s | 0.50 GB | 0.50 GB | 2.14 ms | 116.77 GB/s | 0.75 GB | 0.50 GB | 1.19x |
| (128, 1024, 2048) | 0.50 GB | 5.13 ms | 97.52 GB/s | 1.00 GB | 1.00 GB | 4.27 ms | 116.99 GB/s | 1.50 GB | 1.00 GB | 1.20x |
| (32, 2048, 4096) | 0.50 GB | 6.05 ms | 82.71 GB/s | 1.00 GB | 1.00 GB | 4.32 ms | 115.87 GB/s | 1.50 GB | 1.00 GB | 1.40x |
| (64, 2048, 4096) | 1.00 GB | 12.14 ms | 82.36 GB/s | 2.00 GB | 2.00 GB | 8.62 ms | 116.02 GB/s | 3.00 GB | 2.00 GB | 1.41x |
| (128, 2048, 4096) | 2.00 GB | 24.35 ms | 82.15 GB/s | 4.00 GB | 4.00 GB | 17.21 ms | 116.20 GB/s | 6.01 GB | 4.00 GB | 1.41x |

![LayerNorm Performance](Triformer\images\layernorm.png)

### Softmax
The softmax kernel is also implemented in Triton and it is blazing fast. it was actually more easier than the layer normalization to implement in triton.


![Softmax Performance](Triformer/images/softmax.png)


## Future Plans - To Do
- [ ] Create a library specifically for transformers in vision and language
- [x] Implement the layernorm in Triton 
- [x] Implement the softmax in Triton 
- [ ] Make the TritonLinear more flexible to either use relu or not
- [ ] Fuse the kernels of TritonLinear to see if it speeds up the training process 

