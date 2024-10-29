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
from triformer import TritonLinear, TritonLayerNorm, TritonSoftmax
```

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

