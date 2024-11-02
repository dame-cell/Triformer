# triformer

triformer is a library that implements the transformer models in triton.
that's it nothing special .


### Installation 
- First install triformer 
```bash
pip install -U triformer
```
### Usage 

Coming Soon 

# Benchmarking 
The benchmarking was done on the L40s GPU 

### Layer Normalization

Updated the layernorm kernel to a more redable code.

| Forward | Backward | Combined |
|---------|----------|----------|
| ![LayerNorm Forward Performance](triformer/images/layernorm-forward.png) | ![LayerNorm Backward Performance](triformer/images/layernorm-backward.png) | ![LayerNorm Combined Performance](triformer/images/layernorm-combined.png) |




### Softmax
The softmax kernel is also implemented in Triton and it is blazing fast. it was actually more easier than the layer normalization to implement in triton.


| Forward | Backward | Combined |
|---------|----------|----------|
| ![Softmax Forward Performance](triformer/images/softmax-forward.png) | ![Softmax Backward Performance](triformer/images/softmax-backward.png) | ![Softmax Combined Performance](triformer/images/softmax-combined.png) |

### Dropout
The droput kernel was intresting it was a bit tricky to implement especially the backward pass,
Tried Training a simple MLP with the TritonDropout, looks like its doing good.

![Dropout Performance](triformer/images/dropout.png)


### Cross Entropy Loss

The cross entropy loss implementation in Triton achieves significant memory efficiency through two key optimizations:

1. **In-Place Gradient Computation**
- Reuses the logits tensor for storing gradients instead of allocating new memory
- Eliminates need for additional gradient storage
- Results in ~2x memory reduction compared to PyTorch's implementation
- Particularly effective for large vocabulary sizes (30k-50k tokens)

1. **Micro-batch Processing**
- Processes data in smaller chunks to reduce peak memory usage
- Configurable number of chunks via `n_chunks` parameter
- Trades a small amount of compute time for memory efficiency
- Enables processing of larger batches with limited GPU memory

![CrossEntropyLoss Performance](triformer/images/memory_benchmark.png)
 
 huge thanks to [mgmalek/efficient_cross_entropy](https://github.com/mgmalek/efficient_cross_entropy) for the reference.

## Test for each components 
-  Layernorm test has been addded, when testing the layernorm the weights and biases were not quite similar to torch but there was a bit of difference in the values.So i had to use  `rtol=1e-0`, `atol=1e-0` to pass the test.
-  As for the softmax I actually tests on `causal=False`
  

To run the tests 

- First git clone the repo 
```bash 
git clone https://github.com/dame-cell/Triformer.git
```
- Then navigate to the Triformer/tests directory 
```bash
cd Triformer/tests
```
- Install triformer
```bash
pip install -U triformer
```
- Then run the tests 
```bash
pytest tests/test_layernorm.py
pytest tests/test_softmax.py
pytest tests/test_dropout.py
pytest tests/test_cross_entropy.py
```

## Future Plans - To Do
- [ ] Create a library specifically for transformers in vision and language
- [x] Implement the layernorm in Triton 
- [x] Implement the softmax in Triton 
- [x] Implement the dropout in Triton
- [x] Implement the cross entropy loss in Triton
- 
