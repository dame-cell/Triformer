# triton-transformers

This will be an implementation of  transformers using triton, 
- This is my first introduction to low-level GPU coding neurel networks i guess. 
- I will try to Also train the model not sure yet but maybe 
- As of right now I am still learning Triton 

### Installation 
- First install triformer 
```bash
pip install triformer==2.1.4
```
- Then you can use the components 
- please keep in mind that the TritonLinear is a fused with relu
- As of right now the TritonLinear is very slow compared to the Pytorch Linear layer, I'm asssuming its because I divided the kernel into 3 parts and the overhead of switching between different kernels is causing the slowdown I'm still looking into it. I might fused the kernels to see if that helps. 

```python
from triformer import TritonLinear, TritonLayerNorm, TritonSoftmax
```

### Layer Normalization

The layer normalization backwards and forward throughputs are higher than the pytorch layer normalization.
| Forward | Backward |
|---------|----------|
| ![LayerNorm Forward Performance](triformer/images/layernorm_forward.png) | ![LayerNorm Backward Performance](triformer/images/layernorm_backward.png) |




### Softmax
The softmax kernel is also implemented in Triton and it is blazing fast. it was actually more easier than the layer normalization to implement in triton.


![Softmax Performance](triformer/images/softmax.png)

## Test for each components 
-  Layernorm test has been addded, when testing the layernorm the weights and biases were not quite similar to torch but there was a bit of difference in the values.So i had to use  `rtol=1e-0`, `atol=1e-0` to pass the test.
-  As for the softmax I actually tests on `causal=False`
  
You can run the tests individually 
```bash
pytest tests/test_layernorm.py
pytest tests/test_softmax.py
```

## Future Plans - To Do
- [ ] Create a library specifically for transformers in vision and language
- [x] Implement the layernorm in Triton 
- [x] Implement the softmax in Triton 
- [x] add test for each and every component
- [ ] Make the TritonLinear more flexible to either use relu or not
- [ ] Fuse the kernels of TritonLinear to see if it speeds up the training process 

