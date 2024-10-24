# triton-transformers

This will be an implementation of  transformers using triton, 
- This is my first introduction to low-level GPU coding neurel networks i guess. 
- I will try to Also train the model not sure yet but maybe 
- As of right now I am still learning Triton 

### Installation 
- First install triformer 
```bash
pip install triformer==0.0.1
```
- Then you can use the components 
- please keep in mind that the TritonLinear is a fused with relu
```python
from triformer import TritonLinear
class TritonMLP(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=256):
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

## Training Metrics for TritonMLP on the CIFAR10 Dataset
![image](utils/images/training_metrics2.png)


## Future Plans - To Do
- [ ] Create a library specifically for transformers in vision and language
- [ ] Make the TritonLinear more flexible to different activation functions
