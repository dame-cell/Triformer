# triton-transformers

This will be an implementation of  transformers using triton, 
- This is my first introduction to low-level GPU coding neurel networks i guess. 
- I will try to Also train the model not sure yet but maybe 
- As of right now I am still learning Triton 

### Installation 
- First install triformer 
```bash
pip install triformer==1.3.0
```
- Then you can use the components 
- please keep in mind that the TritonLinear is a fused with relu
  
```python
from triformer import TritonLinear
class TritonMLP(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=768):
        super(TritonMLP, self).__init__()
        self.fc1 = TritonLinear(input_size, hidden_size,use_relu=True)
        self.fc2 = TritonLinear(hidden_size, hidden_size*2,use_relu=True)
        self.fc3 = TritonLinear(hidden_size*2, num_classes,use_relu=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
```

### Try it out!

You can try out the TritonMLP on CIFAR10 dataset using this Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tupdi2hgIEY9zSZ9N47LmdUmbn3IE9pO?usp=sharing)



## Future Plans - To Do
- [ ] Create a library specifically for transformers in vision and language
- [ ] Make the TritonLinear more flexible to different activation functions
