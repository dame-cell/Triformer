# triton-transformers

This will be an implementation of  transformers using triton, 
- This is my first introduction to low-level GPU coding neurel networks i guess. 
- I will try to Also train the model not sure yet but maybe 
- As of right now I am still learning Triton 

## The plan  
my plan is to start with simple neurel network and then move upwards to transformers 

# Triton Kernel Development: Essential Steps and Concepts

## 1. Defining the Kernel Function

- Use the `@triton.jit` decorator to define the kernel.
- Specify input parameters, such as pointers to input/output tensors and metadata like tensor shapes and strides.

## 2. Compute Program ID and Offsets

- **Program ID**: `pid = tl.program_id(axis)`
  - Identifies which part of the tensor the thread block is processing.

- **1D Offset**: 
  ```python
  offset = pid * BLOCK_SIZE
  
## 3. Load Input Data

### Creating ranges:

```python
indices = tl.arange(0, size)
```

### 4. Computing memory pointers:
```python
ptr = base_ptr + offset_m * stride_m + offset_n * stride_n
```
Computes the exact memory address for each thread to load or store data.

### 5. Loading data:
```python
data = tl.load(ptr, mask=mask)
```
Loads data from global memory into registers for computation.

### 6. Performing Computations
Implement the core algorithm using Triton's operations on loaded data.

### 7. Storing Results
Storing data:
```python
tl.store(ptr, data, mask=mask)
```
Writes computed results back to global memory.

