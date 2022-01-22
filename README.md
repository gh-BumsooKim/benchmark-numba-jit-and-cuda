# Benchmark numba jit and cuda
Benchmarking test about acceleration of numpy and parallel processing in gpu using Numba Compiler

### Environment

- OS : Windows 10 64bit
- CPU : Intel i9-10900
- GPU : Nvidia RTX 3080 10GB
- Program Lang : Python 3.9.7
- Compiler : Numba 0.54 for Numpy and CUDA
- Measuring : timeit module

### Test List

- Optimal Compile in Local **CPU** with jit
- Concurrency in **GPU** with cuda

### List

| Function | Data | Numpy | Numba using Jit <br> (Excluding compile time) | CUDA | Code |
|:---:|:---:|:---:|:---:|:---:|:---:|
| numpy.tanh | 1e4 by 1e4 matrix | 111 µs ± 569 ns per loop | 3.41 µs ± 52.4 ns per loop | . | . |
| numpy.multiply | 

### Measuring 

<hr>

Optimized Compilation in CPU :
```python
from numba import jit
import numpry as np

@jit
def func(x):
  return x
  
# Compile
func(1)

# Test
%timeit func(x)
```

GPU Acceleration
```python
from numba import cuda

@cuda.jit
def func(x, y, out):
  idx = cuda.grid(1) # or 2 in 2-d data
  
  y[idx] = x[idx]

block, thread = 32, 256
x_device = cuda.to_device(x)
y_device = cuda.to_device(y)

# Test
%timeit func[block, thread](x_device, y_device); cuda.synchronize()

x_host = x_device.copy_to_host()
```
