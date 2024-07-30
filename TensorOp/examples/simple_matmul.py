from tensor import Tensor
import numpy as np 

x = Tensor([[5, 7], [4, 9]], requires_grad=True)
y = Tensor([[2.0,0], [-5,-2.0]], requires_grad=True)
z = x * y
z.backward()

print(z)