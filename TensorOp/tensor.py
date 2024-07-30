import numpy as np
from dataclasses import dataclass

@dataclass
class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.grad = None
        self.shape = self.data.shape
        self.dtype = self.data.dtype

    def __add__(self, other):
        if isinstance(other, (int, float)):  # scalar addition
            other = Tensor(other, requires_grad=self.requires_grad)
        if self.shape != other.shape:
            raise ValueError("Tensors must have the same shape for addition")
        return Tensor([a + b for a, b in zip(self.data, other.data)], requires_grad=self.requires_grad)

    def __mul__(self, other):
        if isinstance(other, (int, float)):  # scalar multiplication
            other = Tensor(other, requires_grad=self.requires_grad)
        if len(self.shape) == 2 and len(other.shape) == 2:  # matrix multiplication
            if self.shape[1] != other.shape[0]:
                raise ValueError("Incompatible shapes for matrix multiplication")
            result_data = [[sum(a * b for a, b in zip(row, col)) for col in zip(*other.data)] for row in self.data]
            return Tensor(result_data, requires_grad=self.requires_grad)
        elif self.shape == other.shape:  # element-wise multiplication
            return Tensor([a * b for a, b in zip(self.data, other.data)], requires_grad=self.requires_grad)
        else:
            raise ValueError("Incompatible shapes for multiplication")
        
    def backward(self):
        if self.requires_grad:
            if self.grad is None:
                self.grad = np.ones_like(self.data)
            else:
                raise ValueError("Gradient already computed")

    def __str__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def __repr__(self):
        return self.__str__()

    def zero_grad(self):
        if self.requires_grad:
            self.grad = None

