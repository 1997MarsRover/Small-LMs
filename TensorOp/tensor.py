import numpy as np
from dataclasses import dataclass

@dataclass
class Tensor:
    data: np.ndarray
    requires_grad: bool = False
    grad: np.ndarray = None

    def __add__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        else:
            return Tensor(self.data + other, requires_grad=self.requires_grad)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)
        else:
            return Tensor(self.data * other, requires_grad=self.requires_grad)

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

