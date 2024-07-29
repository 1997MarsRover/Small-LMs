## tensor Operations

But what is a Tensor really? a tensor is a -dimensional matrix. Okay, but what is a torch.Tensor? Specifically, what actually happens when the following piece of code is run: a = torch.tensor(1.0, requires_grad=True). It turns out that PyTorch will allocate the data on the heap and returns the pointer to that data as a shared pointer 1

What happens when you create a tensor in using torch.tensor? 

Lets break down what happens when you create a tensor using `torch.tensor`:

`a = torch.tensor(1.0, requires_grad=True)`

1. **Memory Allocation**: PyTorch allocates memory on the **heap** to store the tensor data. The heap is a region of memory where dynamic memory allocation takes place.

2. **Data Storage**: The tensor data (in this case, the value `1.0`) is stored in the allocated memory.

3. **Metadata Creation**: PyTorch creates metadata for the tensor, which includes information such as:
	* **Data Type**: The data type of the tensor (e.g., float, int, etc.).
	* **Shape**: The shape of the tensor (in this case, a scalar).
	* **Stride**: The stride of the tensor (not applicable for scalars).
	* **Requires Grad**: A flag indicating whether the tensor requires gradient computation (in this case, `True`).

4. **Tensor Object Creation**: A `torch.Tensor` object is created, which wraps the allocated memory and metadata.

5. **Shared Pointer**: The `torch.Tensor` object is returned as a **shared pointer** to the allocated memory. This allows multiple references to the same tensor without copying the data.

6. **Registration in PyTorch's Tensor Registry**: The tensor is registered in PyTorch's internal tensor registry, which keeps track of all tensors in the system.

7. **Gradient Buffer Allocation**: Since `requires_grad=True`, PyTorch allocates a gradient buffer to store the gradients of the tensor during backpropagation.


## lets implement a Tensor

```
class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None

    def __str__(self):
        return f'Tensor(data={self.data}, requires_grad={self.requires_grad})'
```

This `Tensor` class has an `__init__` method that takes in `data` and an optional `requires_grad` flag. It also has a `__str__` method to provide a nice string representation of the tensor.

Now, let's add some methods to implement basic tensor operations, such as addition and multiplication.You've got a good start on your notes about tensor operations and implementing a basic `Tensor` class. Here's a curt yet helpful response to get you moving forward:

**Next steps:**

1. Implement `__add__` and `__mul__` methods for tensor addition and multiplication.
2. Consider adding a `backward` method to compute gradients when `requires_grad` is `True`.
3. You might want to add more metadata, such as `shape` and `dtype`, to your `Tensor` class.
4. Think about how you'll handle tensor operations with broadcasting and scalar values.

**Questions to consider:**

1. How will you handle tensor operations with different data types (e.g., float, int)?
2. How will you implement more complex tensor operations, such as matrix multiplication and convolution?
3. How will you optimize your tensor operations for performance?

