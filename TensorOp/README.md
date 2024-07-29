## tensor Operations

But what is a Tensor really? a tensor is a -dimensional matrix. Okay, but what is a torch.Tensor? Specifically, what actually happens when the following piece of code is run: a = torch.tensor(1.0, requires_grad=True). It turns out that PyTorch will allocate the data on the heap and returns the pointer to that data as a shared pointer 1

