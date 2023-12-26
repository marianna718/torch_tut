import torch

x = torch.ones(5,2,dtype= torch.int)
y = torch.tensor([[1,2,3,1,3],[0,0,0,0,0]])
print(x.dtype)
# z = x*y
# print(z)
i = torch.rand(5,3)
print(i)
print(i[1,:]) #row 1 and all the columns
# to resize the tensor 
tens = torch.rand(4,4)
n = tens.view(-1,8)
# -1 means resize , and pytorch will automaticly discide  
# the value of second dimention
print(n)

# converting from numpy to torch
import numpy as np

a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
# now if we change one we will change the other too 
# becouse they both point to the same memory value , and 
# this will happen if they both are on cpu

# now lets try the oposit
c = np.ones(5)
b = torch.from_numpy(a)
print(b)
# now they share memory too
# this happens when they are on gpu
# so we can check if the gpu is available

if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device)
    # so now this tensor is created on gpu
    # or we can basicaly move the operations to gpu
    y = torch.ones(5)
    y = y.to(device)
    z = x + y
    # this will happen on gpu
    # but the following will return an error 
    z.numpy()
    # this will return error becouse numpy can only handle cpu 
    # tensors , to solve this we need to move it back to cpu
    # like this
    z = z.to("cpu")

#  a lot of times we will use next argument in creating 
    # tensors , it is the --> requires_grad = True
    # the use of it , is for future when for the tensor we are going dto 
    # calculate its gradient
gr = torch.ones(5, requires_grad=True) 
# by deffulte its false