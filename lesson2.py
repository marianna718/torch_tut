# autograd and creating gradients
import torch 

# lets say we letter want to calculate the gradient of a 
# function with respect to x , then the arg resquiers_grad = True
x = torch.rand(3, requires_grad=True)
print(x)
# now whenever we use this tensor python will create so called camputational graph for use\

y = x + 2
# basically it will be stored like some kind of neuron , 
# which will look like input1 x and input2 2 and computation wich is plus 
# and the output y , and by this we can then do backpropogation if needed from y
# becouse the y will now have attribute called grad_fn()

print(y)

z = y*y*2
print(z)
# now the z has so called MulBackward attribute
# and if we do 
z = z.mean()
# then our k will have MeanBackward attribute
# print(k)

# and now to calculate any gradient we just will
z.backward()
# k.backward()
print(x.grad)
# in the backward of it it will create jacobian function to do the calculations
#  and becouse its a jacobian function then it needs a vector to multiplie with
# in our case we have used the mean() function , wich is a scalar function 
# but if we dont have it then we need to specify a vector with the same size 
# like this 
v = torch.tensor([0.1, 1.0, 0.01], dtype=torch.float32)

# and then pass it as a argument like this
# z.backward(v)

# -------NOW LETS SEE HOW TO PREVENT PYTHON FROM 
# TRACKING HISTORY AND CALCULATIO=NG GRAD_FN 
# THERE ARE 3 OPTIONS
#   x.requieres_grad_(False)
a = torch.rand(5, requires_grad=True)
print(a.requires_grad_(False))
#   x.detach()
a = torch.rand(5, requires_grad=True)
b = a.detach()
print(a, "and now without", b)
#   with torch.no_grad(): 
    # and then what we need to do
a = torch.rand(5, requires_grad=True)
with torch.no_grad():
    c = a + 2
    print("this is the c", c)
# lifehack for pytorch , whenever the function in pytorch got 
# an underscore at the end d, that means that the function is applieing the 
# changes in place

# DUMMY NETWORK FOR LEARNING

weights = torch.ones(4 , requires_grad=True)

for epoch in range(5):
    model_output = (weights*3).sum()

    # and now we want to calculate the gradients
    model_output.backward()

    print(weights.grad)
    # with this our gradients are getting sumed up , so we need to zero them
    weights.grad.zero_()