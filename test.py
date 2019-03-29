import torch

# x = torch.rand(1, 256, 32, 32)
# x = torch.nn.AvgPool2d(kernel_size=32)(x)

# print(x.size())

x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(torch.max(x,1))
# print('x size: ', x.size())
# print('x size: ', x.size(0))
# print(x, type(x))
# x = x.sum()
# print(x, type(x))
# x = x.item()
# print(x, type(x))
